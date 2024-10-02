// This file contains the main rendering code. Unlike the official implementation,
// instead of using compute shaders and iterating through (possibly) all gaussians,
// we instead use a vertex shader to turn each gaussian into a quad facing the camera
// and then use the fragment shader to paint the gaussian on the quad.
// If we draw the quads in order of depth, with well chosen blending settings we can
// get the same color accumulation rule as in the original paper.
// This approach is faster than the original implementation on webGPU but still substantially
// slow compared to the CUDA impl. The main bottleneck is the sorting of the quads by depth,
// which is done on the CPU but could presumably be replaced by a compute shader sort.

import { PackedGaussians } from './ply';
import { f32, Struct, vec3, mat4x4 } from './packing';
import { InteractiveCamera } from './camera';
import { getShaderCode , getInitSortBufferCode } from './shaders';
import { mat4, Mat4, Vec3 } from 'wgpu-matrix';
import { GpuContext } from './gpu_context';
import { DepthSorter } from './depth_sorter';
import { RadixSortKernel } from 'webgpu-radix-sort';
import { SimpleRender } from './simple_render';


const uniformLayout = new Struct([
    ['viewMatrix', new mat4x4(f32)],
    ['projMatrix', new mat4x4(f32)],
    ['cameraPosition', new vec3(f32)],
    ['tanHalfFovX', f32],
    ['tanHalfFovY', f32],
    ['focalX', f32],
    ['focalY', f32],
    ['scaleModifier', f32],
]);

function mat4toArrayOfArrays(m: Mat4): number[][] {
    return [
        [m[0], m[1], m[2], m[3]],
        [m[4], m[5], m[6], m[7]],
        [m[8], m[9], m[10], m[11]],
        [m[12], m[13], m[14], m[15]],
    ];
}

export class Renderer {
    canvas: HTMLCanvasElement;
    interactiveCamera: InteractiveCamera;
    numGaussians: number;

    context: GpuContext;
    contextGpu: GPUCanvasContext;

    uniformBuffer: GPUBuffer;
    pointDataBuffer: GPUBuffer;
    drawIndexBuffer: GPUBuffer;
    // Sort
    sort_key_buffer : GPUBuffer;
    sort_value_buffer : GPUBuffer;
    
    radixSortKernel ;
    //depthSorter: DepthSorter;

    initSortBindGroup: GPUBindGroup;
    uniformsBindGroup: GPUBindGroup;
    pointDataBindGroup: GPUBindGroup;

    init_sort_pipeline:GPUComputePipeline;
    drawPipeline: GPURenderPipeline;

    depthSortMatrix: number[][];

    // fps counter
    fpsCounter: HTMLLabelElement;
    lastDraw: number;

    simple_render : SimpleRender ; //dummy renderer

    destroyCallback: (() => void) | null = null;

    public static async requestContext(gaussians: PackedGaussians): Promise<GpuContext> {
        const gpu = navigator.gpu;
        if (!gpu) {
            return Promise.reject("WebGPU not supported on this browser! (navigator.gpu is null)");
        }

        const adapter = await gpu.requestAdapter();
        if (!adapter) {
            return Promise.reject("WebGPU not supported on this browser! (gpu.adapter is null)");
        }

        // for good measure, we request 1.5 times the amount of memory we need
        const byteLength = gaussians.gaussiansBuffer.byteLength;
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: 1.5 * byteLength,
                maxBufferSize: 1.5 * byteLength,
            }
        });

        return new GpuContext(gpu, adapter, device);
    }

    // destroy the renderer and return a promise that resolves when it's done (after the next frame)
    public async destroy(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.destroyCallback = resolve;
        });
    }

    constructor(
        canvas: HTMLCanvasElement,
        interactiveCamera: InteractiveCamera,
        gaussians: PackedGaussians,
        context: GpuContext,
        fpsCounter: HTMLLabelElement,
    ) {
        this.canvas = canvas;
        this.interactiveCamera = interactiveCamera;
        this.context = context;
        const contextGpu = canvas.getContext("webgpu");
        if (!contextGpu) {
            throw new Error("WebGPU context not found!");
        }
        this.contextGpu = contextGpu;
        this.fpsCounter = fpsCounter;
        this.lastDraw = performance.now();

        this.numGaussians = gaussians.numGaussians;

        const presentationFormat = "rgba16float" as GPUTextureFormat;

        this.contextGpu.configure({
            device: this.context.device,
            format: presentationFormat,
            alphaMode: 'premultiplied' as GPUCanvasAlphaMode,
        });
        //===========================================
        //             Point Buffer
        //===========================================
        this.pointDataBuffer = this.context.device.createBuffer({
            size: gaussians.gaussianArrayLayout.size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
            label: "renderer.pointDataBuffer",
        });
        new Uint8Array(this.pointDataBuffer.getMappedRange()).set(new Uint8Array(gaussians.gaussiansBuffer));
        this.pointDataBuffer.unmap();

        // Create a GPU buffer for the uniform data.
        this.uniformBuffer = this.context.device.createBuffer({
            size: uniformLayout.size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.uniformBuffer",
        });

        const shaderCode = getShaderCode(
            canvas, 
            gaussians.sphericalHarmonicsDegree,
            gaussians.nShCoeffs,            
            this.canvas.width,
            this.canvas.height ,
        );

        console.log(this.interactiveCamera.getCamera());
        const shaderModule = this.context.device.createShaderModule({ code: shaderCode });

        //===========================================
        //              Pipeline
        //===========================================
        const draw_uniform_bindinglayout = this.context.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'uniform',
                    },
                },              
            ],
        });       
        const draw_data_bindinglayout = this.context.device.createBindGroupLayout({
            label:"draw_data_bindinglayout",
            entries: [
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'read-only-storage',
                    },
                },              
                {
                    binding: 2,                    
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: 'read-only-storage',
                    },
                },              
            ],
        });      
        const draw_pipeline_layout= this.context.device.createPipelineLayout({
            label: "draw_pipeline_layout ",
            bindGroupLayouts : [draw_uniform_bindinglayout,draw_data_bindinglayout],
        });
        this.drawPipeline = this.context.device.createRenderPipeline({
            //layout: "auto",            
            layout : draw_pipeline_layout,
            vertex: {
                module: shaderModule,
                entryPoint: "vs_points",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fs_main",
                targets: [
                    {
                        format: presentationFormat,
                        // with one-minus-dst alpha we can set the src to src.alpha * src.color and
                        // we get that color_new = src.color * src.alpha + dst.color * (1 - src.alpha)
                        // which is the same as the accumulation rule in the paper
                        blend: {
                            color: {
                                srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,
                                dstFactor: "one" as GPUBlendFactor,
                                operation: "add" as GPUBlendOperation,
                            },
                            alpha: {
                                srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,
                                dstFactor: "one" as GPUBlendFactor,
                                operation: "add" as GPUBlendOperation,
                            },
                        }
                    },
                ],
            },
            primitive: {
                topology: "triangle-list",
                //topology: "point-list",
                //topology: "line-list",
                stripIndexFormat: undefined,
                cullMode: undefined,
            },
        });
        //===========================================
        //              Binding group
        //===========================================
        // key_buffer , value buffer
        this.sort_key_buffer = this.context.device.createBuffer({
			size: this.numGaussians * 4,
			usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST| GPUBufferUsage.COPY_SRC,
			mappedAtCreation: false,
		});
        this.sort_value_buffer = this.context.device.createBuffer({
			size: this.numGaussians * 4,
			usage:  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST| GPUBufferUsage.COPY_SRC,
			mappedAtCreation: false,
		});

        // Drawing pass
        this.uniformsBindGroup = this.context.device.createBindGroup({
            //layout: this.drawPipeline.getBindGroupLayout(0),
            layout: draw_uniform_bindinglayout,
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.uniformBuffer,
                },
            }],
        });

        this.pointDataBindGroup = this.context.device.createBindGroup({
            //layout: this.drawPipeline.getBindGroupLayout(1),
            layout : draw_data_bindinglayout,
            entries: [{
                binding: 1,
                resource: {
                    buffer: this.pointDataBuffer,
                },
            },{
                binding: 2,
                resource: {
                    buffer: this.sort_key_buffer,
                },
            }],
        });

        /*
        this.depthSorter = new DepthSorter(this.context, gaussians);
        this.drawIndexBuffer = this.context.device.createBuffer({
           size: 6 * 4 * gaussians.numGaussians,
           usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
           mappedAtCreation: false,
           label: "renderer.drawIndexBuffer",
        });
        */
        //===========================================
        //             Radix Sorter
        //===========================================
        
        this.radixSortKernel = new RadixSortKernel({
            device:  this.context.device,                   // GPUDevice to use
            keys: this.sort_value_buffer,                 // GPUBuffer containing the keys to sort
            values: this.sort_key_buffer,             // (optional) GPUBuffer containing the associated values
            count: this.numGaussians ,               // Number of elements to sort
            check_order: true,               // Whether to check if the input is already sorted to exit early
            bit_count: 32,                    // Number of bits per element. Must be a multiple of 4 (default: 32)
            workgroup_size: { x: 16, y: 16 }, // Workgroup size in x and y dimensions. (x * y) must be a power of two
        })
        console.log(this.radixSortKernel);
        // Init rasix sort pipeline
        const sort_shaderModule =  this.context.device.createShaderModule({
            code: getInitSortBufferCode(this.numGaussians , gaussians.nShCoeffs),
        });


        const init_sort_bindinglayout = this.context.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'uniform',
                    },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'storage',
                    },
                },
                
            ],
        });        
        this.initSortBindGroup = this.context.device.createBindGroup({
            layout: init_sort_bindinglayout,
            label: "initSortBindGroup",
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.sort_key_buffer,
                },
            },{
                binding: 1,
                resource: {
                    buffer: this.sort_value_buffer,
                },
            },{
                binding: 2,
                resource: {
                    buffer: this.uniformBuffer,
                },
            },{
                binding: 3,
                resource: {
                    buffer: this.pointDataBuffer,
                },
            }],
        });

        const init_sort_pipeline_layout= this.context.device.createPipelineLayout({
            bindGroupLayouts: [init_sort_bindinglayout],
        });
        this.init_sort_pipeline  = this.context.device.createComputePipeline({
            layout: init_sort_pipeline_layout, 
            compute: {
                module: sort_shaderModule,
                entryPoint: "main",
            }
        });
        

        const indices = new Uint32Array([0, 1, 2, 1, 3, 2,]);
		this.drawIndexBuffer = this.context.device.createBuffer({
			size: indices.byteLength,
			usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true,
		});
		new Uint32Array(this.drawIndexBuffer.getMappedRange()).set(indices);
		this.drawIndexBuffer.unmap();


        this.simple_render= new SimpleRender(context , canvas , 
            this.pointDataBuffer,
            this.uniformBuffer,
            this.sort_key_buffer); //dummy 
        // start the animation loop
        requestAnimationFrame(() => this.animate(true));
    }

    private destroyImpl(): void {
        if (this.destroyCallback === null) {
            throw new Error("destroyImpl called without destroyCallback set!");
        }

        this.uniformBuffer.destroy();
        this.pointDataBuffer.destroy();
        this.drawIndexBuffer.destroy();
        //this.depthSorter.destroy();
        this.context.destroy();
        this.destroyCallback();
    }

    draw(nextFrameCallback: FrameRequestCallback): void {                
        const init_encoder = this.context.device.createCommandEncoder();
        const cs_initSortBuffer_pass = init_encoder.beginComputePass();
        cs_initSortBuffer_pass.setPipeline(this.init_sort_pipeline);
        cs_initSortBuffer_pass.setBindGroup(0 , this.initSortBindGroup);
        cs_initSortBuffer_pass.dispatchWorkgroups(Math.max(this.numGaussians/8 ,8) , 1,1);
        cs_initSortBuffer_pass.end();

        this.context.device.queue.submit([init_encoder.finish()])
        
        const commandEncoder = this.context.device.createCommandEncoder();        
        const sort_pass = commandEncoder.beginComputePass()
        this.radixSortKernel.dispatch(sort_pass) // Sort keysBuffer and valuesBuffer in-place on the GPU
        sort_pass.end()
        this.context.device.queue.submit([commandEncoder.finish()])
        

        //==========
        //     Test        
        //==========
        

        /*
        const _data_size = 4;
		const _buffer_size = this.numGaussians * _data_size;
		//const buffer = this.simple_render.vertexBuffer;       
		const buffer = this.sort_key_buffer;       

		const readBuffer = this.context.device.createBuffer({			
			size: _buffer_size,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
			mappedAtCreation: false,
			label: "read back buffer"
		});
		const _cmdPass = this.context.device.createCommandEncoder();
        
		_cmdPass.copyBufferToBuffer(buffer, 0, readBuffer, 0, _buffer_size);
		this.context.device.queue.submit([ _cmdPass.finish()]);
				
		readBuffer.mapAsync(GPUMapMode.READ).then(() => {
			const result = new Uint32Array(readBuffer.getMappedRange());
			
			console.log(result);
			readBuffer.unmap();
		});

        */

       this.simple_render.draw(this.numGaussians);
       /*       
        const RenderEncoder = this.context.device.createCommandEncoder();  
        
        const textureView = this.contextGpu.getCurrentTexture().createView();
        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                storeOp: "store" as GPUStoreOp,
                loadOp: "clear" as GPULoadOp,
                }],
                };
                
        const passEncoder = RenderEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.drawPipeline);
        
        passEncoder.setBindGroup(0, this.uniformsBindGroup);
        passEncoder.setBindGroup(1, this.pointDataBindGroup);
        
        passEncoder.setIndexBuffer(this.drawIndexBuffer, "uint32" as GPUIndexFormat)        
        //passEncoder.drawIndexed(this.numGaussians * 6, 1, 0, 0, 0);
        passEncoder.drawIndexed( 6, this.numGaussians);
        
        passEncoder.end();
        
        this.context.device.queue.submit([RenderEncoder.finish()]);
        */
        
        // fps counter
        const now = performance.now();
        const fps = 1000 / (now - this.lastDraw);
        this.lastDraw = now;
        this.fpsCounter.innerText = 'FPS: ' + fps.toFixed(2);
        this.fpsCounter.style.display = 'block';
        requestAnimationFrame(nextFrameCallback);
    }

    animate(forceDraw?: boolean) {
        if (this.destroyCallback !== null) {
            this.destroyImpl();
            return;
        }
        if (!this.interactiveCamera.isDirty() && !forceDraw) {
            requestAnimationFrame(() => this.animate());
            return;
        }
        const camera = this.interactiveCamera.getCamera();

        const position = camera.getPosition();

        const tanHalfFovX = 0.5 * this.canvas.width / camera.focalX;
        const tanHalfFovY = 0.5 * this.canvas.height / camera.focalY;

        this.depthSortMatrix = mat4toArrayOfArrays(camera.viewMatrix);
        console.log(camera);
        let uniformsMatrixBuffer = new ArrayBuffer(this.uniformBuffer.size);

        let viewMat = mat4toArrayOfArrays(mat4.transpose(camera.viewMatrix))
        let projMat = mat4toArrayOfArrays(mat4.transpose(camera.perspective))
        console.log( viewMat);        
        console.log( projMat);
        let uniforms = {
            viewMatrix: mat4toArrayOfArrays(camera.viewMatrix),
            //projMatrix: mat4toArrayOfArrays(camera.getProjMatrix()),
            projMatrix: mat4toArrayOfArrays( camera.perspective),
            cameraPosition: Array.from(position),
            tanHalfFovX: tanHalfFovX,
            tanHalfFovY: tanHalfFovY,
            focalX: camera.focalX,
            focalY: camera.focalY,
            scaleModifier: camera.scaleModifier,

        };
        uniformLayout.pack(0, uniforms, new DataView(uniformsMatrixBuffer));

        this.context.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            uniformsMatrixBuffer,
            0,
            uniformsMatrixBuffer.byteLength
        );

        this.draw(() => this.animate());
    }
}