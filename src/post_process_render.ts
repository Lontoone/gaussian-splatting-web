import { GpuContext } from "./gpu_context";

export class PostProcessRenderer{
	
	context :GpuContext;
	canvas: HTMLCanvasElement;
	contextGpu: GPUCanvasContext;
	vbo : GPUBuffer;
	post_process_pipeline :GPURenderPipeline;
	
	bindGroup :GPUBindGroup;	
	//texutreView : GPUTextureView;
	texture:GPUTexture;

	constructor(_context:GpuContext , _canvas:HTMLCanvasElement , texture:GPUTexture ){
		this.context = _context;
		this.canvas = _canvas;		
		this.texture = texture;
		const contextGpu = _canvas.getContext("webgpu");
        if (!contextGpu) {
            throw new Error("WebGPU context not found!");
        }
		this.contextGpu = contextGpu;
		const canvasFormat =  'rgba16float';
		const vertices = new Float32Array([
			//   X,    Y,
			  -1, -1,
			  1, -1,
			  1,  1,
			  
			  -1, 1,
			  1,  1,
			  -1,  -1,
		]);

		this.vbo = this.context.device.createBuffer({
			label: "Cell vertices",
			size: vertices.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
		  });

		this.context.device.queue.writeBuffer(this.vbo, /*bufferOffset=*/0, vertices);
		const vertexBufferLayout: GPUVertexBufferLayout= {
			arrayStride: 8,
			attributes: [{
			  format: "float32x2",  // vec2 = 8 bytes
			  offset: 0,
			  shaderLocation: 0, // Position, see vertex shader , between [0,15]
			}],
		};

		const cellShaderModule = this.context.device.createShaderModule({
			label: "Cell shader",
			code: `
			@vertex
			fn vertexMain(@location(0) pos: vec2f) -> @builtin(position) vec4f {
				return vec4f(pos.x, pos.y, 0, 1);
			}

			@group(0) @binding(0) var myTexture: texture_2d<f32>;
			@group(0) @binding(1) var mySampler: sampler;
			@fragment
			fn fragmentMain(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4f {
			    var uv = fragCoord.xy / vec2<f32>(${this.canvas.width}, ${this.canvas.height});
				uv.y = 1-uv.y;
				//return vec4f(uv.x, uv.y, 0, 1);			
				var color:vec4f = textureSample(myTexture, mySampler, uv)  ;

				color.a = saturate(color.a * 1.5);

				if(color.a <0.99){
					discard;
				}
				/*
				let pw = 1.0/2.2;
				color.r= pow(color.r, pw);
				color.g= pow(color.g, pw);
				color.b= pow(color.b, pw);
				*/
				return color;
			}
			`
		});
		const bindGroupLayout = this.context.device.createBindGroupLayout({
			entries: [
				
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					texture: {
						sampleType: 'float',
					},
				},				
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					sampler: {
						type: 'filtering',
					},
				},
			],
		});
		const pipeline_layout= this.context.device.createPipelineLayout({
            label: "Simple draw layout ",
            bindGroupLayouts : [bindGroupLayout],
        });
		this.post_process_pipeline = this.context.device.createRenderPipeline({
			label: "post_process_pipeline",
			layout: pipeline_layout,
			vertex: {
				module: cellShaderModule,
				entryPoint: "vertexMain",
				buffers: [vertexBufferLayout]
			},
			fragment: {
				module: cellShaderModule,
				entryPoint: "fragmentMain",
				targets: [{
					format: canvasFormat
				}]
			}
		});

		const sampler = this.context.device.createSampler({
			label: "mip",
			magFilter: "linear",
			minFilter: "linear",
		  });
		// Step 3: Create a bind group
		

		this.bindGroup = this.context.device.createBindGroup({
			layout: bindGroupLayout,
			entries: [
			  {
				binding: 0,
				resource: this.texture.createView(),
			  },
			  {
				binding: 1,
				resource: sampler,
			  },
			],
		  });
		
		

	}

	public draw():void{
		const textureView = this.contextGpu.getCurrentTexture().createView();
		
		const renderPassDescriptor : GPURenderPassDescriptor = {
			colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0},
                storeOp: "store" as GPUStoreOp,
                loadOp: "clear" as GPULoadOp,
            }],
		};
		const commandEncoder = this.context.device.createCommandEncoder();
		const pass = commandEncoder.beginRenderPass(renderPassDescriptor);

		pass.setPipeline(this.post_process_pipeline);
		pass.setVertexBuffer(0 , this.vbo);
		pass.setBindGroup(0 , this.bindGroup);
		pass.draw(6,2);
		pass.end();

		this.context.device.queue.submit([commandEncoder.finish()]);
		
	}
}