import { GpuContext } from "./gpu_context";

const screen_size = 500.0;  //temp

const shDeg3Code = `
    // spherical harmonic coefficients
    const SH_C0 = 0.28209479177387814f;
    const SH_C1 = 0.4886025119029199f;
    const SH_C2 = array(
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f
    );
    const SH_C3 = array(
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f
    );

    fn compute_color_from_sh(position: vec3<f32>, sh: array<vec3<f32>, 16>) -> vec3<f32> {
        let dir = normalize(position - uniforms.camera_position);
        var result = SH_C0 * sh[0];

        // if deg > 0
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result = result + SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;

        // if (sh_degree > 1) {
        result = result +
            SH_C2[0] * xy * sh[4] +
            SH_C2[1] * yz * sh[5] +
            SH_C2[2] * (2. * zz - xx - yy) * sh[6] +
            SH_C2[3] * xz * sh[7] +
            SH_C2[4] * (xx - yy) * sh[8];
        
        // if (sh_degree > 2) {
        result = result +
            SH_C3[0] * y * (3. * xx - yy) * sh[9] +
            SH_C3[1] * xy * z * sh[10] +
            SH_C3[2] * y * (4. * zz - xx - yy) * sh[11] +
            SH_C3[3] * z * (2. * zz - 3. * xx - 3. * yy) * sh[12] +
            SH_C3[4] * x * (4. * zz - xx - yy) * sh[13] +
            SH_C3[5] * z * (xx - yy) * sh[14] +
            SH_C3[6] * x * (xx - 3. * yy) * sh[15];

        // unconditional
        result = result + 0.5;

        return max(result, vec3<f32>(0.));
    }
`;
function get_simple_shader(){
	return `
	${shDeg3Code}
	const n_sh_coeffs = 16;
	struct PointInput {
            @location(0) position: vec3<f32>,
            @location(1) log_scale: vec3<f32>,
            @location(2) rot: vec4<f32>,
            @location(3) opacity_logit: f32,
            sh: array<vec3<f32>, n_sh_coeffs>,
        };
	struct Uniforms {
            viewMatrix: mat4x4<f32>,
            projMatrix: mat4x4<f32>,
            camera_position: vec3<f32>,
            tan_fovx: f32,
            tan_fovy: f32,
            focal_x: f32,
            focal_y: f32,
            scale_modifier: f32,
        };
	struct PointOutput {
		@builtin(position) position: vec4<f32>,
		@location(0) color: vec3<f32>,
		@location(1) uv: vec2<f32>,
		@location(2) conic_and_opacity: vec4<f32>,
	};

	
	fn sigmoid(x: f32) -> f32 {
		if (x >= 0.) {
			return 1. / (1. + exp(-x));
		} else {
			let z = exp(x);
			return z / (1. + z);
		}
	}

	fn compute_cov3d(log_scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {
		// ------------ CalcMatrixFromRotationScale ----------------
		//let modifier = uniforms.scale_modifier;
		let modifier = 1.0;
		let S = mat3x3<f32>(
			log_scale.x * modifier, 0., 0.,
			0., log_scale.y * modifier, 0.,
			0., 0., log_scale.z * modifier,
		);
		/*
		let r = rot.x;
		let x = rot.y;
		let y = rot.z;
		let z = rot.w;

		let R = mat3x3<f32>(
			1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
			2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
			2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y),
		);
		*/
		let x = rot.x;
		let y = rot.y;
		let z = rot.z;
		let w = rot.w;

		let R = mat3x3<f32>(
			1-2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
          	2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x),
          	2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)
		);

		let M = S * R;

		// ------------ CalcCovariance3D ----------------
		let Sigma = transpose(M) * M;		
		return array<f32, 6>(
			Sigma[0][0],
			Sigma[0][1],
			Sigma[0][2],
			Sigma[1][1],
			Sigma[1][2],
			Sigma[2][2],
		);
	} 

	fn compute_cov2d(position: vec3<f32>, log_scale: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
		let cov3d = compute_cov3d(log_scale, rot);
		let aspect = uniforms.projMatrix[0][0] / uniforms.projMatrix[1][1] ;  // = 1

		var viewPos = uniforms.viewMatrix * vec4<f32>(position, 1.0);

		let tanFovX: f32 = 1.0 / uniforms.projMatrix[0][0];
		let tanFovY: f32 = 1.0 / (uniforms.projMatrix[1][1] * aspect);
		//    	float tanFovX = rcp(matrixP._m00);
    	//		float tanFovY = rcp(matrixP._m11 * aspect);
		//let limx = 1.3 * uniforms.tan_fovx;
		//let limy = 1.3 * uniforms.tan_fovy;

		let screenParams_x = f32(${screen_size}); //TODO;
		
		let limX = 1.3 * tanFovX;
		let limY = 1.3 * tanFovY;
		
		viewPos.x = clamp(viewPos.x / viewPos.z, -limX, limX) * viewPos.z;
    	viewPos.y = clamp(viewPos.y / viewPos.z, -limY, limY) * viewPos.z;
		//viewPos.x = min(limx, max(-limx, txtz)) * t.z;
		//viewPos.y = min(limy, max(-limy, tytz)) * t.z;
		
		//let txtz = t.x / t.z;
		//let tytz = t.y / t.z;

		let focal = screenParams_x * uniforms.projMatrix[0][0] / 2;

		/*
		let J = mat4x4(
			uniforms.focal_x / t.z, 0., -(uniforms.focal_x * t.x) / (t.z * t.z), 0.,
			0., uniforms.focal_y / t.z, -(uniforms.focal_y * t.y) / (t.z * t.z), 0.,
			0., 0., 0., 0.,
			0., 0., 0., 0.,
		);
		*/

		let J = mat3x3(
			focal / viewPos.z, 0.0, -(focal * viewPos.x) / (viewPos.z * viewPos.z),
        	0.0, focal / viewPos.z, -(focal * viewPos.y) / (viewPos.z * viewPos.z),
        	0.0, 0.0, 0.0
		);

		//let W = (mat3x3(uniforms.viewMatrix));	
		let W = mat3x3<f32>(
			uniforms.viewMatrix[0].xyz,
			uniforms.viewMatrix[1].xyz,
			uniforms.viewMatrix[2].xyz
		);
		let T = W * J;

		let Vrk = mat3x3(
			cov3d[0], cov3d[1], cov3d[2],
			cov3d[1], cov3d[3], cov3d[4],
			cov3d[2], cov3d[4], cov3d[5],			
		);

		var cov = transpose(T) * transpose(Vrk) * T;
		

		// Apply low-pass filter: every Gaussian should be at least
		// one pixel wide/high. Discard 3rd row and column.
		cov[0][0] += 0.3;
		cov[1][1] += 0.3;

		return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
	}

	@group(0) @binding(0) var<storage, read> points: array<PointInput>;
	@group(0) @binding(1) var<uniform> 				uniforms: Uniforms;
	@group(0) @binding(2)  var<storage, read> sorted_idx: array<u32>;
		
	@fragment
	fn fs_main(input: PointOutput) ->@location(0) vec4f  {
			//return vec4<f32>(1.0, 0.0, 1.0, 1.0); 			
			//return vec4f(abs(input.uv) , 0 ,1);
			//return vec4f(input.uv , 0 ,1);

			var opacity = input.conic_and_opacity.w;   
			var color : vec4<f32> = vec4<f32> (input.color ,opacity);
			let power :f32 = -dot(input.uv, input.uv);
			var alpha :f32 = exp(power);
			if(opacity>=0){
				alpha = saturate(alpha * opacity );
			}
			if(alpha < 1.0/255.0){
				discard;
			}
				
			//return vec4<f32>(color.rgb, alpha);
			//return vec4f(input.uv , 1 ,1);
			return vec4<f32>(0,0,0, alpha);
			//return vec4<f32>(alpha,alpha,alpha, 1);
		}
		
	@vertex
	fn vs_points(
		@builtin(vertex_index) vtxID: u32,
    	@builtin(instance_index) instID: u32,
		@location(0) pos: vec3f ) -> PointOutput {
		
		let idx = vtxID;
		let p_idx = sorted_idx[instID];
		var point = points[p_idx] ;		
		//var clipPos =  uniforms.projMatrix  * vec4f( point.position , 1);
		var clipPos =  uniforms.projMatrix * uniforms.viewMatrix  * vec4f( point.position , 1);

		var output: PointOutput;
		var quadPos = vec2<f32>(
			f32(idx&1), 
			f32((idx>>1)&1)
			) * 2.0 -1.0 ;
        quadPos *=2;
		

		output.uv  = quadPos;		
        output.position  = clipPos  ;
		// Problem: 當rotation = 0 , cov2d.xyz = 0
		let cov2d = compute_cov2d(point.position, point.log_scale, point.rot);

		//================= DecomposeCovariance =====================
		let diag1 =  cov2d.x;
		let diag2 =  cov2d.z;
		let offDiag =  cov2d.y;				
		
        
        var mid =  0.5 *  (diag1 + diag2);
        var radius = length(vec2<f32>((diag1 - diag2) /2.0  , offDiag));
        var lambda1 = mid + radius;
        var lambda2 = max(mid - radius , 0.1);
        var diagVec : vec2<f32> = normalize(vec2<f32>(offDiag , lambda1 - diag1));
        diagVec.y = -diagVec.y;
        
        let maxSize :f32 = 4096.0;
        //let v1 : vec2<f32> = min(sqrt(2.0 * lambda1) , maxSize) * diagVec;
        let v1 : vec2<f32> = diagVec ;
        let v2 : vec2<f32> = min(sqrt(2.0 * lambda2) , maxSize) * vec2<f32>(diagVec.y , -diagVec.x);

		// TODO: V1,V2 is 0  // lambda1 2 is 0
		let _ScreenParams : vec2<f32> = vec2<f32>(${screen_size},${screen_size});       
		//let deltaScreenPos :vec2<f32> = vec2<f32>(quadPos.x , quadPos.y) * 20 /_ScreenParams.xy;
		let deltaScreenPos :vec2<f32> = vec2<f32>(quadPos.x * v1 + quadPos.y * v2) * 2 /_ScreenParams.xy;
		

		//output.uv = (v1);
        
        output.position  .x += deltaScreenPos.x * clipPos.w;
        output.position  .y += deltaScreenPos.y * clipPos.w;

		output.color = compute_color_from_sh(point.position, point.sh);

		//================= Other ======================
		let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;	
        let det_inv = 1.0 / det;
		
        let conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
        output.conic_and_opacity = vec4<f32>(conic, sigmoid(point.opacity_logit));
		return output;
		//return  uniforms.projMatrix  * uniforms.viewMatrix *vec4f(pos.x, pos.y, pos.z, 1);

	}
	`

}



export class SimpleRender{
	context: GpuContext;
    contextGpu: GPUCanvasContext;
	
	pipeline : GPURenderPipeline;
	vertexBuffer  :GPUBuffer;
	drawIndexBuffer  :GPUBuffer;	
	pointBindGroup: GPUBindGroup;

	constructor(
		_contex:GpuContext,
		_canvas:HTMLCanvasElement,
		_pointBuffer:GPUBuffer,
		_uniformBuffer:GPUBuffer,
		_sortIdxBuffer:GPUBuffer,
	){
		const presentationFormat = "rgba16float" as GPUTextureFormat;
		const contextGpu = _canvas.getContext("webgpu");
        if (!contextGpu) {
            throw new Error("WebGPU context not found!");
        }
        this.contextGpu = contextGpu;
		this.context = _contex;

		// Dummy data:
		const vertices = new Float32Array([
			0.0,  0.5, 0.0,
		   -0.5, -0.5, 0.0,
			0.5, -0.5, 0.0,
		]);
		
		this.vertexBuffer = this.context.device.createBuffer({
			size: vertices.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST| GPUBufferUsage.COPY_SRC,
		});
		this.context.device.queue.writeBuffer(this.vertexBuffer, 0, vertices);
		const vertexBufferLayout : GPUVertexBufferLayout= {
			arrayStride: 12,
			attributes: [{
			  format: "float32x3", 
			  offset: 0,
			  shaderLocation: 0, // Position, 
			}],
		  };
		
		const draw_bindinglayout = this.context.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {type: 'read-only-storage',},
                },              
				{
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {type: 'uniform',},
                },          
				{
                    binding: 2,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {type: 'read-only-storage',},
                },    
            ],
        });       

		this.pointBindGroup  = this.context.device.createBindGroup({
			layout : draw_bindinglayout,
            entries: [
				{
                binding: 0,
                resource: {buffer: _pointBuffer,},
            },
			{
                binding: 1,
                resource: {buffer: _uniformBuffer,},
            },
			{
                binding: 2,
                resource: {buffer: _sortIdxBuffer,},
            },
		]
		});

		
		const draw_pipeline_layout= this.context.device.createPipelineLayout({
            label: "Simple draw layout ",
            bindGroupLayouts : [draw_bindinglayout],
        });

		this.pipeline = this.context.device.createRenderPipeline({
			vertex: {
				module: this.context.device.createShaderModule({
					code: get_simple_shader(),
				}),
				entryPoint: 'vs_points',		
				buffers:[vertexBufferLayout]		
			},
			fragment: {
				module: this.context.device.createShaderModule({
					code: get_simple_shader(),
				}),
				entryPoint: 'fs_main',
				targets: [{
					format: presentationFormat,
					blend: {
							//one-minus-dst-alpha
                            color: {
                                //srcFactor: "src-alpha" as GPUBlendFactor,
                                srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,
                                dstFactor: "one" as GPUBlendFactor,
                                operation: "add" as GPUBlendOperation,
                            },
                            alpha: {
                                //srcFactor: "src-alpha" as GPUBlendFactor,
                                srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,
                                dstFactor: "one" as GPUBlendFactor,
                                operation: "add" as GPUBlendOperation,
                            },
                        }
				}],
			},
			primitive: {
				topology: 'triangle-list',
				//topology: 'point-list',
				//topology: 'line-list',
				stripIndexFormat: undefined,
                cullMode: undefined,
			},
			layout: draw_pipeline_layout
			//layout: "auto"
		});	

		const indices = new Uint32Array([0, 1, 2, 1, 3, 2,]);
		this.drawIndexBuffer = this.context.device.createBuffer({
			size: indices.byteLength,
			usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true,
		});
		new Uint32Array(this.drawIndexBuffer.getMappedRange()).set(indices);
		this.drawIndexBuffer.unmap();
	}


	public draw(gs_number:number) :void{
		const commandEncoder = this.context.device.createCommandEncoder();
		const textureView = this.contextGpu.getCurrentTexture().createView();

		const renderPassDescriptor : GPURenderPassDescriptor = {
			colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                storeOp: "store" as GPUStoreOp,
                loadOp: "clear" as GPULoadOp,
            }],
		};

		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
		passEncoder.setPipeline(this.pipeline);
		passEncoder.setVertexBuffer(0, this.vertexBuffer);		
		passEncoder.setBindGroup(0, this.pointBindGroup);
		//passEncoder.draw(3);
		passEncoder.setIndexBuffer(this.drawIndexBuffer, "uint32" as GPUIndexFormat) 
		passEncoder.drawIndexed( 6, gs_number);
		passEncoder.end();

		this.context.device.queue.submit([commandEncoder.finish()]);

		//console.log("draw");
	}
}