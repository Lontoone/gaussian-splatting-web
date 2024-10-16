import { GpuContext } from "./gpu_context";

const screen_size = 600.0;  //this is only for debug, do not use it

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
function get_simple_shader(width:number , height:number){
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

	fn CalcMatrixFromRotationScale(rot: vec4<f32>, scale: vec3<f32>) -> mat3x3<f32> {
		let modifier = uniforms.scale_modifier;
			let ms = mat3x3<f32>(
				scale.x  * modifier, 0.0, 0.0,
				0.0, scale.y  * modifier, 0.0,
				0.0, 0.0, scale.z  * modifier
			);

			let x = rot.x;
			let y = rot.y;
			let z = rot.z;
			let w = rot.w;

			let mr = mat3x3<f32>(
				1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
				2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
				2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)
			);

			return mr * ms;
		}
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
		let modifier = uniforms.scale_modifier;
		
		let S = mat3x3<f32>(
			(log_scale.x) * modifier, 0., 0.,
			0., (log_scale.y) * modifier, 0.,
			0., 0., (log_scale.z) * modifier,
		);
		
		let x = rot.x;
		let y = rot.y;
		let z = rot.z;
		let w = rot.w;

		let R = mat3x3<f32>(
			1-2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
          	2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x),
          	2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)
		);

		let M =  R * S;

		// ------------ CalcCovariance3D ----------------
		let Sigma = M * transpose(M) ;		
		return array<f32, 6>(
			/*
			*/
			Sigma[0][0],
			Sigma[0][1],
			Sigma[0][2],
			Sigma[1][1],
			Sigma[1][2],			
			Sigma[2][2],			
		);
	} 

	@group(0) @binding(0) var<storage, read> points: array<PointInput>;
	@group(0) @binding(1) var<uniform> 				uniforms: Uniforms;
	@group(0) @binding(2)  var<storage, read> sorted_idx: array<u32>;
	@group(0) @binding(3)  var<storage, read_write> debug_arr: array<u32>;
		
	@fragment
	fn fs_main(input: PointOutput) ->@location(0) vec4f  {
		let selectedColor : vec3<f32> = vec3<f32> (1,0,1);
		var opacity = input.conic_and_opacity.w;   
		var color : vec4<f32> = vec4<f32> (input.color ,opacity);
		let power :f32 = -dot(input.uv, input.uv);
		var alpha :f32 = exp(power);
		if(opacity>=0){
			alpha = saturate(alpha * opacity );
		}
		/*
		*/
		else{
			if(alpha > 7.0/255.0){
				if(alpha < 10.0 /255.0){
					alpha = 1;
					color = vec4<f32> (selectedColor , color.a);
				}
				alpha = saturate(alpha +0.3);
			}
			color = vec4<f32> ( mix(input.color.rgb , selectedColor , 0.5 ) , color.a);
		}
		if(alpha < 1.0/255.0){
			discard;
		}
		
		
		return vec4<f32>(input.color * alpha, alpha);
		//return vec4<f32>(alpha,alpha,alpha, 1);
		//return vec4<f32>(color.rgb, 1);
		//return color;
		}
	fn asfloat(hex: u32) -> f32 {
		let float_value = bitcast<f32>(hex);
		return float_value;
	}
	fn safe_normalize_v2(v: vec2<f32>) -> vec2<f32> {
		let epsilon = 1e-10;
		var x = v.x;
		var y = v.y;
		if(v.x !=0){
			x= x +epsilon;
		}
		if(v.y!=0){
			y= y +epsilon;
		}
		return normalize( vec2f(x,y));
	}
	@vertex
	fn vs_points(
		@builtin(vertex_index) vtxID: u32,
    	@builtin(instance_index) instID: u32,
		@location(0) pos: vec3f ) -> PointOutput {
		
		var output: PointOutput;
		let p_idx = sorted_idx[instID];
		var point = points[p_idx] ;				
		let idx = vtxID;

		var clipPos =  uniforms.projMatrix * uniforms.viewMatrix  * vec4f( point.position , 1);	

		if(clipPos.w<=0){
			let nanfloat = asfloat(0x7fc00000);
			output.position = vec4<f32>(nanfloat , nanfloat , nanfloat,nanfloat); // NaN discards the primitive
		}
		else{
		
			var quadPos = vec2<f32>(
				f32(idx&1), 
				f32((idx>>1)&1)
				) * 2.0 -1 ;
			quadPos *=2;
			output.uv  = quadPos;
			output.position  = clipPos  ;
								

			let splatRotScaleMat : mat3x3<f32> = CalcMatrixFromRotationScale(point.rot, point.log_scale);

			let sig :  mat3x3<f32> = splatRotScaleMat * transpose(splatRotScaleMat);
			var cov3d0 : vec3f = vec3f (sig[0][0] , sig[0][1] , sig[0][2]  );
			var cov3d1 : vec3f = vec3f (sig[1][1] , sig[1][2] , sig[2][2]  );
			
			//output.uv *= cov3d1.yz;   // Why it broke?
			let splatScale = 1.0;
        	let splatScale2 = splatScale * splatScale;
			cov3d0 *= splatScale2;
			cov3d1 *= splatScale2;
			
			let _VecScreenParams = vec4f(${width},${height},0,0);

			var viewPos:vec3f = (uniforms.viewMatrix * vec4<f32>(point.position, 1.0)).xyz;
			let aspect = uniforms.projMatrix[0][0] / uniforms.projMatrix[1][1] ;  			

			let tanFovX: f32 = 1.0 / uniforms.projMatrix[0][0];
			let tanFovY: f32 = 1.0 / (uniforms.projMatrix[1][1] * aspect);

			let limx = 1.3 * tanFovX;
			let limy = 1.3 * tanFovY;
			let txtz = viewPos.x / viewPos.z;
			let tytz = viewPos.y / viewPos.z;

			viewPos.x = min(limx, max(-limx, txtz)) * viewPos.z;
			viewPos.y = min(limy, max(-limy, tytz)) * viewPos.z;

			let focal = _VecScreenParams.x * uniforms.projMatrix[0][0] / 2;
			let J = mat3x3(
				focal / viewPos.z, 0., -(focal * viewPos.x) / (viewPos.z * viewPos.z),
				0., focal / viewPos.z, -(focal * viewPos.y) / (viewPos.z * viewPos.z),
				0., 0., 0., 
			);

			let W = mat3x3<f32>(
				uniforms.viewMatrix[0].xyz,
				uniforms.viewMatrix[1].xyz,
				uniforms.viewMatrix[2].xyz
			);
			
			let T = J * W;

			let Vrk = mat3x3(
				cov3d0.x, cov3d0.y, cov3d0.z,
				cov3d0.y, cov3d1.x, cov3d1.y,
				cov3d0.z, cov3d1.y, cov3d1.z	
			);

			var cov2d_mat = T * ((Vrk) * transpose(T));
			cov2d_mat[0][0] += 0.3;
			cov2d_mat[1][1] += 0.3;

			let cov2d :vec3f  = vec3f(cov2d_mat[0][0] , -cov2d_mat[0][1] , cov2d_mat[1][1]);
			

			let diag1 =  cov2d.x;
			let diag2 =  cov2d.z;
			let offDiag =  cov2d.y;					
			
			var mid =  0.5 *  (diag1 + diag2);
			var radius = length(vec2<f32>((diag1 - diag2) /2.0  , offDiag));
			var lambda1 = mid + radius;
			var lambda2 = max(mid - radius , 0.1);
			var diagVec : vec2<f32> = safe_normalize_v2(vec2<f32>(offDiag , lambda1 - diag1));
			diagVec.y = -diagVec.y;
			
			let maxSize :f32 = 4096.0;
			let v1 : vec2<f32> = min(sqrt(2.0 * lambda1) , maxSize) * diagVec;        
			let v2 : vec2<f32> = min(sqrt(2.0 * lambda2) , maxSize) * vec2<f32>(diagVec.y , -diagVec.x);
		
		let _ScreenParams : vec2<f32> = vec2<f32>(${width},${height});       		
		let deltaScreenPos :vec2<f32> = vec2<f32>(quadPos.x * v1 + quadPos.y * v2) * 2 /_ScreenParams.xy;		

        output.position  .x += deltaScreenPos.x * clipPos.w;
        output.position  .y += deltaScreenPos.y * clipPos.w;
		output.color = compute_color_from_sh(point.position, point.sh);
		
		//================= Other ======================
		let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;	
        let det_inv = 1.0 / det;
		
        let conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
        	output.conic_and_opacity = vec4<f32>(conic, sigmoid(point.opacity_logit));
		}
		return output;

	}
	`
}

const screenPar_w = 600;
function get_calcViewData_Shader(WORKGROUP_SIZE:Number , count : number){
	return`
		struct PointInput {
            @location(0) position: vec3<f32>,
            @location(1) log_scale: vec3<f32>,
            @location(2) rot: vec4<f32>,
            @location(3) opacity_logit: f32,
            sh: array<vec3<f32>, 16>,
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

		fn CalcMatrixFromRotationScale(rot: vec4<f32>, scale: vec3<f32>) -> mat3x3<f32> {
			let ms = mat3x3<f32>(
				scale.x, 0.0, 0.0,
				0.0, scale.y, 0.0,
				0.0, 0.0, scale.z
			);

			let x = rot.x;
			let y = rot.y;
			let z = rot.z;
			let w = rot.w;

			let mr = mat3x3<f32>(
				1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
				2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
				2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)
			);

			return mr * ms;
		}
		fn safe_normalize_v2(v: vec2<f32>) -> vec2<f32> {
			let epsilon = 1e-10;
			var x = v.x;
			var y = v.y;
			if(v.x !=0){
				x= x +epsilon;
			}
			if(v.y!=0){
				y= y +epsilon;
			}
			return normalize( vec2f(x,y));
		}


	 	@group(0) @binding(0) var<storage,read_write> 	splat_pos		: array<vec4f>;
		@group(0) @binding(1) var<storage,read_write> 	splat_axis	: array<vec4f>; 				
		@group(0) @binding(2) var<storage, read> points: array<PointInput>;
		@group(0) @binding(3) var<uniform> 				uniforms: Uniforms;

		@compute
		@workgroup_size(  ${WORKGROUP_SIZE}, 1,1 )
		fn main(
			@builtin(workgroup_id) workgroup_id : vec3<u32>,
			@builtin(local_invocation_id) local_invocation_id : vec3<u32>,
			@builtin(global_invocation_id) global_invocation_id : vec3<u32>,
			@builtin(local_invocation_index) local_invocation_index: u32,
			@builtin(num_workgroups) num_workgroups: vec3<u32>) {
			let workgroup_index =  
				workgroup_id.x +
				workgroup_id.y * num_workgroups.x +
				workgroup_id.z * num_workgroups.x * num_workgroups.y;
			let idx =
				workgroup_index * ${WORKGROUP_SIZE} +
				local_invocation_index;
                
            if(idx >= ${count}){
                return;
            }

			var point = points[idx] ;				
			var clipPos =  uniforms.projMatrix * uniforms.viewMatrix  * vec4f( point.position , 1);
			splat_pos[idx] = clipPos ;


			let splatRotScaleMat : mat3x3<f32> = CalcMatrixFromRotationScale(point.rot, point.log_scale);			
			
			let sig :  mat3x3<f32> = splatRotScaleMat * transpose(splatRotScaleMat);
			var cov3d0 : vec3f = vec3f (sig[0][0] , sig[0][1] , sig[0][2]  );
			var cov3d1 : vec3f = vec3f (sig[1][1] , sig[1][2] , sig[2][2]  );
			
			let _VecScreenParams = vec4f(${screen_size},${screen_size},0,0);

			// Cov2d:			
			var viewPos:vec3f = (uniforms.viewMatrix * vec4<f32>(point.position, 1.0)).xyz;
			let aspect = uniforms.projMatrix[0][0] / uniforms.projMatrix[1][1] ;  // = 1

			let tanFovX: f32 = 1.0 / uniforms.projMatrix[0][0];
			let tanFovY: f32 = 1.0 / (uniforms.projMatrix[1][1] * aspect);

			let limx = 1.3 * tanFovX;
			let limy = 1.3 * tanFovY;
			let txtz = viewPos.x / viewPos.z;
			let tytz = viewPos.y / viewPos.z;

			viewPos.x = min(limx, max(-limx, txtz)) * viewPos.z;
			viewPos.y = min(limy, max(-limy, tytz)) * viewPos.z;

			let focal = _VecScreenParams.x * uniforms.projMatrix[0][0] / 2;
			let J = mat3x3(
				focal / viewPos.z, 0., -(focal * viewPos.x) / (viewPos.z * viewPos.z),
				0., focal / viewPos.z, -(focal * viewPos.y) / (viewPos.z * viewPos.z),
				0., 0., 0., 
			);

			let W = mat3x3<f32>(
				uniforms.viewMatrix[0].xyz,
				uniforms.viewMatrix[1].xyz,
				uniforms.viewMatrix[2].xyz
			);
			
			let T = J * W;

			let Vrk = mat3x3(
				cov3d0.x, cov3d0.y, cov3d0.z,
				cov3d0.y, cov3d1.x, cov3d1.y,
				cov3d0.z, cov3d1.y, cov3d1.z	
			);

			//var cov2d_mat = transpose(T) * transpose(Vrk) * T;
			var cov2d_mat = T * ((Vrk) * transpose(T));
			cov2d_mat[0][0] += 0.3;
			cov2d_mat[1][1] += 0.3;

			let cov2d :vec3f  = vec3f(cov2d_mat[0][0] , -cov2d_mat[0][1] , cov2d_mat[1][1]);

			let diag1 =  cov2d.x;
			let diag2 =  cov2d.z;
			var offDiag =  cov2d.y;						
			
			var mid =  0.5 *  (diag1 + diag2);
			var radius = length(vec2<f32>((diag1 - diag2) /2.0  , offDiag));
			var lambda1 = mid + radius;
			var lambda2 = max(mid - radius , 0.1);

			
			//var diagVec : vec2<f32> = normalize(vec2<f32>(offDiag , lambda1 - diag1));
			var diagVec : vec2<f32> = safe_normalize_v2(vec2<f32>(offDiag , lambda1 - diag1));
			diagVec.y = -diagVec.y;
			
			let maxSize :f32 = 4096.0;
			let v1 : vec2<f32> = min(sqrt(2.0 * lambda1) , maxSize) * diagVec;        
			let v2 : vec2<f32> = min(sqrt(2.0 * lambda2) , maxSize) * vec2<f32>(diagVec.y , -diagVec.x);
			
			//splat_pos[idx] = vec4f(diagVec,offDiag , lambda1 - diag1 );
			//splat_pos[idx] = vec4f(point.rot);
			//splat_pos[idx] = vec4f(v1 , v2);
			//splat_pos[idx] = vec4f(splatRotScaleMat[idx].xyz ,0);
			//splat_pos[idx] = vec4f( cov2d_mat[idx] , f32(idx));
			//splat_pos[idx] = vec4f( focal, tanFovX, tanFovY, tytz);
			//splat_pos[idx] = vec4f( diag1 , diag2 , offDiag,0);
			splat_pos[idx] = vec4f( viewPos,0);

			splat_axis[idx] = vec4f(v1 , v2);
		
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


	pre_processPipeline : GPUComputePipeline;
	pp_splat_pos_Buffer:GPUBuffer;
	pp_splat_axis_Buffer:GPUBuffer;
	
	preprocess_BindGroup:GPUBindGroup;
	framebuffer  : GPUTexture;

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

		let shader_code = get_simple_shader( _canvas.width , _canvas.height);

		this.pipeline = this.context.device.createRenderPipeline({
			vertex: {
				module: this.context.device.createShaderModule({
					code: shader_code,
				}),
				entryPoint: 'vs_points',		
				buffers:[vertexBufferLayout]		
			},
			fragment: {
				module: this.context.device.createShaderModule({
					code: shader_code,
				}),
				entryPoint: 'fs_main',
				targets: [{
					format: presentationFormat,
					blend: {
							//one-minus-dst-alpha
                            color: {
                                srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,                                
                                //srcFactor: "src-alpha" as GPUBlendFactor,
                                dstFactor: "one" as GPUBlendFactor,
                                operation: "add" as GPUBlendOperation,
                            },
                            alpha: {
                                srcFactor: "one-minus-dst-alpha" as GPUBlendFactor,
                                //srcFactor: "src-alpha" as GPUBlendFactor,
								
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

		//========================================================
		//					Pre-process pipeline
		//========================================================
		
		let point_number = 100 ; 
		this.pp_splat_pos_Buffer = this.context.device.createBuffer({
			size: point_number * 4 * 4,  
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST| GPUBufferUsage.COPY_SRC,
		});
		this.context.device.queue.writeBuffer(this.pp_splat_pos_Buffer, 0, new Float32Array(point_number*4));
		this.pp_splat_axis_Buffer = this.context.device.createBuffer({
			size: point_number * 4 * 4,  
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST| GPUBufferUsage.COPY_SRC,
		});
		this.context.device.queue.writeBuffer(this.pp_splat_axis_Buffer, 0, new Float32Array(point_number*4));


		const preprocess_bindinglayout = this.context.device.createBindGroupLayout({
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
                    buffer: {type: 'read-only-storage',},
                },   
				{
                    binding: 3,
                    visibility:GPUShaderStage.COMPUTE,
                    buffer: {type: 'uniform',},
                },          
				  
            ],
        });        
		this.preprocess_BindGroup = this.context.device.createBindGroup({
            layout: preprocess_bindinglayout,
            label: "pre_process BindGroup",
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.pp_splat_pos_Buffer,
                },
            },{
                binding: 1,
                resource: {
                    buffer: this.pp_splat_axis_Buffer,
                },
            },	{
                binding: 2,
                resource: {buffer: _pointBuffer,},
            },
			{
                binding: 3,
                resource: {buffer: _uniformBuffer,},
            },
		
		],
        });
		const preprocess_pipeline_layout= this.context.device.createPipelineLayout({
            bindGroupLayouts: [preprocess_bindinglayout],
        });
        this.pre_processPipeline  = this.context.device.createComputePipeline({
            layout: preprocess_pipeline_layout, 
            compute: {
                module: this.context.device.createShaderModule({
					code: get_calcViewData_Shader(8 , point_number ),
				}),				
                entryPoint: "main",
            }
        });


		//=========================
		//			Texture view
		//=========================
		const textureDescriptor: GPUTextureDescriptor = {
			size: [_canvas.width, _canvas.height, 1],
			format: presentationFormat,
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		};
		const framebufferTexture: GPUTexture = this.context.device.createTexture(textureDescriptor);		
		this.framebuffer = framebufferTexture;
	}


	public draw(gs_number:number) :void{
		const commandEncoder = this.context.device.createCommandEncoder();
		//const textureView = this.contextGpu.getCurrentTexture().createView();
		const textureView = this.framebuffer.createView();

		const renderPassDescriptor : GPURenderPassDescriptor = {
			colorAttachments: [{
                //view: textureView,
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 0},
                storeOp: "store" as GPUStoreOp,
                loadOp: "clear" as GPULoadOp,
            }],
		};


		//============ Preprocess =============
		const pp_encoder = this.context.device.createCommandEncoder();
        const cs_ppr_pass = pp_encoder.beginComputePass();
        cs_ppr_pass.setPipeline(this.pre_processPipeline);
        cs_ppr_pass.setBindGroup(0 , this.preprocess_BindGroup);
        cs_ppr_pass.dispatchWorkgroups(Math.max(gs_number/8 ,8) , 1,1);
        cs_ppr_pass.end();
		this.context.device.queue.submit([pp_encoder.finish()])

		// paste data
		const debug_size = Math.min(gs_number,100);
		const _data_size = 4*4;
		const _buffer_size = debug_size * _data_size;		
		const buffer = this.pp_splat_pos_Buffer;       

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
			const result = new Float32Array(readBuffer.getMappedRange());
			console.log("=============== Read Back =================");
			console.log(result);
			/*
			for(var i = 0 ; i <  debug_size; i++){
				console.log(result[i]);
			}
			*/			
			readBuffer.unmap();
		});


		//============ Draw =============
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