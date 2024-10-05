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

const shDeg2Code = `
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

    fn compute_color_from_sh(position: vec3<f32>, sh: array<vec3<f32>, 9>) -> vec3<f32> {
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
        
        // unconditional
        result = result + 0.5;

        return max(result, vec3<f32>(0.));
    }
`;

const shDeg1Code = `
    // spherical harmonic coefficients
    const SH_C0 = 0.28209479177387814f;
    const SH_C1 = 0.4886025119029199f;

    fn compute_color_from_sh(position: vec3<f32>, sh: array<vec3<f32>, 4>) -> vec3<f32> {
        let dir = normalize(position - uniforms.camera_position);
        var result = SH_C0 * sh[0];

        // if deg > 0
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result = result + SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

        // unconditional
        result = result + 0.5;

        return max(result, vec3<f32>(0.));
    }
`;


export function getShaderCode(canvas: HTMLCanvasElement, shDegree: number, nShCoeffs: number , screenPar_w :number,screenPar_h :number ) {
    const shComputeCode = {
        1: shDeg1Code,
        2: shDeg2Code,
        3: shDeg3Code,
    }[shDegree];

    const shaderCode = `
// for some reason passing these as uniform is broken
const canvas_height = ${canvas.height};
const canvas_width = ${canvas.width};
const sh_degree = ${shDegree};
const n_sh_coeffs = ${nShCoeffs};

struct PointInput {
    @location(0) position: vec3<f32>,
    @location(1) log_scale: vec3<f32>,
    @location(2) rot: vec4<f32>,
    @location(3) opacity_logit: f32,
    sh: array<vec3<f32>, n_sh_coeffs>,
};

struct PointOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) conic_and_opacity: vec4<f32>,
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

${shComputeCode}

fn sigmoid(x: f32) -> f32 {
    if (x >= 0.) {
        return 1. / (1. + exp(-x));
    } else {
        let z = exp(x);
        return z / (1. + z);
    }
}

fn compute_cov3d(log_scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {
		let modifier = uniforms.scale_modifier;
		let S = mat3x3<f32>(
			log_scale.x * modifier, 0., 0.,
			0., log_scale.y * modifier, 0.,
			0., 0., log_scale.z * modifier,
		);

		let r = rot.x;
		let x = rot.y;
		let y = rot.z;
		let z = rot.w;

		let R = mat3x3<f32>(
			1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
			2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
			2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y),
		);

		let M = S * R;
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

	fn ndc2pix(v: f32, size: u32) -> f32 {
		return ((v + 1.0) * f32(size) - 1.0) * 0.5;
	}

	fn compute_cov2d(position: vec3<f32>, log_scale: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
		let cov3d = compute_cov3d(log_scale, rot);
		let aspect = 1.0;  //TODO!
		var t = uniforms.viewMatrix * vec4<f32>(position, 1.0);
		let tanFovX: f32 = 1.0 / uniforms.projMatrix[0][0];
		let tanFovY: f32 = 1.0 / (uniforms.projMatrix[1][1] * aspect);
		//    	float tanFovX = rcp(matrixP._m00);
    	//		float tanFovY = rcp(matrixP._m11 * aspect);
		//let limx = 1.3 * uniforms.tan_fovx;
		//let limy = 1.3 * uniforms.tan_fovy;

		let screenParams_x = f32(${screenPar_w}); //TODO;
		let focal = screenParams_x * uniforms.projMatrix[0][0] / 2;

		let limx = 1.3 * tanFovX;
		let limy = 1.3 * tanFovY;

		let txtz = t.x / t.z;
		let tytz = t.y / t.z;

		t.x = min(limx, max(-limx, txtz)) * t.z;
		t.y = min(limy, max(-limy, tytz)) * t.z;
		/*
		let J = mat4x4(
			uniforms.focal_x / t.z, 0., -(uniforms.focal_x * t.x) / (t.z * t.z), 0.,
			0., uniforms.focal_y / t.z, -(uniforms.focal_y * t.y) / (t.z * t.z), 0.,
			0., 0., 0., 0.,
			0., 0., 0., 0.,
		);
		*/
		let J = mat4x4(
			focal / t.z, 0., -(focal * t.x) / (t.z * t.z), 0.,
			0., focal / t.z, -(focal * t.y) / (t.z * t.z), 0.,
			0., 0., 0., 0.,
			0., 0., 0., 0.,
		);

		let W = transpose(uniforms.viewMatrix);

		//let T = W *  J;   // Origin , but transpose is better?
		let T = W *  transpose(J);

		let Vrk = mat4x4(
			cov3d[0], cov3d[1], cov3d[2], 0.,
			cov3d[1], cov3d[3], cov3d[4], 0.,
			cov3d[2], cov3d[4], cov3d[5], 0.,
			0., 0., 0., 0.,
		);

		var cov = transpose(T) * transpose(Vrk) * T;

		// Apply low-pass filter: every Gaussian should be at least
		// one pixel wide/high. Discard 3rd row and column.
		cov[0][0] += 0.3;
		cov[1][1] += 0.3;

		return vec3<f32>(cov[0][0], cov[0][1], cov[1][1]);
	}


@binding(0) @group(0) var<uniform> uniforms: Uniforms;
@binding(1) @group(1) var<storage, read> points: array<PointInput>;
@binding(2) @group(1) var<storage, read> sorted_idx: array<u32>;

const quadVertices = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
);

fn asfloat(hex: u32) -> f32 {
    let float_value = bitcast<f32>(hex);
    return float_value;
}
@vertex
fn vs_points(
    //@builtin(vertex_index) vertex_index: u32
    @builtin(vertex_index) vtxID: u32,
    @builtin(instance_index) instID: u32
    ) -> PointOutput {

    var output: PointOutput;
    let p_idx = sorted_idx[instID];
    let point = points[p_idx];
    let idx = vtxID;

    var clipPos = uniforms.projMatrix  * uniforms.viewMatrix *  vec4<f32>(point.position, 1.0);    
    //var clipPos = uniforms.projMatrix  *  vec4<f32>(point.position, 1.0);    

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
        
        let cov2d = compute_cov2d(point.position, point.log_scale, point.rot);
        let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        let det_inv = 1.0 / det;
        let conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
        output.conic_and_opacity = vec4<f32>(conic, sigmoid(point.opacity_logit));
        
        var mid =  0.5 * (cov2d.x + cov2d.z);
        var radius = length(vec2<f32>((cov2d.x - cov2d.z) /2.0  , cov2d.y));
        var lambda1 = mid + radius;
        var lambda2 = max(mid - radius , 0.1);
        var diagVec : vec2<f32> = normalize(vec2<f32>(cov2d.y , lambda1 - lambda2));
        diagVec.y = -diagVec.y;
        
        let maxSize :f32 = 4096.0;
        let v1 : vec2<f32> = min(sqrt(2.0 * lambda1) , maxSize) * diagVec;
        let v2 : vec2<f32> = min(sqrt(2.0 * lambda2) , maxSize) * vec2<f32>(diagVec.y , -diagVec.x);
        
        let _ScreenParams : vec2<f32> = vec2<f32>(${screenPar_w} , ${screenPar_h});       
        
        let deltaScreenPos :vec2<f32> = (quadPos.x * v1 + quadPos.y * v2) * 2 / _ScreenParams.xy;
        
        output.position  .x += deltaScreenPos.x * clipPos.w;
        output.position  .y += deltaScreenPos.y * clipPos.w;
        
        output.color = compute_color_from_sh(point.position, point.sh);
    }


	
    /*
    let cov2d = compute_cov2d(point.position, point.log_scale, point.rot);
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    let det_inv = 1.0 / det;
    let conic = vec3<f32>(cov2d.z * det_inv, -cov2d.y * det_inv, cov2d.x * det_inv);
    output.conic_and_opacity = vec4<f32>(conic, sigmoid(point.opacity_logit));

    var projPosition = uniforms.projMatrix * vec4<f32>(point.position, 1.0);   
    if(projPosition.w <=0){
        //behind camera
        output.uv = vec2f(-99,-99);
        return output;
    }
    var mid =  0.5 * (cov2d.x + cov2d.z);
	var radius = length(vec2<f32>((cov2d.x - cov2d.z) /2.0  , cov2d.y));
	var lambda1 = mid + radius;
	var lambda2 = max(mid - radius , 0.1);
	var diagVec : vec2<f32> = normalize(vec2<f32>(cov2d.y , lambda1 - lambda2));
	diagVec.y = -diagVec.y;
		
	let maxSize :f32 = 4096.0;
	let v1 : vec2<f32> = min(sqrt(2.0 * lambda1) , maxSize) * diagVec;
	let v2 : vec2<f32> = min(sqrt(2.0 * lambda2) , maxSize) * vec2<f32>(diagVec.y , -diagVec.x);

    let _ScreenParams : vec2<f32> = vec2<f32>(${screenPar_w} , ${screenPar_h});
    
    output.uv  = quadPos;
    output.position  = projPosition;
    let deltaScreenPos :vec2<f32> = (quadPos.x * v1 + quadPos.y * v2) * 2 / _ScreenParams.xy;

	output.position  .x += deltaScreenPos.x * projPosition.w;
	output.position  .y += deltaScreenPos.y * projPosition.w;

    output.color = compute_color_from_sh(point.position, point.sh);
    */

    return output;
}

@fragment
fn fs_main(input: PointOutput) -> @location(0) vec4<f32> {
   
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
    
    
    //return vec4<f32>(input.color * alpha, alpha);
    return vec4<f32>(alpha,alpha,alpha, 1);
    //return vec4<f32>(color.rgb, 1);
    //return color;
}
`;

    return shaderCode;
}




//======================
//      Init Sort buffer
//======================

export function getInitSortBufferCode(count:number , nShCoeffs : number) {
    const WORKGROUP_SIZE = 8;
    return `
        
        const n_sh_coeffs = ${nShCoeffs};
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

        @group(0) @binding(0) var<storage,read_write> 	gaussian_keys_unsorted		: array<u32>;
		@group(0) @binding(1) var<storage,read_write> 	gaussian_values_unsorted	: array<u32>; 
		@group(0) @binding(2) var<uniform> 				uniforms: Uniforms;
		@group(0) @binding(3) var<storage,read_write> 	splatPos: array<PointInput>; 		
		
		fn float_to_sortable_uint(f: f32) -> u32 {
			let fu: u32 = bitcast<u32>(f);
			let mask: u32 = bitcast<u32>(-(bitcast<i32>(fu) >> 31)) | 0x80000000u;
			return fu ^ mask;
		}

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
            /*
            */
			let idx =
				workgroup_index * ${WORKGROUP_SIZE} +
				local_invocation_index;
                
            if(idx >= ${count}){
                    return;
            }
            

			gaussian_keys_unsorted[idx] = idx;
			let pos : vec3<f32> = (uniforms.viewMatrix * vec4<f32> (splatPos[idx].position.xyz , 1.0)).xyz;
			gaussian_values_unsorted[idx] = float_to_sortable_uint(pos.z);
			//gaussian_values_unsorted[idx] =pos.z;
			//gaussian_values_unsorted[idx] =idx;
			
			
		}
    `
}