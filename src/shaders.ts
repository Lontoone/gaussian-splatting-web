

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