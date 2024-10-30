import { Vec4 ,Vec3, vec3} from "wgpu-matrix";

export function dot(a : Vec4 , b:Vec4) : number{
	return a[0] * b[0] + 
		a[1]*b[1] +
		a[2]*b[2] +
		a[3]*b[3] ;
}

export function dotF(a: Float32Array , b :Float32Array){
	var r = 0;
	for(var i = 0 ; i < a.length; i++){
		r+= a[i] * b[i];
	}
	return r;
}

export function saturate(value: number): number {
    return Math.min(Math.max(value, 0), 1);
}


export function getlength(v:Vec3):number{
	return vec3.length(v);
	
}

export function compare_raw_vertex( x:number , y :number , z:number, v:Vec3):boolean{	
	return x+y+z > v[0]+v[1]+v[2];
}