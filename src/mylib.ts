import { Vec4 } from "wgpu-matrix";

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
