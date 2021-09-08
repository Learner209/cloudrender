#version 330 core
#define DIST_DEGREE 5

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec4 vertexColor;
layout(location = 2) in vec3 vertexNorm;

out VS_OUT {
	vec3 pose;
    vec4 color;
	float depth;
	vec3 normal;
	vec3 MVnormal;
} vs_out;

uniform mat4 M;
uniform mat4 V;
uniform float distorsion_coeff[DIST_DEGREE];
uniform vec2 center_off;
uniform vec2 focal_dist;
uniform float far;
void main(){
	mat4 MV = V*M;
	vec4 vertexPosMV = MV * vec4(vertexPos, 1);
	vs_out.pose = vec3(M * vec4(vertexPos, 1.0));
	vec2 xy1 = vertexPosMV.xy/vertexPosMV.z;
	float radius_sq = dot(xy1,xy1);
	float radius_quad = radius_sq*radius_sq;
	float radial_distorsion = 1+distorsion_coeff[0]*radius_sq+distorsion_coeff[1]*radius_quad+
		distorsion_coeff[4]*radius_quad*radius_sq;
	vec2 tan_distorsion = vec2(2*distorsion_coeff[2]*xy1.x*xy1.y+distorsion_coeff[3]*(radius_sq+2*xy1.x*xy1.x),
								distorsion_coeff[2]*(radius_sq+2*xy1.y*xy1.y)+2*distorsion_coeff[3]*xy1.x*xy1.y);
	vec2 xy2 = xy1*radial_distorsion+tan_distorsion;
	vec2 res = focal_dist*xy2+center_off;
	if (res.x>1||res.x<-1||res.y>1||res.y<-1)
	{
		gl_Position = vec4(-res,
	                   -2.,
					   1.0);
	}
	else
	{
		gl_Position = vec4(res,
	                   length(vertexPosMV)*(sign(vertexPosMV.z))/far*2-1,
					   1.0);
	}

//	float pos_d = sign(vertexPosMV.z)*(far-length(vertexPosMV));
//	pos_d = (pos_d>0.05)?pos_d:-1;



	vs_out.color = vertexColor;
	vs_out.depth = abs(vertexPosMV.z);
	vs_out.normal = mat3(M) * vertexNorm;
}