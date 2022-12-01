#version 330 core
#define SHADOWMAPS_MAX 6

layout(location = 0) in vec3 vertexPos;
layout(location = 1) in vec4 vertexColor;
layout(location = 2) in int vertexId;
layout(location = 3) in vec3 vertexNorm;

out VS_OUT {
    vec4 color;
	int inst_id;
	vec3 norm;
	vec4 poseMV;
	vec4 pose_shadow[SHADOWMAPS_MAX];
} vs_out;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform bool shadowmap_enabled[SHADOWMAPS_MAX];
uniform mat4 shadowmap_MVP[SHADOWMAPS_MAX];
void main(){
	mat4 MV = V*M;
	vec4 vertexPosMV = MV * vec4(vertexPos, 1);
	gl_Position = P * vertexPosMV;

	vs_out.color = vertexColor;
	vs_out.inst_id = vertexId;
	vs_out.poseMV = vertexPosMV;
//	vs_out.normMV = (MV * vec4(vertexNorm, 0)).xyz;
	vs_out.norm = vertexNorm;
	for (int i=0; i<SHADOWMAPS_MAX && shadowmap_enabled[i]; ++i)
		vs_out.pose_shadow[i] = shadowmap_MVP[i] * vec4(vertexPos, 1);
}