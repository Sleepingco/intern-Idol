# shaders.py - 셰이더 코드 및 관리
from OpenGL.GL import *

# -------------------- 배경 셰이더 --------------------
VS_BG = """
#version 330
layout(location=0) in vec2 aPos; layout(location=1) in vec2 aUV;
out vec2 vUV;
void main(){ vUV=aUV; gl_Position=vec4(aPos,0.0,1.0); }
"""

FS_BG = """
#version 330
in vec2 vUV; out vec4 frag;
// Foreground (camera), Background (image/GIF), and Person mask
uniform sampler2D uTexFG;   // GL_TEXTURE0
uniform sampler2D uTexBG;   // GL_TEXTURE8
uniform sampler2D uMask;    // GL_TEXTURE7 (R8)
uniform int uUseComposite;  // 0: passthrough FG, 1: composite
void main(){
    vec4 fg = texture(uTexFG, vUV);
    if(uUseComposite==0){ frag = fg; return; }
    float m = texture(uMask, vUV).r; // [0..1]
    vec3 bg = texture(uTexBG, vUV).rgb;
    vec3 rgb = mix(bg, fg.rgb, m);
    frag = vec4(rgb, 1.0);
}
"""

# -------------------- 스킨 셰이더 --------------------
VS_SKIN = """
#version 330
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNor;
layout(location=2) in vec2 aUV;
layout(location=3) in ivec4 aJoints;
layout(location=4) in vec4  aWeights;

uniform mat4 uModel;
uniform mat4 uViewProj;
uniform int  uHasSkin;
uniform mat4 uBones[64];

out vec3 vN; out vec3 vW; out vec2 vUV;
void main(){
    vec4 p_model = vec4(aPos, 1.0);
    vec3 n_model = aNor;
    vec4 p_world;
    vec3 n_world;
    if(uHasSkin==1){
        mat4 skin = aWeights.x*uBones[aJoints.x]
                  + aWeights.y*uBones[aJoints.y]
                  + aWeights.z*uBones[aJoints.z]
                  + aWeights.w*uBones[aJoints.w];
        p_world = skin * p_model;
        n_world = mat3(skin) * n_model;
        vW = p_world.xyz; vN = n_world; vUV = aUV;
        gl_Position = uViewProj * p_world;
    } else {
        vec4 wp = uModel * p_model;
        vW = wp.xyz; vN = mat3(uModel) * n_model; vUV = aUV;
        gl_Position = uViewProj * wp;
    }
}
"""

FS_SKIN = """
#version 330
in vec3 vN; in vec3 vW; in vec2 vUV;
out vec4 frag;

uniform vec4  uColor;
uniform sampler2D uBaseTex;
uniform sampler2D uMetallicRoughnessTex;
uniform int   uHasTex;
uniform int   uHasMetallicRoughnessTex;
uniform float uMetallicFactor;
uniform float uRoughnessFactor;

uniform int   uUseGate;  uniform vec3 uGateN0; uniform float uGateD0;
uniform int   uUseJaw;   uniform sampler2D uJawTex; uniform vec2 uScreen; uniform vec2 uViewport; uniform float uSegThr;
uniform float uAlphaCut; uniform int uPremulti;
uniform vec3  uLightDir;

void main(){
    if(uUseGate==1){ if(dot(uGateN0, vW) + uGateD0 > 0.0) discard; }
    if(uUseJaw==1){ vec2 suv = (gl_FragCoord.xy - uViewport) / uScreen; if(texture(uJawTex, suv).r > uSegThr) discard; }

    vec4 base = uColor;
    if(uHasTex==1){ base *= texture(uBaseTex, vUV); }
    if(base.a < uAlphaCut) discard;

    vec3 n = normalize(vN);
    if(length(n) < 1e-6) n = vec3(0,0,1);
    vec3 L = normalize(uLightDir);
    float diff = max(dot(n, L), 0.0);

    vec3 V = normalize(-vW);
    vec3 H = normalize(uLightDir + V);
    float NdotL = max(dot(n, uLightDir), 0.0);
    float NdotV = max(dot(n, V), 0.0);
    float NdotH = max(dot(n, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float metallic = uMetallicFactor;
    float roughness = uRoughnessFactor;
    if(uHasMetallicRoughnessTex == 1) {
        vec4 mrSample = texture(uMetallicRoughnessTex, vUV);
        metallic *= mrSample.b;
        roughness *= mrSample.g;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    vec3 F0 = mix(vec3(0.04), base.rgb, metallic);

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
    float D = alpha2 / (3.14159 * denom * denom);

    float k = alpha / 2.0;
    float G1L = NdotL / (NdotL * (1.0 - k) + k);
    float G1V = NdotV / (NdotV * (1.0 - k) + k);
    float G = G1L * G1V;

    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - metallic);

    vec3 numerator = D * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.001;
    vec3 specular = numerator / denominator;

    vec3 diffuse = kD * base.rgb / 3.14159;
    vec3 radiance = vec3(2.0);
    vec3 lit = (diffuse + specular) * radiance * NdotL + base.rgb * 0.03;

    lit = lit / (lit + vec3(1.0));
    lit = pow(lit, vec3(1.0/2.2));

    if(uPremulti==1) lit *= base.a;
    frag = vec4(lit, base.a);
}
"""

# -------------------- 오클루전 셰이더 --------------------
VS_OCC = """
#version 330
layout(location=0) in vec3 aPos;
uniform mat4 uModel;
uniform mat4 uViewProj;
out vec3 vWorld;
void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vWorld  = wp.xyz;
    gl_Position = uViewProj * wp;
}
"""

FS_OCC = """
#version 330
in vec3 vWorld;
out vec4 frag;
uniform vec3  uColor;
uniform float uAlpha;
uniform int   uUseGate;
uniform vec3  uGateN0;
uniform float uGateD0;
uniform int   uUseSeg;
uniform sampler2D uSegTex;
uniform vec2  uScreen;
uniform vec2  uViewport;
uniform float uSegThr;
void main(){
    if(uUseGate==1){
        float side = dot(uGateN0, vWorld) + uGateD0;
        if(side > 0.0) discard;
    }
    if(uUseSeg==1){
        vec2 uv = (gl_FragCoord.xy - uViewport) / uScreen;
        float m = texture(uSegTex, uv).r;
        if(m < uSegThr) discard;
    }
    frag = vec4(uColor, uAlpha);
}
"""

# -------------------- 메쉬 셰이더 --------------------
VS_MESH = """
#version 330
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNor;
layout(location=2) in vec2 aUV;
uniform mat4 uModel;
uniform mat4 uViewProj;
out vec3 vN; out vec3 vW; out vec2 vUV;
void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vW = wp.xyz; vN = mat3(uModel) * aNor; vUV = aUV;
    gl_Position = uViewProj * wp;
}
"""

FS_MESH = """
#version 330
in vec3 vN;
in vec3 vW;
in vec2 vUV;
out vec4 frag;

uniform int   uUseGate;
uniform vec3  uGateN0;
uniform float uGateD0;

uniform sampler2D texBase;      // unit 0
uniform sampler2D texAlpha;     // unit 1
uniform sampler2D texNormal;    // unit 2
uniform sampler2D texRoughness; // unit 3

uniform int   uUseJaw;
uniform sampler2D uJawTex;
uniform int   uUseSeg;
uniform sampler2D uSegTex;
uniform vec2  uScreen;
uniform vec2  uViewport;
uniform float uSegThr;

uniform vec3  uLightDir;
uniform float uAlphaCut;
uniform int   uPremulti;
uniform int   uHardCutout; // new: hard cutout like texture.py
uniform int   uMode;       // kept for compatibility, not used

vec3 perturbNormal(vec3 pos, vec2 uv, vec3 normal){
    vec3 dp1 = dFdx(pos);
    vec3 dp2 = dFdy(pos);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 N = normalize(normal);
    vec3 T = normalize(dp1 * duv2.y - dp2 * duv1.y);
    vec3 B = normalize(-dp1 * duv2.x + dp2 * duv1.x);
    vec3 mapN = texture(texNormal, uv).xyz * 2.0 - 1.0;
    mapN.y = -mapN.y; // OpenGL convention fix if needed
    return normalize(mat3(T, B, N) * mapN);
}

void main()
{
    // keep existing occlusion conditions
    if(uUseGate==1){
        if(dot(uGateN0, vW) + uGateD0 > 0.0) discard;
    }
    if(uUseJaw==1){
        vec2 suv = (gl_FragCoord.xy - uViewport) / uScreen;
        if(texture(uJawTex, suv).r > uSegThr) discard;
    }
    if(uUseSeg==1){
        vec2 suv = (gl_FragCoord.xy - uViewport) / uScreen;
        if(texture(uSegTex, suv).r < uSegThr) discard;
    }

    // flip V for typical OBJ
    vec2 uv = vec2(vUV.x, 1.0 - vUV.y);

    // 4 textures together (texture.py style)
    vec4 baseTex   = texture(texBase, uv);
    float alphaMap = texture(texAlpha, uv).r;
    float alpha    = baseTex.a * alphaMap;

    if(alpha < uAlphaCut) discard;
    if(uHardCutout == 1) alpha = 1.0; // hard cutout to avoid edge halo

    float rough = clamp(texture(texRoughness, uv).r, 0.04, 1.0);
    vec3 N = perturbNormal(vW, uv, vN);

    vec3 L = normalize(uLightDir);
    vec3 V = normalize(-vW);
    vec3 H = normalize(L + V);

    float diff      = max(dot(normalize(N), L), 0.0);
    float shininess = mix(128.0, 4.0, rough);                 // rougher -> wider
    float spec      = pow(max(dot(normalize(N), H), 0.0), shininess) * (1.0 - rough);

    vec3 albedo = baseTex.rgb;
    if(uPremulti == 1) albedo *= alpha;                       // premultiplied alpha

    vec3 color = albedo * (0.2 + 0.8 * diff) + vec3(spec);    // simple lit output
    frag = vec4(color, alpha);
}
"""


def compile_shader(src, stype):
    """셰이더 컴파일"""
    sid = glCreateShader(stype)
    glShaderSource(sid, src)
    glCompileShader(sid)
    if glGetShaderiv(sid, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(sid).decode())
    return sid


def link_program(vs, fs):
    """셰이더 프로그램 링크"""
    pid = glCreateProgram()
    glAttachShader(pid, vs)
    glAttachShader(pid, fs)
    glLinkProgram(pid)
    if glGetProgramiv(pid, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(pid).decode())
    glDetachShader(pid, vs)
    glDetachShader(pid, fs)
    glDeleteShader(vs)
    glDeleteShader(fs)
    return pid


# -------------------- 스티커 셰이더 --------------------
VS_STICKER = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
uniform mat4 uOrtho;
out vec2 vUV;
void main() {
    gl_Position = uOrtho * vec4(aPos.x, aPos.y, 0.0, 1.0);
    vUV = aUV;
}
"""

FS_STICKER = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTexture;
uniform float uAlpha;
void main() {
    vec4 texColor = texture(uTexture, vUV);
    fragColor = vec4(texColor.rgb, texColor.a * uAlpha);
}
"""

def create_all_programs():
    """모든 셰이더 프로그램 생성"""
    prog_bg = link_program(
        compile_shader(VS_BG, GL_VERTEX_SHADER),
        compile_shader(FS_BG, GL_FRAGMENT_SHADER),
    )
    prog_skin = link_program(
        compile_shader(VS_SKIN, GL_VERTEX_SHADER),
        compile_shader(FS_SKIN, GL_FRAGMENT_SHADER),
    )
    prog_mesh = link_program(
        compile_shader(VS_MESH, GL_VERTEX_SHADER),
        compile_shader(FS_MESH, GL_FRAGMENT_SHADER),
    )
    prog_occ = link_program(
        compile_shader(VS_OCC, GL_VERTEX_SHADER),
        compile_shader(FS_OCC, GL_FRAGMENT_SHADER),
    )
    # 스티커 프로그램 추가
    prog_sticker = link_program(
        compile_shader(VS_STICKER, GL_VERTEX_SHADER),
        compile_shader(FS_STICKER, GL_FRAGMENT_SHADER),
    )
    return prog_bg, prog_skin, prog_mesh, prog_occ, prog_sticker
