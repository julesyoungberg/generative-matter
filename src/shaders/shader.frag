// NOTE: This shader requires being manually compiled to SPIR-V in order to
// avoid having downstream users require building shaderc and compiling the
// shader themselves. If you update this shader, be sure to also re-compile it
// and update `frag.spv`. You can do so using `glslangValidator` with the
// following command: `glslangValidator -V shader.frag`

#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) buffer PositionBuffer { vec2[] positions; };
layout(set = 0, binding = 1) uniform Uniforms {
    uint particle_count;
    float width;
    float height;
    float speed;
    float attraction_strength;
    float repulsion_strength;
    float attraction_range;
    float repulsion_range;
    float center_strength;
    float particle_radius;
    float collision_response;
    float momentum;
    float max_acceleration;
    float max_velocity;
    uint num_bins_x;
    uint num_bins_y;
    float bin_size_x;
    float bin_size_y;
    uint num_bins;
};

vec3 hash(in vec3 x) {
    const vec3 k = vec3(0.3183099, 0.3678794, 0.3456789);
    x = x*k + k.yzx;
    return -1.0 + 2.0*fract( 16.0 * k*fract( x.x*x.y*(x.x+x.y)) );
}

// returns 3D value noise
float noise(in vec3 x) {
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);

    
    // gradients
    vec3 ga = hash(p+vec3(0.0,0.0,0.0));
    vec3 gb = hash(p+vec3(1.0,0.0,0.0));
    vec3 gc = hash(p+vec3(0.0,1.0,0.0));
    vec3 gd = hash(p+vec3(1.0,1.0,0.0));
    vec3 ge = hash(p+vec3(0.0,0.0,1.0));
    vec3 gf = hash(p+vec3(1.0,0.0,1.0));
    vec3 gg = hash(p+vec3(0.0,1.0,1.0));
    vec3 gh = hash(p+vec3(1.0,1.0,1.0));
    
    // projections
    float va = dot(ga, w-vec3(0.0,0.0,0.0));
    float vb = dot(gb, w-vec3(1.0,0.0,0.0));
    float vc = dot(gc, w-vec3(0.0,1.0,0.0));
    float vd = dot(gd, w-vec3(1.0,1.0,0.0));
    float ve = dot(ge, w-vec3(0.0,0.0,1.0));
    float vf = dot(gf, w-vec3(1.0,0.0,1.0));
    float vg = dot(gg, w-vec3(0.0,1.0,1.0));
    float vh = dot(gh, w-vec3(1.0,1.0,1.0));
	
    // interpolation
    return va + 
           u.x*(vb-va) + 
           u.y*(vc-va) + 
           u.z*(ve-va) + 
           u.x*u.y*(va-vb-vc+vd) + 
           u.y*u.z*(va-vc-ve+vg) + 
           u.z*u.x*(va-vb-ve+vf) + 
           u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
}

float fbm(in vec3 x, in float H, in int numOctaves) {
    float G = exp2(-H);
    float f = 1.0;
    float a = 1.0;
    float t = 0.0;
    for (int i = 0; i < numOctaves; i++) {
        t += a*noise(f*x);
        f *= 2.0;
        a *= G;
    }
    return t;
}

vec3 get_color(vec2 position) {
    vec3 color = vec3(0.0);
    const float range = particle_radius + 0.5;
    float metaball = 0.0;
    float min_dist = max(width, height);

    for (uint i = 0; i < particle_count; i++) {
        vec2 particle_position = positions[i];
        vec2 diff = position - particle_position;
        float d = length(diff);
        min_dist = min(d, min_dist);
        metaball += range * range / dot(diff, diff);

        // if (d < particle_radius) {
        //     density = 1.0;
        //     break;
        // } else if (d < range) {
        //     density += d / range;
        // }
    }

    // add metaball
    color = mix(color, vec3(fbm(vec3(position, min_dist), 1.0, 2)) + 0.5, smoothstep(1.0, 1.1, metaball));

    // add center dot
    color = mix(color, vec3(1.0), smoothstep(particle_radius + 0.1, particle_radius, min_dist));

    return color;
}

void main() {
    // get the corresponding world position
    vec2 position = tex_coords;
    position.y = 1.0 - position.y;
    position -= 0.5;
    position *= vec2(width, height);

    f_color = vec4(get_color(position), 1.0);
}
