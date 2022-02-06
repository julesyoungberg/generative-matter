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

vec3 get_color(vec2 position) {
    const float particle_size = 2.0;
    const float range = 20.0;
    vec3 color = vec3(0.0);

    for (uint i = 0; i < particle_count; i++) {
        vec2 particle_position = positions[i];
        vec2 diff = position - particle_position;
        
        float d = length(diff);
        float v = smoothstep(particle_size + 0.5, particle_size, d);
        color += v;
        color += pow(range, 2.0) / pow(d, 2.0) * 0.001;
    }

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
