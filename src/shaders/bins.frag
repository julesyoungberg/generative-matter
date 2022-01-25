#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) buffer BinBuffer { uint[] bins; };
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

void main() {
    vec2 st = vec2(tex_coords.x, 1.0 - tex_coords.y);
    vec2 position = st * vec2(num_bins_x, num_bins_y);
    ivec2 bin_xy = ivec2(floor(position));
    uint bin_index = uint(bin_xy.y * num_bins_x + bin_xy.x);

    float value = float(bins[bin_index]); // / float(particle_count / 10);

    vec3 color = vec3(value);

    f_color = vec4(color, 1.0);
}
