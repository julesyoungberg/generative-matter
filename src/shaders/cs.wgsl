[[block]]
struct PositionBuffer {
    positions: [[stride(8)]] array<vec2<f32>>;
};

[[block]]
struct VelocityBuffer {
    velocities: [[stride(8)]] array<vec2<f32>>;
};

[[block]]
struct Uniforms {
    size: vec2<f32>;
    speed: f32;
    particle_count: u32;
    attraction_strength: f32;
    repulsion_strength: f32;
};

[[group(0), binding(0)]]
var<storage, read_write> position_buffer_in: PositionBuffer;
[[group(0), binding(1)]]
var<storage, read_write> position_buffer_out: PositionBuffer;
[[group(0), binding(2)]]
var<storage, read_write> velocity_buffer: VelocityBuffer;
[[group(0), binding(3)]]
var<uniform> uniforms: Uniforms;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main([[builtin(global_invocation_id)]] id: vec3<u32>) {
    let index = id.x;
    var position = position_buffer_in.positions[index];
    var velocity = velocity_buffer.velocities[index];
    var acceleration = vec2<f32>(0.0, 0.0);
    
    // var i: i32 = 0;
    // loop {
    //     if (i == uniforms.particle_count) {
    //         break;
    //     }

    //     if (i == index) {
    //         continue;
    //     }
        
    //     let other_position = position_buffer_in.positions[i];
    //     let diff = (other_position - position);
    //     let dist = length(diff);

    //     if (dist < 0.5) {
    //         continue;
    //     }

    //     let dir = normalize(diff);
    //     let r2 = dir * dir;
    //     let attraction_force = dir * uniforms.attraction_strength / r2;
    //     let repulsion_force = dir * uniforms.repulsion_strength / r2;

    //     acceleration = acceleration + attraction_force + repulsion_force;

    //     continuing {
    //         i = i + 1;
    //     }
    // }

    // velocity = velocity + acceleration * uniforms.speed;
    // position = position + acceleration * uniforms.speed;

    // if (position.x < uniforms.size.x * 0.5) {
    //     position.x = position.x + uniforms.size.x;
    // } else {
    //     if (position.x > uniforms.size.x * 0.5) {
    //         position.x = position.y - uniforms.size.x;
    //     }
    // }

    // if (position.y < uniforms.size.y * 0.5) {
    //     position.y = position.y + uniforms.size.y;
    // } else {
    //     if (position.y > uniforms.size.y * 0.5) {
    //         position.y = position.y - uniforms.size.y;
    //     }
    // }

    // velocity_buffer.velocities[i] = velocity;
    // position_buffer_out.positions[i] = position;

    return;
}
