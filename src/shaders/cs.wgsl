[[block]]
struct Particle {
    position: [[stride(2)]] vec2<f32>;
    velocity: [[stride(2)]] vec2<f32>;
};

[[block]]
struct Buffer {
    data: [[stride(4)]] array<f32>;
};

[[block]]
struct Uniforms {
    size: vec2<f32>;
    speed: f32;
    particle_count: u32;
    attraction_strength: f32;
    repulsion_force: f32;
};

[[group(0), binding(0)]]
var<storage, read_write> buffer: Buffer;
[[group(0), binding(1)]]
var<uniform> uniforms: Uniforms;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main([[builtin(global_invocation_id)]] id: vec3<u32>) {
    // let index = id.x;
    // let p = buffer[index];
    // let acceleration = vec2<f32>(0.0, 0.0);
    
    // for (let i: u32 = 0; i < uniforms.particle_count; i += 1) {
    //     if (i == index) {
    //         continue;
    //     }
        
    //     let other = buffer[i];
    //     let diff = (other.position - p.position);
    //     let dist = diff.length();

    //     if (dist < 0.5) {
    //         continue;
    //     }

    //     let dir = diff.normalize();
    //     let r2 = dir * dir;
    //     let attraction_force = dir * uniforms.attraction_strength / r2;
    //     let repulsion_force = dir * uniforms.repulsion_strength / r2;

    //     acceleration += attraction_force + repulsion_force;
    // }

    // p.velocity += acceleration * uniforms.speed;
    // p.position += acceleration * uniforms.speed;

    // if (p.position.x < uniforms.size.x * 0.5) {
    //     p.position.x += uniforms.size.x;
    // } else if (p.position.x > uniforms.size.x * 0.5) {
    //     p.position.x -= uniforms.size.x;
    // }

    // if (p.position.y < uniforms.size.y * 0.5) {
    //     p.position.y += uniforms.size.y;
    // } else if (p.position.y > uniforms.size.y * 0.5) {
    //     p.position.y -= uniforms.size.y;
    // }

    return;
}
