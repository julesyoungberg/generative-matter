use glsl_layout::float;
use glsl_layout::*;
use nannou::prelude::*;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Uniform)]
pub struct Uniforms {
    pub particle_count: uint,
    pub width: float,
    pub height: float,
    pub speed: float,
    pub attraction_strength: float,
    pub repulsion_strength: float,
    pub attraction_range: float,
    pub repulsion_range: float,
    pub center_strength: float,
    pub particle_radius: float,
    pub collision_response: float,
    pub momentum: float,
    pub max_acceleration: float,
    pub max_velocity: float,
}

impl Uniforms {
    pub fn new(particle_count: uint, width: float, height: float) -> Self {
        Uniforms {
            particle_count,
            width,
            height,
            speed: 1.0,
            attraction_strength: 2.8,
            repulsion_strength: 3.0,
            attraction_range: 20.0, // 0.045,
            repulsion_range: 130.0, // 1.2,
            center_strength: 0.0001,
            particle_radius: 2.0,
            collision_response: 0.1,
            momentum: 0.97,
            max_acceleration: 0.0,
            max_velocity: 5.0,
        }
    }
}

pub struct UniformBuffer {
    pub data: Uniforms,
    pub buffer: wgpu::Buffer,
}

impl UniformBuffer {
    pub fn new(device: &wgpu::Device, particle_count: uint, width: float, height: float) -> Self {
        let data = Uniforms::new(particle_count, width, height);

        let std140_uniforms = data.std140();
        let uniforms_bytes = std140_uniforms.as_raw();
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("uniform-buffer"),
            contents: uniforms_bytes,
            usage,
        });

        Self { data, buffer }
    }

    pub fn update(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        // An update for the uniform buffer with the current time.
        let std140_uniforms = self.data.std140();
        let uniforms_bytes = std140_uniforms.as_raw();
        let uniforms_size = uniforms_bytes.len();
        let usage = wgpu::BufferUsages::COPY_SRC;
        let new_uniform_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: None,
            contents: uniforms_bytes,
            usage,
        });

        encoder.copy_buffer_to_buffer(
            &new_uniform_buffer,
            0,
            &self.buffer,
            0,
            uniforms_size as u64,
        );
    }
}
