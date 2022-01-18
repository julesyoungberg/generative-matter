use nannou::prelude::*;
use rand;
use rand::Rng;

use crate::compute::*;
use crate::uniforms::*;
use crate::util::*;

pub struct ParticleSystem {
    pub position_buffer_in: wgpu::Buffer,
    pub position_buffer_out: wgpu::Buffer,
    pub velocity_buffer: wgpu::Buffer,
    pub buffer_size: u64,
    pub initial_positions: Vec<Point2>,
    pub compute: Compute,
}

impl ParticleSystem {
    pub fn new(
        app: &App,
        device: &wgpu::Device,
        uniforms: &UniformBuffer,
        max_radius: f32,
    ) -> Self {
        let mut positions = vec![];
        let mut velocities = vec![];

        for _ in 0..uniforms.data.particle_count {
            let position_angle =
                rand::thread_rng().gen_range(-std::f32::consts::PI, std::f32::consts::PI);
            let position_radius = rand::thread_rng().gen_range(0.0, max_radius);
            let position_x = position_radius * position_angle.cos();
            let position_y = position_radius * position_angle.sin();
            let position = pt2(position_x, position_y);
            positions.push(position);

            let velocity_x = rand::thread_rng().gen_range(-1.0, 1.0);
            let velocity_y = rand::thread_rng().gen_range(-1.0, 1.0);
            velocities.push(pt2(velocity_x, velocity_y));
        }

        let position_bytes = vectors_as_byte_vec(&positions);
        let velocity_bytes = vectors_as_byte_vec(&velocities);

        // Create the buffers that will store the result of our compute operation.
        let buffer_size = (uniforms.data.particle_count as usize * std::mem::size_of::<Vec2>())
            as wgpu::BufferAddress;

        let position_buffer_in = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("particle-positions-in"),
            contents: &position_bytes[..],
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::COPY_SRC,
        });

        let position_buffer_out = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("particle-positions-out"),
            contents: &position_bytes[..],
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::COPY_SRC,
        });

        let velocity_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("particle-velocities"),
            contents: &velocity_bytes[..],
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::COPY_SRC,
        });

        // Create the compute shader module.
        let update_cs_mod =
            compile_shader(app, device, "update.comp", shaderc::ShaderKind::Compute);

        let buffers = vec![&position_buffer_in, &position_buffer_out, &velocity_buffer];
        let buffer_sizes = vec![buffer_size, buffer_size, buffer_size];

        let compute = Compute::new::<Uniforms>(
            device,
            Some(buffers),
            Some(buffer_sizes),
            Some(&uniforms.buffer),
            &update_cs_mod,
        )
        .unwrap();

        Self {
            position_buffer_in,
            position_buffer_out,
            velocity_buffer,
            buffer_size,
            initial_positions: positions,
            compute,
        }
    }

    pub fn copy_positions_from_out_to_in(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.position_buffer_out,
            0,
            &self.position_buffer_in,
            0,
            self.buffer_size,
        );
    }
}

pub fn float_as_bytes(data: &f32) -> &[u8] {
    unsafe { wgpu::bytes::from(data) }
}

pub fn vectors_as_byte_vec(data: &[Vec2]) -> Vec<u8> {
    let mut bytes = vec![];
    data.iter().for_each(|v| {
        bytes.extend(float_as_bytes(&v.x));
        bytes.extend(float_as_bytes(&v.y));
    });
    bytes
}
