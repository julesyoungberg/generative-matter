use nannou::prelude::*;
use nannou::wgpu::CommandEncoder;
use rand;
use rand::Rng;

use crate::compute::*;
use crate::uniforms::*;
use crate::util::*;

pub struct ParticleSystem {
    pub position_in_buffer: wgpu::Buffer,
    pub position_out_buffer: wgpu::Buffer,
    pub velocity_buffer: wgpu::Buffer,
    pub buffer_size: u64,
    pub initial_positions: Vec<Point2>,
    pub compute: Compute,
    pub particle_count: u32,
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
        let buffer_size = (uniforms.data.particle_count as usize * std::mem::size_of::<Point2>())
            as wgpu::BufferAddress;

        let position_in_buffer = device.create_buffer_with_data(
            &position_bytes[..],
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        );

        let position_out_buffer = device.create_buffer_with_data(
            &position_bytes[..],
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        );

        let velocity_buffer = device.create_buffer_with_data(
            &velocity_bytes[..],
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        );

        // Create the compute shader module.
        let update_cs_mod =
            compile_shader(app, device, "update.comp", shaderc::ShaderKind::Compute);

        let buffers = vec![&position_in_buffer, &position_out_buffer, &velocity_buffer];
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
            position_in_buffer,
            position_out_buffer,
            velocity_buffer,
            buffer_size,
            initial_positions: positions,
            compute,
            particle_count: uniforms.data.particle_count,
        }
    }

    fn copy_positions_out_to_in(&self, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.position_out_buffer,
            0,
            &self.position_in_buffer,
            0,
            self.buffer_size,
        );
    }

    pub fn update(&self, encoder: &mut CommandEncoder) {
        self.compute.compute(encoder, self.particle_count);
        self.copy_positions_out_to_in(encoder);
    }
}

pub fn float_as_bytes(data: &f32) -> &[u8] {
    unsafe { wgpu::bytes::from(data) }
}

pub fn vectors_as_byte_vec(data: &[Point2]) -> Vec<u8> {
    let mut bytes = vec![];
    data.iter().for_each(|v| {
        bytes.extend(float_as_bytes(&v.x));
        bytes.extend(float_as_bytes(&v.y));
    });
    bytes
}
