use glsl_layout::*;
use nannou::prelude::*;

use crate::compute::*;
use crate::particles::*;
use crate::uniforms::*;
use crate::util::*;

pub struct RadixSort {
    count: Compute,
    scan: Compute,
    reorder: Compute,
    buffer_size: wgpu::BufferAddress,
    bin_count_buffer: wgpu::Buffer,
    prefix_sum_buffer: wgpu::Buffer,
    num_bins: u32,
    particle_count: u32,
}

/// Implements the radix sort algorithm on the GPU with compute shaders
impl RadixSort {
    pub fn new(
        app: &App,
        device: &wgpu::Device,
        particle_system: &ParticleSystem,
        uniforms: &UniformBuffer,
    ) -> Self {
        println!("compiling shaders");

        let count_cs_mod = compile_shader(app, device, "count.comp", shaderc::ShaderKind::Compute);

        let scan_cs_mod = compile_shader(app, device, "scan.comp", shaderc::ShaderKind::Compute);

        let reorder_cs_mod =
            compile_shader(app, device, "reorder.comp", shaderc::ShaderKind::Compute);

        println!("creating buffers");

        let buffer_size =
            (uniforms.data.num_bins as usize * std::mem::size_of::<uint>()) as wgpu::BufferAddress;

        let zeros = vec![0_u8; uniforms.data.num_bins as usize * 4];

        let bin_count_buffer = device.create_buffer_with_data(
            &zeros[..],
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        );

        let prefix_sum_buffer = device.create_buffer_with_data(
            &zeros[..],
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
        );

        println!("creating computes");

        println!("creating count");

        let count_buffers = vec![&particle_system.position_out_buffer, &prefix_sum_buffer];
        let count_buffer_sizes = vec![particle_system.buffer_size, buffer_size];
        let count = Compute::new::<Uniforms>(
            device,
            Some(count_buffers),
            Some(count_buffer_sizes),
            Some(&uniforms.buffer),
            &count_cs_mod,
        )
        .expect("failed to create count compute instance");

        println!("creating scan");

        let scan_buffers = vec![&prefix_sum_buffer];
        let scan_buffer_sizes = vec![buffer_size];
        let scan = Compute::new::<Uniforms>(
            device,
            Some(scan_buffers),
            Some(scan_buffer_sizes),
            Some(&uniforms.buffer),
            &scan_cs_mod,
        )
        .expect("failed to create scan compute instance");

        println!("creating reorder");

        let reorder_buffers = vec![
            &particle_system.position_out_buffer,
            &particle_system.position_in_buffer,
            &prefix_sum_buffer,
            &bin_count_buffer,
        ];
        let reorder_buffer_sizes = vec![
            particle_system.buffer_size,
            particle_system.buffer_size,
            buffer_size,
            buffer_size,
        ];
        let reorder = Compute::new::<Uniforms>(
            device,
            Some(reorder_buffers),
            Some(reorder_buffer_sizes),
            Some(&uniforms.buffer),
            &reorder_cs_mod,
        )
        .expect("failed to create reorder compute instance");

        Self {
            count,
            scan,
            reorder,
            buffer_size,
            bin_count_buffer,
            prefix_sum_buffer,
            num_bins: uniforms.data.num_bins,
            particle_count: uniforms.data.particle_count,
        }
    }

    pub fn update(&self, encoder: &mut wgpu::CommandEncoder) {
        self.count.compute(encoder, self.particle_count);
        self.scan.compute(encoder, self.num_bins);
        self.reorder.compute(encoder, self.particle_count);
    }
}
