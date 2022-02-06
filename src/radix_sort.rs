use glsl_layout::*;
use nannou::prelude::*;

use crate::compute::*;
use crate::particles::*;
use crate::render::*;
use crate::uniforms::*;
use crate::util::*;

pub struct RadixSort {
    count: Compute,
    scan: Compute,
    reorder: Compute,
    pub buffer_size: wgpu::BufferAddress,
    pub bin_count_buffer: wgpu::Buffer,
    pub prefix_sum_buffer: wgpu::Buffer,
    num_bins: u32,
    particle_count: u32,
    pub debug: CustomRenderer,
}

/// Implements the radix sort algorithm on the GPU with compute shaders
impl RadixSort {
    pub fn new(
        app: &App,
        device: &wgpu::Device,
        particle_system: &ParticleSystem,
        uniforms: &UniformBuffer,
        sample_count: u32,
    ) -> Self {
        println!("compiling shaders");

        let count_cs_mod = compile_shader(app, device, "count.comp", shaderc::ShaderKind::Compute);

        let scan_cs_mod = compile_shader(app, device, "scan.comp", shaderc::ShaderKind::Compute);

        let reorder_cs_mod =
            compile_shader(app, device, "reorder.comp", shaderc::ShaderKind::Compute);

        let vs_mod = compile_shader(app, device, "shader.vert", shaderc::ShaderKind::Vertex);

        let bins_fs_mod = compile_shader(app, device, "bins.frag", shaderc::ShaderKind::Fragment);

        println!("creating buffers");

        let buffer_size =
            (uniforms.data.num_bins as usize * std::mem::size_of::<uint>()) as wgpu::BufferAddress;

        let zeros = vec![0_u8; buffer_size as usize];

        println!("creating bin count buffer");

        let bin_count_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("bin-count-buffer"),
            contents: &zeros[..],
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        println!("creating prefix sum buffer");

        let prefix_sum_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("prefix-sum-buffer"),
            contents: &zeros[..],
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

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
            &particle_system.velocity_out_buffer,
            &particle_system.velocity_in_buffer,
            &prefix_sum_buffer,
            &bin_count_buffer,
        ];
        let reorder_buffer_sizes = vec![
            particle_system.buffer_size,
            particle_system.buffer_size,
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

        let debug = CustomRenderer::new::<Uniforms>(
            device,
            &vs_mod,
            &bins_fs_mod,
            Some(&vec![&prefix_sum_buffer]),
            Some(&vec![&buffer_size]),
            None,
            None,
            Some(&uniforms.buffer),
            uniforms.data.width as u32,
            uniforms.data.height as u32,
            1,
            sample_count,
        )
        .unwrap();

        Self {
            count,
            scan,
            reorder,
            buffer_size,
            bin_count_buffer,
            prefix_sum_buffer,
            num_bins: uniforms.data.num_bins,
            particle_count: uniforms.data.particle_count,
            debug,
        }
    }

    fn clear_buffer(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
    ) {
        let zeros = vec![0_u8; self.buffer_size as usize];
        let zeros_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: None,
            contents: &zeros[..],
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        encoder.copy_buffer_to_buffer(&zeros_buffer, 0, buffer, 0, self.buffer_size);
    }

    fn clear_buffers(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.clear_buffer(device, encoder, &self.bin_count_buffer);
        self.clear_buffer(device, encoder, &self.prefix_sum_buffer);
    }

    pub fn update(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        // self.clear_buffers(device, encoder);
        self.count.compute(encoder, self.particle_count);
        self.scan.compute(encoder, self.num_bins);
        self.reorder.compute(encoder, self.particle_count);
        // self.debug.render(encoder);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BinConfig {
    pub grid_x: u64,
    pub grid_y: u64,
    pub bin_x: f64,
    pub bin_y: f64,
    pub num_bins: u64,
}

impl BinConfig {
    pub fn new(raw_width: u64, raw_height: u64, x_scale: i32, y_scale: i32) -> Self {
        let mut grid_x = raw_width.next_power_of_two();
        let mut grid_y = raw_height.next_power_of_two();
        println!("initial grid_x: {:?}, grid_y: {:?}", grid_x, grid_y);

        if x_scale > 0 {
            let mut i = 0;
            while i < x_scale {
                grid_x *= 2;
                i += 1;
            }
        }

        if x_scale < 0 {
            let mut i = 0;
            while i > x_scale {
                grid_x /= 2;
                i -= 1;
            }
        }

        if y_scale > 0 {
            let mut i = 0;
            while i < y_scale {
                grid_y *= 2;
                i += 1;
            }
        }

        if y_scale < 0 {
            let mut i = 0;
            while i > y_scale {
                grid_y /= 2;
                i -= 1;
            }
        }

        let bin_x = raw_width as f64 / grid_x as f64;
        let bin_y = raw_height as f64 / grid_y as f64;

        let num_bins = grid_x * grid_y;

        Self {
            grid_x,
            grid_y,
            bin_x,
            bin_y,
            num_bins,
        }
    }

    pub fn update_uniforms(&self, uniforms: &mut UniformBuffer) {
        uniforms.data.num_bins_x = self.grid_x as u32;
        uniforms.data.num_bins_y = self.grid_y as u32;
        uniforms.data.bin_size_x = self.bin_x as f32;
        uniforms.data.bin_size_y = self.bin_y as f32;
        uniforms.data.num_bins = self.num_bins as u32;
    }
}
