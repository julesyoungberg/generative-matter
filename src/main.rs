use std::sync::{Arc, Mutex};

use nannou::prelude::*;
use particles::ParticleSystem;

mod capture;
mod compute;
mod particles;
mod radix_sort;
mod render;
mod uniforms;
mod util;

struct Model {
    positions: Arc<Mutex<Vec<Point2>>>,
    threadpool: futures::executor::ThreadPool,
    particle_system: ParticleSystem,
    uniforms: uniforms::UniformBuffer,
    radix_sort: radix_sort::RadixSort,
    frame_capturer: capture::FrameCapturer,
}

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const PARTICLE_COUNT: u32 = 3;

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let window_id = app
        .new_window()
        .size(WIDTH, HEIGHT)
        .view(view)
        .build()
        .unwrap();
    let window = app.window(window_id).unwrap();
    let device = window.swap_chain_device();
    let sample_count = window.msaa_samples();

    println!("creating uniforms");

    // Create the buffer that will store the uniforms.
    let mut uniforms =
        uniforms::UniformBuffer::new(device, PARTICLE_COUNT, WIDTH as f32, HEIGHT as f32);

    let bin_config = radix_sort::BinConfig::new(WIDTH as u64, HEIGHT as u64, -9, -10);
    println!("bin config: {:?}", bin_config);
    bin_config.update_uniforms(&mut uniforms);

    println!("creating particle system");

    let particle_system =
        particles::ParticleSystem::new(app, device, &uniforms, WIDTH as f32 * 0.1);

    println!("creating radix sort");

    let radix_sort =
        radix_sort::RadixSort::new(app, device, &particle_system, &uniforms, sample_count);

    println!("finalizing reasources");

    // Create a thread pool capable of running our GPU buffer read futures.
    let threadpool = futures::executor::ThreadPool::new().unwrap();
    let positions = particle_system.initial_positions.clone();

    let frame_capturer = capture::FrameCapturer::new(app);

    println!("creating model");

    Model {
        positions: Arc::new(Mutex::new(positions)),
        threadpool,
        particle_system,
        uniforms,
        radix_sort,
        frame_capturer,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let window = app.main_window();
    let device = window.swap_chain_device();

    // create a buffer for reading the particle positions
    let read_position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-positions"),
        size: model.particle_system.buffer_size,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
    });

    let read_bin_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-bins"),
        size: model.radix_sort.buffer_size,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
    });

    // The encoder we'll use to encode the compute pass.
    let desc = wgpu::CommandEncoderDescriptor {
        label: Some("particle-compute"),
    };
    let mut encoder = device.create_command_encoder(&desc);

    // model.uniforms.update(device, &mut encoder);

    model.radix_sort.update(device, &mut encoder);

    // model.particle_system.update(&mut encoder);

    encoder.copy_buffer_to_buffer(
        &model.particle_system.position_in_buffer,
        0,
        &read_position_buffer,
        0,
        model.particle_system.buffer_size,
    );

    // encoder.copy_buffer_to_buffer(
    //     &model.particle_system.position_in_buffer,
    //     0,
    //     &model.particle_system.position_out_buffer,
    //     0,
    //     model.particle_system.buffer_size,
    // );

    // encoder.copy_buffer_to_buffer(
    //     &model.particle_system.velocity_in_buffer,
    //     0,
    //     &model.particle_system.velocity_out_buffer,
    //     0,
    //     model.particle_system.buffer_size,
    // );

    encoder.copy_buffer_to_buffer(
        &model.radix_sort.prefix_sum_buffer,
        0,
        &read_bin_buffer,
        0,
        model.radix_sort.buffer_size,
    );

    // model
    //     .frame_capturer
    //     .take_snapshot(device, &mut encoder, &model.uniform_texture);

    // Submit the compute pass to the device's queue.
    window.swap_chain_queue().submit(&[encoder.finish()]);

    // model.frame_capturer.save_frame(app);

    // Spawn a future that reads the result of the compute pass.
    let positions = model.positions.clone();
    let buffer_size = model.particle_system.buffer_size;
    let read_positions_future = async move {
        let result = read_position_buffer.map_read(0, buffer_size).await;
        if let Ok(mapping) = result {
            if let Ok(mut positions) = positions.lock() {
                let bytes = &mapping.as_slice();
                // "Cast" the slice of bytes to a slice of Vec2 as required.
                let slice = {
                    let len = bytes.len() / std::mem::size_of::<Point2>();
                    let ptr = bytes.as_ptr() as *const Point2;
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                };

                positions.copy_from_slice(slice);
                println!("read: {:?}", positions);
            }
        }
    };

    model.threadpool.spawn_ok(read_positions_future);

    let bin_buffer_size = model.radix_sort.buffer_size;
    let read_bins_future = async move {
        let result = read_bin_buffer.map_read(0, bin_buffer_size).await;
        if let Ok(mapping) = result {
            let bytes = &mapping.as_slice();

            // "Casst" the slice of bytes to a slice of Vec2 as required.
            let slice = {
                let len = bytes.len() / std::mem::size_of::<u32>();
                let ptr = bytes.as_ptr() as *const u32;
                unsafe { std::slice::from_raw_parts(ptr, len) }
            };

            println!("bins: {:?}", slice);
        }
    };

    model.threadpool.spawn_ok(read_bins_future);
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(BLACK);
    let draw = app.draw();

    if let Ok(positions) = model.positions.lock() {
        // println!("drawing: {:?}", positions);
        for &p in positions.iter() {
            draw.ellipse()
                .radius(model.uniforms.data.particle_radius)
                .color(WHITE)
                .x_y(p.x, p.y);
        }
    }

    draw.to_frame(app, &frame).unwrap();
}
