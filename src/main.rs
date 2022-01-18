use std::sync::{Arc, Mutex};

use nannou::prelude::*;
use particles::ParticleSystem;

// mod capture;
mod compute;
mod particles;
mod radix_sort;
mod uniforms;
mod util;

struct Model {
    positions: Arc<Mutex<Vec<Vec2>>>,
    threadpool: futures::executor::ThreadPool,
    particle_system: ParticleSystem,
    uniforms: uniforms::UniformBuffer,
    radix_sort: radix_sort::RadixSort,
}

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const PARTICLE_COUNT: u32 = 8000;

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

    // Create the buffer that will store the uniforms.
    let uniforms =
        uniforms::UniformBuffer::new(device, PARTICLE_COUNT, WIDTH as f32, HEIGHT as f32);

    let particle_system =
        particles::ParticleSystem::new(app, device, &uniforms, WIDTH as f32 * 0.1);

    let radix_sort = radix_sort::RadixSort::new(app, device, &particle_system, &uniforms);

    // Create a thread pool capable of running our GPU buffer read futures.
    let threadpool = futures::executor::ThreadPool::new().unwrap();
    let positions = particle_system.initial_positions.clone();

    Model {
        positions: Arc::new(Mutex::new(positions)),
        threadpool,
        particle_system,
        uniforms,
        radix_sort,
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
        mapped_at_creation: false,
    });

    // The encoder we'll use to encode the compute pass.
    let desc = wgpu::CommandEncoderDescriptor {
        label: Some("particle-compute"),
    };
    let mut encoder = device.create_command_encoder(&desc);

    // model.uniforms.update(device, &mut encoder);

    model.radix_sort.update(&mut encoder);

    model.particle_system.update(&mut encoder);

    encoder.copy_buffer_to_buffer(
        &model.particle_system.position_buffer_out,
        0,
        &read_position_buffer,
        0,
        model.particle_system.buffer_size,
    );

    // Submit the compute pass to the device's queue.
    window.swap_chain_queue().submit(Some(encoder.finish()));

    // Spawn a future that reads the result of the compute pass.
    let positions = model.positions.clone();
    let read_positions_future = async move {
        let slice = read_position_buffer.slice(..);
        if let Ok(_) = slice.map_async(wgpu::MapMode::Read).await {
            if let Ok(mut positions) = positions.lock() {
                let bytes = &slice.get_mapped_range()[..];
                // "Cast" the slice of bytes to a slice of Vec2 as required.
                let slice = {
                    let len = bytes.len() / std::mem::size_of::<Vec2>();
                    let ptr = bytes.as_ptr() as *const Vec2;
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                };

                positions.copy_from_slice(slice);
            }
        }
    };

    model.threadpool.spawn_ok(read_positions_future);
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(BLACK);
    let draw = app.draw();

    if let Ok(positions) = model.positions.lock() {
        for &p in positions.iter() {
            draw.ellipse()
                .radius(model.uniforms.data.particle_radius)
                .color(WHITE)
                .x_y(p.x, p.y);
        }
    }

    draw.to_frame(app, &frame).unwrap();
}
