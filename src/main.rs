use nannou::prelude::*;
use nannou::wgpu::BufferInitDescriptor;
use rand;
use rand::Rng;
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Particle {
    position: Vec2,
    velocity: Vec2,
}

struct Model {
    compute: Compute,
    particles: Arc<Mutex<Vec<Particle>>>,
    threadpool: futures::executor::ThreadPool,
}

struct Compute {
    particle_buffer: wgpu::Buffer,
    particle_buffer_size: wgpu::BufferAddress,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Uniforms {
    size: Vec2,
    speed: f32,
    particle_count: u32,
    attraction_strength: f32,
    repulsion_strength: f32,
}

const PARTICLE_COUNT: u32 = 1000;
const WIDTH: u32 = 1440;
const HEIGHT: u32 = 512;

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let w_id = app
        .new_window()
        .size(WIDTH, HEIGHT)
        .view(view)
        .build()
        .unwrap();
    let window = app.window(w_id).unwrap();
    let device = window.device();

    // Create the compute shader module.
    let cs_desc = wgpu::include_wgsl!("shaders/cs.wgsl");
    let cs_mod = device.create_shader_module(&cs_desc);

    // Create the buffer that will store the result of our compute operation.
    let particle_buffer_size =
        (PARTICLE_COUNT as usize * std::mem::size_of::<Particle>()) as wgpu::BufferAddress;
    let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles"),
        size: particle_buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create the buffer that will store the uniforms.
    let uniforms = create_uniforms();
    let uniforms_bytes = uniforms_as_bytes(&uniforms);
    let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-buffer"),
        contents: uniforms_bytes,
        usage,
    });

    // Create the bind group and pipeline.
    let bind_group_layout = create_bind_group_layout(device);
    let bind_group = create_bind_group(
        device,
        &bind_group_layout,
        &particle_buffer,
        particle_buffer_size,
        &uniform_buffer,
    );
    let pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
    let pipeline = create_compute_pipeline(device, &pipeline_layout, &cs_mod);

    let compute = Compute {
        particle_buffer,
        particle_buffer_size,
        uniform_buffer,
        bind_group,
        pipeline,
    };

    // The vector that we will write particle values to.
    let particles = Arc::new(Mutex::new(vec![]));

    if let Ok(mut prtcls) = particles.lock() {
        let hwidth = WIDTH as f32 * 0.5;
        let hheight = HEIGHT as f32 * 0.5;
        let position_x = rand::thread_rng().gen_range(-hwidth, hwidth);
        let position_y = rand::thread_rng().gen_range(-hheight, hheight);
        let velocity_x = rand::thread_rng().gen_range(-1.0, 1.0);
        let velocity_y = rand::thread_rng().gen_range(-1.0, 1.0);

        let p = Particle {
            position: pt2(position_x, position_y),
            velocity: pt2(velocity_x, velocity_y),
        };

        prtcls.push(p);
    }

    // Create a thread pool capable of running our GPU buffer read futures.
    let threadpool = futures::executor::ThreadPool::new().unwrap();

    Model {
        compute,
        particles,
        threadpool,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let window = app.main_window();
    let device = window.device();
    let compute = &mut model.compute;

    // The buffer into which we'll read some data.
    let read_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-particles"),
        size: compute.particle_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // An update for the uniform buffer with the current time.
    let uniforms = create_uniforms();
    let uniforms_size = std::mem::size_of::<Uniforms>() as wgpu::BufferAddress;
    let uniforms_bytes = uniforms_as_bytes(&uniforms);
    let usage = wgpu::BufferUsages::COPY_SRC;
    let new_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-data-transfer"),
        contents: uniforms_bytes,
        usage,
    });

    // The encoder we'll use to encode the compute pass.
    let desc = wgpu::CommandEncoderDescriptor {
        label: Some("particle-compute"),
    };
    let mut encoder = device.create_command_encoder(&desc);
    encoder.copy_buffer_to_buffer(
        &new_uniform_buffer,
        0,
        &compute.uniform_buffer,
        0,
        uniforms_size,
    );

    {
        let pass_desc = wgpu::ComputePassDescriptor {
            label: Some("nannou-wgpu_compute_shader-compute_pass"),
        };
        let mut cpass = encoder.begin_compute_pass(&pass_desc);
        cpass.set_pipeline(&compute.pipeline);
        cpass.set_bind_group(0, &compute.bind_group, &[]);
        cpass.dispatch(PARTICLE_COUNT as u32, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &compute.particle_buffer,
        0,
        &read_buffer,
        0,
        compute.particle_buffer_size,
    );

    // Submit the compute pass to the device's queue.
    window.queue().submit(Some(encoder.finish()));

    // Spawn a future that reads the result of the compute pass.
    let particles = model.particles.clone();
    let future = async move {
        println!("reading buffer");
        let slice = read_buffer.slice(..);
        println!("mapping");
        if let Ok(_) = slice.map_async(wgpu::MapMode::Read).await {
            println!("locking particles");
            if let Ok(mut particles) = particles.lock() {
                println!("locked");
                let bytes = &slice.get_mapped_range()[..];
                println!("read bytes");
                // "Cast" the slice of bytes to a slice of particles as required.
                let p = {
                    let len = bytes.len() / std::mem::size_of::<Particle>();
                    let ptr = bytes.as_ptr() as *const Particle;
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                };
                println!("casted and copying");
                println!("p length: {:?}", p.len());
                println!("particles length: {:?}", particles.len());
                particles.copy_from_slice(p);
                println!("done");
            }
        }
    };
    model.threadpool.spawn_ok(future);
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(BLACK);
    let draw = app.draw();

    if let Ok(particles) = model.particles.lock() {
        for &p in particles.iter() {
            draw.ellipse().color(WHITE).x_y(p.position.x, p.position.y);
        }
    }

    draw.to_frame(app, &frame).unwrap();
}

fn create_uniforms() -> Uniforms {
    Uniforms {
        size: pt2(WIDTH as f32, HEIGHT as f32),
        speed: 1.0,
        particle_count: PARTICLE_COUNT,
        attraction_strength: 1.0,
        repulsion_strength: 1.0,
    }
}

fn uniforms_as_bytes(uniforms: &Uniforms) -> &[u8] {
    unsafe { wgpu::bytes::from(uniforms) }
}

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage_dynamic = false;
    let storage_readonly = false;
    let uniform_dynamic = false;
    wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(
            wgpu::ShaderStages::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
        .uniform_buffer(wgpu::ShaderStages::COMPUTE, uniform_dynamic)
        .build(device)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    oscillator_buffer: &wgpu::Buffer,
    oscillator_buffer_size: wgpu::BufferAddress,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let buffer_size_bytes = std::num::NonZeroU64::new(oscillator_buffer_size).unwrap();
    wgpu::BindGroupBuilder::new()
        .buffer_bytes(oscillator_buffer, 0, Some(buffer_size_bytes))
        .buffer::<Uniforms>(uniform_buffer, 0..1)
        .build(device, layout)
}

fn create_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("nannou"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    cs_mod: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let desc = wgpu::ComputePipelineDescriptor {
        label: Some("nannou"),
        layout: Some(layout),
        module: &cs_mod,
        entry_point: "main",
    };
    device.create_compute_pipeline(&desc)
}
