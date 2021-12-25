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
    velocities: Arc<Mutex<Vec<Vec2>>>,
    positions: Arc<Mutex<Vec<Vec2>>>,
    threadpool: futures::executor::ThreadPool,
}

struct Compute {
    position_buffer_in: wgpu::Buffer,
    position_buffer_out: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
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

    // Create the buffers that will store the result of our compute operation.
    let buffer_size =
        (PARTICLE_COUNT as usize * std::mem::size_of::<Vec2>()) as wgpu::BufferAddress;

    let position_buffer_in = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle-positions-1"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let position_buffer_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle-positions-2"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle-velocitiess"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::MAP_READ,
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
        &position_buffer_in,
        &position_buffer_out,
        &velocity_buffer,
        buffer_size,
        &uniform_buffer,
    );
    let pipeline_layout = create_pipeline_layout(device, &bind_group_layout);
    let pipeline = create_compute_pipeline(device, &pipeline_layout, &cs_mod);

    let compute = Compute {
        position_buffer_in,
        position_buffer_out,
        velocity_buffer,
        buffer_size,
        uniform_buffer,
        bind_group,
        pipeline,
    };

    // The vector that we will write particle values to.
    let positions = Arc::new(Mutex::new(vec![]));
    let velocities = Arc::new(Mutex::new(vec![]));

    if let Ok(mut pstns) = positions.lock() {
        let hwidth = WIDTH as f32 * 0.5;
        let hheight = HEIGHT as f32 * 0.5;
        let position_x = rand::thread_rng().gen_range(-hwidth, hwidth);
        let position_y = rand::thread_rng().gen_range(-hheight, hheight);
        pstns.push(pt2(position_x, position_y));
    }

    if let Ok(mut vlcts) = velocities.lock() {
        let velocity_x = rand::thread_rng().gen_range(-1.0, 1.0);
        let velocity_y = rand::thread_rng().gen_range(-1.0, 1.0);
        vlcts.push(pt2(velocity_x, velocity_y));
    }

    // Create a thread pool capable of running our GPU buffer read futures.
    let threadpool = futures::executor::ThreadPool::new().unwrap();

    Model {
        compute,
        positions,
        velocities,
        threadpool,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let window = app.main_window();
    let device = window.device();
    let compute = &mut model.compute;

    // create a buffer for reading the particle positions
    let read_position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-positions"),
        size: compute.buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // create a buffer for reading particle velocities
    let read_velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-velocities"),
        size: compute.buffer_size,
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
            label: Some("compute_pass"),
        };
        let mut cpass = encoder.begin_compute_pass(&pass_desc);
        cpass.set_pipeline(&compute.pipeline);
        cpass.set_bind_group(0, &compute.bind_group, &[]);
        cpass.dispatch(PARTICLE_COUNT as u32, 1, 1);
    }

    encoder.copy_buffer_to_buffer(
        &compute.position_buffer_out,
        0,
        &compute.position_buffer_in,
        0,
        compute.buffer_size,
    );

    encoder.copy_buffer_to_buffer(
        &compute.position_buffer_out,
        0,
        &read_position_buffer,
        0,
        compute.buffer_size,
    );

    encoder.copy_buffer_to_buffer(
        &compute.velocity_buffer,
        0,
        &read_velocity_buffer,
        0,
        compute.buffer_size,
    );

    // Submit the compute pass to the device's queue.
    window.queue().submit(Some(encoder.finish()));

    // Spawn a future that reads the result of the compute pass.
    let positions = model.positions.clone();
    let read_positions_future = async move {
        println!("reading position buffer");
        let slice = read_position_buffer.slice(..);
        println!("mapping position buffer");
        if let Ok(_) = slice.map_async(wgpu::MapMode::Read).await {
            println!("locking positions");
            if let Ok(mut positions) = positions.lock() {
                println!("locked positions");
                let bytes = &slice.get_mapped_range()[..];
                println!("read position bytes");
                // "Cast" the slice of bytes to a slice of Vec2 as required.
                let slice = {
                    let len = bytes.len() / std::mem::size_of::<Vec2>();
                    let ptr = bytes.as_ptr() as *const Vec2;
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                };

                println!("casted and copying");
                println!("slice length: {:?}", slice.len());
                println!("positions length: {:?}", positions.len());
                positions.copy_from_slice(slice);
                println!("done");
            }
        }
    };

    model.threadpool.spawn_ok(read_positions_future);

    let velocities = model.velocities.clone();
    let future = async move {
        println!("reading velocity buffer");
        let slice = read_velocity_buffer.slice(..);
        println!("mapping velocity buffer");
        if let Ok(_) = slice.map_async(wgpu::MapMode::Read).await {
            println!("locking velocities");
            if let Ok(mut velocities) = velocities.lock() {
                println!("locked velocities");
                let bytes = &slice.get_mapped_range()[..];
                println!("read position bytes");
                // "Cast" the slice of bytes to a slice of Vec2 as required.
                let slice = {
                    let len = bytes.len() / std::mem::size_of::<Vec2>();
                    let ptr = bytes.as_ptr() as *const Vec2;
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                };

                println!("casted and copying");
                println!("slice length: {:?}", slice.len());
                println!("velocities length: {:?}", velocities.len());
                velocities.copy_from_slice(slice);
                println!("done");
            }
        }
    };

    model.threadpool.spawn_ok(future);
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(BLACK);
    let draw = app.draw();

    if let Ok(positions) = model.positions.lock() {
        for &p in positions.iter() {
            draw.ellipse().color(WHITE).x_y(p.x, p.y);
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
        .storage_buffer(
            wgpu::ShaderStages::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
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
    position_buffer_1: &wgpu::Buffer,
    position_buffer_2: &wgpu::Buffer,
    velocity_buffer: &wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let buffer_size_bytes = std::num::NonZeroU64::new(buffer_size).unwrap();
    wgpu::BindGroupBuilder::new()
        .buffer_bytes(position_buffer_1, 0, Some(buffer_size_bytes))
        .buffer_bytes(position_buffer_2, 1, Some(buffer_size_bytes))
        .buffer_bytes(velocity_buffer, 2, Some(buffer_size_bytes))
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
