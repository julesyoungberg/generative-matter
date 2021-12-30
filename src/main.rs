use glsl_layout::float;
use glsl_layout::*;
use nannou::prelude::*;
use nannou::ui::prelude::*;
use nannou::ui::DrawToFrameError;
use nannou::wgpu::BufferInitDescriptor;
use rand;
use rand::Rng;
use std::sync::{Arc, Mutex};

mod components;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Particle {
    position: Vec2,
    velocity: Vec2,
}

widget_ids! {
    /// UI widget ids
    pub struct WidgetIds {
        controls_container,
        controls_wrapper,
        speed,
        attraction_strength,
        attraction_range,
        repulsion_strength,
        repulsion_range,
        center_strength,
        particle_radius,
        collision_response,
        momentum,
        max_acceleration,
        max_velocity,
    }
}

struct Model {
    compute: Compute,
    uniforms: Uniforms,
    positions: Arc<Mutex<Vec<Vec2>>>,
    threadpool: futures::executor::ThreadPool,
    widget_ids: WidgetIds,
    ui: Ui,
    window_id: WindowId,
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
#[derive(Debug, Default, Clone, Copy, Uniform)]
pub struct Uniforms {
    particle_count: uint,
    width: float,
    height: float,
    speed: float,
    attraction_strength: float,
    repulsion_strength: float,
    attraction_range: float,
    repulsion_range: float,
    center_strength: float,
    particle_radius: float,
    collision_response: float,
    momentum: float,
    max_acceleration: float,
    max_velocity: float,
}

const WIDTH: u32 = 1080;
const HEIGHT: u32 = 1080;
const PARTICLE_COUNT: u32 = 5000;

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

    // The vector that we will write particle values to.
    let mut positions = vec![];
    let mut velocities = vec![];

    let qwidth = WIDTH as f32 * 0.25;
    let qheight = HEIGHT as f32 * 0.25;
    for _ in 0..PARTICLE_COUNT {
        let position_x = rand::thread_rng().gen_range(-qwidth, qwidth);
        let position_y = rand::thread_rng().gen_range(-qheight, qheight);
        let position = pt2(position_x, position_y);
        positions.push(position);

        let velocity_x = rand::thread_rng().gen_range(-1.0, 1.0);
        let velocity_y = rand::thread_rng().gen_range(-1.0, 1.0);
        velocities.push(pt2(velocity_x, velocity_y));
    }

    let position_bytes = vectors_as_byte_vec(&positions);
    let velocity_bytes = vectors_as_byte_vec(&velocities);

    // Create the buffers that will store the result of our compute operation.
    let buffer_size =
        (PARTICLE_COUNT as usize * std::mem::size_of::<Vec2>()) as wgpu::BufferAddress;

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

    // Create the buffer that will store the uniforms.
    let uniforms = create_uniforms();
    println!("uniforms: {:?}", uniforms);
    let std140_uniforms = uniforms.std140();
    let uniforms_bytes = std140_uniforms.as_raw();
    let usage = wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST;
    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("uniform-buffer"),
        contents: uniforms_bytes,
        usage,
    });

    // Create the compute shader module.
    let cs_mod = wgpu::shader_from_spirv_bytes(device, include_bytes!("shaders/comp.spv"));

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

    // Create a thread pool capable of running our GPU buffer read futures.
    let threadpool = futures::executor::ThreadPool::new().unwrap();

    // create UI
    let mut ui = app.new_ui().build().unwrap();
    let widget_ids = WidgetIds::new(ui.widget_id_generator());

    Model {
        compute,
        uniforms,
        positions: Arc::new(Mutex::new(positions)),
        threadpool,
        widget_ids,
        ui,
        window_id,
    }
}

fn update_ui(model: &mut Model) {
    let ui = &mut model.ui.set_widgets();
    let height = 420.0;

    components::container([220.0, height])
        .top_left_with_margin(10.0)
        .set(model.widget_ids.controls_container, ui);

    components::wrapper([200.0, height - 20.0])
        .parent(model.widget_ids.controls_container)
        .top_left_with_margin(10.0)
        .set(model.widget_ids.controls_wrapper, ui);

    if let Some(value) = components::slider(model.uniforms.speed, 0.0, 1.0)
        .parent(model.widget_ids.controls_wrapper)
        .top_left()
        .label("Speed")
        .set(model.widget_ids.speed, ui)
    {
        model.uniforms.speed = value;
    }

    if let Some(value) = components::slider(model.uniforms.attraction_strength, 0.0, 200.0)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Attraction Strength")
        .set(model.widget_ids.attraction_strength, ui)
    {
        model.uniforms.attraction_strength = value;
    }

    if let Some(value) = components::slider(model.uniforms.repulsion_strength, 0.0, 200.0)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Repulsion Strength")
        .set(model.widget_ids.repulsion_strength, ui)
    {
        model.uniforms.repulsion_strength = value;
    }

    let max_range = WIDTH.min(HEIGHT) as f32;

    if let Some(value) = components::slider(model.uniforms.attraction_range, 0.0, max_range)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Attraction Range")
        .set(model.widget_ids.attraction_range, ui)
    {
        model.uniforms.attraction_range = value;
    }

    if let Some(value) = components::slider(model.uniforms.repulsion_range, 0.0, max_range)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Repulsion Range")
        .set(model.widget_ids.repulsion_range, ui)
    {
        model.uniforms.repulsion_range = value;
    }

    if let Some(value) = components::slider(model.uniforms.center_strength, 0.0, 1.0)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Center Strength")
        .set(model.widget_ids.center_strength, ui)
    {
        model.uniforms.center_strength = value;
    }

    if let Some(value) = components::slider(model.uniforms.particle_radius, 0.0, 10.0)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Particle Radius")
        .set(model.widget_ids.particle_radius, ui)
    {
        model.uniforms.particle_radius = value;
    }

    if let Some(value) = components::slider(model.uniforms.collision_response, 0.0, 1.5)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Collision Response")
        .set(model.widget_ids.collision_response, ui)
    {
        model.uniforms.collision_response = value;
    }

    if let Some(value) = components::slider(model.uniforms.momentum, 0.0, 1.0)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Momentum")
        .set(model.widget_ids.momentum, ui)
    {
        model.uniforms.momentum = value;
    }

    if let Some(value) = components::slider(model.uniforms.max_acceleration, 0.0, 2.0)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Max Acceleration")
        .set(model.widget_ids.max_acceleration, ui)
    {
        model.uniforms.max_acceleration = value;
    }

    if let Some(value) = components::slider(model.uniforms.max_velocity, 0.0, 2.0)
        .parent(model.widget_ids.controls_wrapper)
        .down(10.0)
        .label("Max Velocity")
        .set(model.widget_ids.max_velocity, ui)
    {
        model.uniforms.max_velocity = value;
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    update_ui(model);

    let window = app.main_window();
    let device = window.swap_chain_device();
    let compute = &mut model.compute;

    // create a buffer for reading the particle positions
    let read_position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read-positions"),
        size: compute.buffer_size,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    // An update for the uniform buffer with the current time.
    let std140_uniforms = model.uniforms.std140();
    let uniforms_bytes = std140_uniforms.as_raw();
    let uniforms_size = uniforms_bytes.len();
    let usage = wgpu::BufferUsage::COPY_SRC;
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
        uniforms_size as u64,
    );

    {
        let pass_desc = wgpu::ComputePassDescriptor {
            label: Some("compute-pass"),
        };
        let mut cpass = encoder.begin_compute_pass(&pass_desc);
        cpass.set_pipeline(&compute.pipeline);
        cpass.set_bind_group(0, &compute.bind_group, &[]);
        cpass.dispatch(PARTICLE_COUNT, 1, 1);
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

fn draw_ui(app: &App, model: &Model, frame: &Frame) {
    let color_attachment_desc = frame.color_attachment_descriptor();
    let primitives = model.ui.draw();
    let window = app
        .window(model.window_id)
        .ok_or(DrawToFrameError::InvalidWindow)
        .unwrap();
    let mut ui_encoder = frame.command_encoder();
    ui::encode_render_pass(
        &model.ui,
        &window,
        primitives,
        color_attachment_desc,
        &mut *ui_encoder,
    )
    .unwrap();
}

fn view(app: &App, model: &Model, frame: Frame) {
    frame.clear(BLACK);
    let draw = app.draw();

    if let Ok(positions) = model.positions.lock() {
        for &p in positions.iter() {
            draw.ellipse()
                .radius(model.uniforms.particle_radius)
                .color(WHITE)
                .x_y(p.x, p.y);
        }
    }

    draw.to_frame(app, &frame).unwrap();

    draw_ui(app, model, &frame);
}

fn create_uniforms() -> Uniforms {
    Uniforms {
        particle_count: PARTICLE_COUNT,
        width: WIDTH as f32,
        height: HEIGHT as f32,
        speed: 1.0,
        attraction_strength: 32.0,
        repulsion_strength: 30.0,
        attraction_range: 13.0, // 0.045,
        repulsion_range: 20.0,  // 1.2,
        center_strength: 0.0001,
        particle_radius: 2.0,
        collision_response: 0.5,
        momentum: 0.8,
        max_acceleration: 0.0,
        max_velocity: 1.0,
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

fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage_dynamic = false;
    let storage_readonly = false;
    let uniform_dynamic = false;
    wgpu::BindGroupLayoutBuilder::new()
        .storage_buffer(
            wgpu::ShaderStage::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
        .storage_buffer(
            wgpu::ShaderStage::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
        .storage_buffer(
            wgpu::ShaderStage::COMPUTE,
            storage_dynamic,
            storage_readonly,
        )
        .uniform_buffer(wgpu::ShaderStage::COMPUTE, uniform_dynamic)
        .build(device)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    position_buffer_in: &wgpu::Buffer,
    position_buffer_out: &wgpu::Buffer,
    velocity_buffer: &wgpu::Buffer,
    buffer_size: wgpu::BufferAddress,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let buffer_size_bytes = std::num::NonZeroU64::new(buffer_size).unwrap();
    wgpu::BindGroupBuilder::new()
        .buffer_bytes(position_buffer_in, 0, Some(buffer_size_bytes))
        .buffer_bytes(position_buffer_out, 0, Some(buffer_size_bytes))
        .buffer_bytes(velocity_buffer, 0, Some(buffer_size_bytes))
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
