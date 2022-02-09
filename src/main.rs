use nannou::prelude::*;
use particles::ParticleSystem;

mod capture;
mod compute;
mod particles;
mod render;
mod uniforms;
mod util;

struct Model {
    particle_system: ParticleSystem,
    uniforms: uniforms::UniformBuffer,
    frame_capturer: capture::FrameCapturer,
    render: render::CustomRenderer,
}

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const PARTICLE_COUNT: u32 = 1500;

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
    let device = window.device();
    let sample_count = window.msaa_samples();

    println!("creating uniforms");

    // Create the buffer that will store the uniforms.
    let uniforms =
        uniforms::UniformBuffer::new(device, PARTICLE_COUNT, WIDTH as f32, HEIGHT as f32);

    println!("creating particle system");

    let particle_system =
        particles::ParticleSystem::new(app, device, &uniforms, WIDTH as f32 * 0.1);

    println!("finalizing reasources");

    let frame_capturer = capture::FrameCapturer::new(app);

    println!("loading shaders");
    let vs_mod = util::compile_shader(app, device, "shader.vert", shaderc::ShaderKind::Vertex);
    let fs_mod = util::compile_shader(app, device, "shader.frag", shaderc::ShaderKind::Fragment);

    let render = render::CustomRenderer::new::<uniforms::Uniforms>(
        device,
        &vs_mod,
        &fs_mod,
        Some(&vec![&particle_system.position_out_buffer]),
        Some(&vec![&particle_system.buffer_size]),
        None,
        None,
        Some(&uniforms.buffer),
        WIDTH,
        HEIGHT,
        sample_count,
        sample_count,
    )
    .unwrap();

    Model {
        particle_system,
        uniforms,
        frame_capturer,
        render,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let window = app.main_window();
    let device = window.device();

    // The encoder we'll use to encode the compute pass.
    let desc = wgpu::CommandEncoderDescriptor {
        label: Some("particle-compute"),
    };
    let mut encoder = device.create_command_encoder(&desc);

    model.uniforms.update(device, &mut encoder);

    model.particle_system.update(&mut encoder);

    model.render.render(&mut encoder);

    encoder.copy_buffer_to_buffer(
        &model.particle_system.position_out_buffer,
        0,
        &model.particle_system.position_in_buffer,
        0,
        model.particle_system.buffer_size,
    );

    encoder.copy_buffer_to_buffer(
        &model.particle_system.velocity_out_buffer,
        0,
        &model.particle_system.velocity_in_buffer,
        0,
        model.particle_system.buffer_size,
    );

    model
        .frame_capturer
        .take_snapshot(device, &mut encoder, &model.render.output_texture);

    // Submit the compute pass to the device's queue.
    window.queue().submit(Some(encoder.finish()));

    model.frame_capturer.save_frame(app);
}

fn view(_app: &App, model: &Model, frame: Frame) {
    let mut encoder = frame.command_encoder();
    model
        .render
        .texture_reshaper
        .encode_render_pass(frame.texture_view(), &mut *encoder);
}
