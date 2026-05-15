//! Entry point for the path tracer.
//!
//! Two run modes:
//!   - `--image`             → render to a PPM file (offline batch mode)
//!   - default (no flag)     → open an interactive viewer window
//!
//! Pass `--airplane` to swap the GLB car scene for the OBJ airplane scene.

mod bvh;
mod camera;
mod config;
mod geometry;
mod loader;
mod material;
mod math;
mod mesh;
mod render;
mod rng;
mod scene;
mod scenes;

use std::env;
use std::error::Error;

use minifb::{Key, Scale, Window, WindowOptions};

use crate::camera::{Camera, CameraController};
use crate::config::{
    ASPECT_RATIO, IMAGE_HEIGHT, IMAGE_OUTPUT_FILE, IMAGE_SAMPLES, IMAGE_WIDTH,
};
use crate::math::{Color, Point3, Vec3};
use crate::render::{render_sample, write_ppm};
use crate::scenes::{airplane_scene, car_scene};

fn main() -> Result<(), Box<dyn Error>> {
    if env::args().any(|arg| arg == "--image") {
        return render_image();
    }
    run_viewer()
}

/// Builds the requested scene, opens a window, and continuously accumulates
/// path-traced samples while the user interacts with the camera.
fn run_viewer() -> Result<(), Box<dyn Error>> {
    let use_airplane = env::args().any(|arg: String| arg == "--airplane");
    let (scene, start_pos, look_at) = if use_airplane {
        (
            airplane_scene()?,
            Point3::new(26.0, 4.0, 0.0),
            Point3::new(0.0, 0.0, 0.0),
        )
    } else {
        (
            car_scene()?,
            Point3::new(16.0, 6.0, 24.0),
            Point3::new(0.0, 1.0, 0.0),
        )
    };

    let mut controller = CameraController::new(start_pos, look_at);
    let mut camera = controller.camera();
    let mut sample_count = 0_usize;
    let mut accumulation = vec![Color::ZERO; IMAGE_WIDTH * IMAGE_HEIGHT];
    let mut framebuffer = vec![0_u32; IMAGE_WIDTH * IMAGE_HEIGHT];

    let mut window = Window::new(
        "Rust Path Tracer - scroll zoom, left-drag orbit, right-drag pan, R reset, Esc quit",
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        WindowOptions {
            resize: false,
            scale: Scale::X1,
            ..WindowOptions::default()
        },
    )?;
    window.set_target_fps(60);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Reset accumulation any time the camera moves — otherwise old samples
        // would blend with new geometry and ghost.
        let camera_moving = controller.update(&window);
        if camera_moving {
            camera = controller.camera();
            sample_count = 0;
            accumulation.fill(Color::ZERO);
        }

        render_sample(
            &scene,
            &camera,
            sample_count,
            &mut accumulation,
            &mut framebuffer,
            !camera_moving,
        );
        sample_count += 1;

        window.set_title(&format!(
            "Rust Path Tracer - samples: {sample_count} - scroll zoom, left-drag orbit, right-drag pan, R, Esc"
        ));
        window.update_with_buffer(&framebuffer, IMAGE_WIDTH, IMAGE_HEIGHT)?;
    }

    Ok(())
}

/// Renders a fixed number of samples of the airplane scene to a PPM file. Used
/// when the binary is invoked with `--image`.
fn render_image() -> Result<(), Box<dyn Error>> {
    let scene = airplane_scene()?;
    let camera = Camera::new(
        Point3::new(8.0, 4.2, 10.0),
        Point3::new(0.0, 0.25, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        42.0,
        ASPECT_RATIO,
    );
    let mut accumulation = vec![Color::ZERO; IMAGE_WIDTH * IMAGE_HEIGHT];
    let mut framebuffer = vec![0_u32; IMAGE_WIDTH * IMAGE_HEIGHT];

    for sample in 0..IMAGE_SAMPLES {
        eprintln!("image sample {}/{}", sample + 1, IMAGE_SAMPLES);
        render_sample(&scene, &camera, sample, &mut accumulation, &mut framebuffer, true);
    }

    write_ppm(IMAGE_OUTPUT_FILE, &framebuffer)?;
    eprintln!("wrote {IMAGE_OUTPUT_FILE}");
    Ok(())
}
