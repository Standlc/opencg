//! Path-tracing core: tracing a single ray, light sampling, and the
//! parallel image-rendering loop.

use std::fs::File;
use std::io::{BufWriter, Write};

use rayon::prelude::*;

use crate::camera::Camera;
use crate::config::{IMAGE_HEIGHT, IMAGE_WIDTH, MAX_DEPTH, PATH_TRACING, RR_MIN_DEPTH};
use crate::geometry::Hit;
use crate::material::Material;
use crate::math::{color_to_u32_scaled, onb, Color, Interval, Ray};
use crate::rng::Rng;
use crate::scene::Scene;

/// Renders one path-traced sample for every pixel in parallel and folds it
/// into `accumulation`, then writes the running average into `framebuffer`.
pub fn render_sample(
    scene: &Scene,
    camera: &Camera,
    sample_count: usize,
    accumulation: &mut [Color],
    framebuffer: &mut [u32],
) {
    let scale = 1.0 / (sample_count + 1) as f64;

    // Parallelise across rows — rayon handles the work-stealing.
    accumulation
        .par_chunks_mut(IMAGE_WIDTH)
        .zip(framebuffer.par_chunks_mut(IMAGE_WIDTH))
        .enumerate()
        .for_each(|(y, (acc_row, fb_row))| {
            render_row(scene, camera, sample_count, scale, y, acc_row, fb_row);
        });
}

/// Traces one new sample for every pixel in row `y`. `acc_row` accumulates
/// linear-space radiance; `fb_row` receives the gamma-corrected u32 packed
/// for display.
pub fn render_row(
    scene: &Scene,
    camera: &Camera,
    sample_count: usize,
    scale: f64,
    y: usize,
    acc_row: &mut [Color],
    fb_row: &mut [u32],
) {
    for (x, (pixel, packed)) in acc_row.iter_mut().zip(fb_row.iter_mut()).enumerate() {
        // Each (x, y, sample) gets its own deterministic RNG stream.
        let mut rng = Rng::for_pixel(x, y, sample_count);
        // Jittered subpixel sampling for antialiasing.
        let u = (x as f64 + rng.next_f64()) / (IMAGE_WIDTH - 1) as f64;
        let v = 1.0 - (y as f64 + rng.next_f64()) / (IMAGE_HEIGHT - 1) as f64;
        *pixel += trace(camera.ray(u, v), scene, MAX_DEPTH, &mut rng);
        *packed = color_to_u32_scaled(*pixel, scale);
    }
}

/// Russian-roulette + next-event-estimation path tracer.
///
/// Walks bounces up to `max_depth`. At each diffuse hit we sample lights
/// directly (NEE) instead of waiting for the random walk to find them.
/// Specular bounces still let emission accumulate naturally to avoid
/// double-counting via NEE.
pub fn trace(initial_ray: Ray, scene: &Scene, max_depth: usize, rng: &mut Rng) -> Color {
    let mut ray = initial_ray;
    let mut throughput = Color::new(1.0, 1.0, 1.0);
    let mut color = Color::ZERO;
    // True on the first bounce and after any specular bounce — those are the
    // bounces where hitting an emitter should add its emission directly.
    let mut specular_bounce = true;

    for depth in 0..max_depth {
        if let Some(hit) = scene.hit(ray, Interval::new(0.001, f64::INFINITY)) {
            if specular_bounce {
                color += throughput * hit.material.emission();
            }

            if let Some((scattered, attenuation)) = hit.material.scatter(ray, &hit, rng) {
                if hit.material.is_specular() {
                    specular_bounce = true;
                } else {
                    // Diffuse: sample lights explicitly. The direct contribution
                    // uses the local BRDF (attenuation) and the throughput so
                    // far.
                    color += throughput * attenuation * sample_lights(scene, &hit, rng);
                    specular_bounce = false;
                }

                throughput = throughput * attenuation;

                // Russian roulette: probabilistically terminate dim paths
                // after a few bounces, weighting survivors so the estimator
                // stays unbiased.
                if depth >= RR_MIN_DEPTH {
                    let survival = throughput.max_component().min(1.0).max(0.05);
                    if rng.next_f64() > survival {
                        return color;
                    }
                    throughput = throughput / survival;
                }

                if !PATH_TRACING {
                    return color;
                }
                ray = scattered;
            } else {
                // Absorbed (e.g. emissive surface scattered nothing back).
                return color;
            }
        } else {
            // Missed all geometry — add the procedural sky and stop.
            color += throughput * sky_color(ray);
            return color;
        }
    }

    color
}

/// Gradient sky used as the environment light. Warm at the horizon, deep
/// purple-blue overhead, and smoothly fading to black below the horizon.
fn sky_color(ray: Ray) -> Color {
    let direction = ray.direction.unit();
    let y = direction.y;

    let horizon = Color::new(0.62, 0.20, 0.06);
    let mid = Color::new(0.22, 0.08, 0.20);
    let zenith = Color::new(0.02, 0.03, 0.14);

    if y >= 0.0 {
        if y < 0.25 {
            // Warm horizon → purple band.
            let t = y / 0.25;
            horizon * (1.0 - t) + mid * t
        } else {
            // Purple band → deep zenith.
            let t = (y - 0.25) / 0.75;
            mid * (1.0 - t) + zenith * t
        }
    } else {
        // Below the horizon: smoothly fade the horizon color into black so
        // there's no visible seam where the ground (or empty space) begins.
        let t = y.abs().sqrt().clamp(0.0, 1.0);
        horizon * (1.0 - t)
    }
}

/// Next-event estimation: for every emissive sphere, sample a random direction
/// inside its solid-angle cone, shadow-test it, and accumulate the resulting
/// direct-lighting contribution.
pub fn sample_lights(scene: &Scene, hit: &Hit, rng: &mut Rng) -> Color {
    if scene.lights.is_empty() {
        return Color::ZERO;
    }

    let mut total = Color::ZERO;

    for light in &scene.lights {
        let emission = match light.material {
            Material::Emissive { emission } => emission,
            _ => continue,
        };

        let to_center = light.center - hit.point;
        let dist_sq = to_center.length_squared();
        let dist = dist_sq.sqrt();

        // We're inside the light — no useful sample.
        if dist <= light.radius {
            continue;
        }

        // Compute the solid angle the sphere subtends from the hit point.
        let sin_theta_max_sq = (light.radius * light.radius / dist_sq).min(1.0);
        let cos_theta_max = (1.0 - sin_theta_max_sq).sqrt();
        let solid_angle = 2.0 * std::f64::consts::PI * (1.0 - cos_theta_max);

        // Sample a direction uniformly inside that cone.
        let cos_theta = 1.0 - rng.next_f64() * (1.0 - cos_theta_max);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * std::f64::consts::PI * rng.next_f64();

        let dir_unit = to_center / dist;
        let (tangent, bitangent) = onb(dir_unit);
        let sample_dir = tangent * (sin_theta * phi.cos())
            + bitangent * (sin_theta * phi.sin())
            + dir_unit * cos_theta;

        let cos_i = sample_dir.dot(hit.normal);
        if cos_i <= 0.0 {
            continue;
        }

        // Shadow ray — only count this light if nothing blocks the path.
        let shadow_max = dist - light.radius - 0.001;
        if shadow_max > 0.001
            && scene
                .hit(
                    Ray::new(hit.point, sample_dir),
                    Interval::new(0.001, shadow_max),
                )
                .is_some()
        {
            continue;
        }

        // Lambert BRDF over PI, weighted by the cone's solid angle.
        total += emission * (cos_i * solid_angle / std::f64::consts::PI);
    }

    total
}

/// Writes a framebuffer to a plain ASCII PPM file (P3). Slow but trivially
/// portable — fine for one-off offline renders.
pub fn write_ppm(path: &str, framebuffer: &[u32]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "P3")?;
    writeln!(writer, "{IMAGE_WIDTH} {IMAGE_HEIGHT}")?;
    writeln!(writer, "255")?;

    for pixel in framebuffer {
        let r = (pixel >> 16) & 0xff;
        let g = (pixel >> 8) & 0xff;
        let b = pixel & 0xff;
        writeln!(writer, "{r} {g} {b}")?;
    }

    Ok(())
}
