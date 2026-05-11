#![allow(dead_code)]

use minifb::{Key, MouseButton, MouseMode, Scale, Window, WindowOptions};
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::path::{Path, PathBuf};

const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: usize = 1280;
const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;
const MAX_DEPTH: usize = 18;
const RR_MIN_DEPTH: usize = 3;
const PATH_TRACING: bool = true; // false = direct light sampling only (no indirect bounces)
const MOVE_SPEED: f64 = 0.35;
const LOOK_SPEED: f64 = 0.035;
const MOUSE_LOOK_SPEED: f64 = 0.004;
const IMAGE_SAMPLES: usize = 12;
const IMAGE_OUTPUT_FILE: &str = "airplane_scene.ppm";
const AIRPLANE_OBJ_PATH: &str = "/Users/stan/Downloads/Airplane_v1_L1.123c4a6fedec-1680-4a36-a228-b0d440a4f280/11803_Airplane_v1_l1.obj";
// const AIRPLANE_OBJ_PATH: &str = "/Users/stan/Downloads/Model Rmk3/Rmk3.obj";

fn main() -> Result<(), Box<dyn Error>> {
    if env::args().any(|arg| arg == "--image") {
        render_image()?;
        return Ok(());
    }

    let scene = airplane_scene()?;
    let mut controller =
        CameraController::new(Point3::new(13.0, 2.0, 3.0), Point3::new(0.0, 0.0, 0.0));
    let mut camera = controller.camera();
    let mut sample_count = 0_usize;
    let mut accumulation = vec![Color::ZERO; IMAGE_WIDTH * IMAGE_HEIGHT];
    let mut framebuffer = vec![0_u32; IMAGE_WIDTH * IMAGE_HEIGHT];
    let mut window = Window::new(
        "Rust Path Tracer - WASD move, click-drag/arrow look, Space/Shift, R reset, Esc quit",
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
        if controller.update(&window) {
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
        );
        sample_count += 1;

        window.set_title(&format!(
            "Rust Path Tracer - samples: {sample_count} - WASD, drag/arrows, Space/Shift, R, Esc"
        ));
        window.update_with_buffer(&framebuffer, IMAGE_WIDTH, IMAGE_HEIGHT)?;
    }

    Ok(())
}

fn render_sample(
    scene: &Scene,
    camera: &Camera,
    sample_count: usize,
    accumulation: &mut [Color],
    framebuffer: &mut [u32],
) {
    let scale = 1.0 / (sample_count + 1) as f64;

    accumulation
        .par_chunks_mut(IMAGE_WIDTH)
        .zip(framebuffer.par_chunks_mut(IMAGE_WIDTH))
        .enumerate()
        .for_each(|(y, (acc_row, fb_row))| {
            render_row(scene, camera, sample_count, scale, y, acc_row, fb_row);
        });
}

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
        render_sample(
            &scene,
            &camera,
            sample,
            &mut accumulation,
            &mut framebuffer,
        );
    }

    write_ppm(IMAGE_OUTPUT_FILE, &framebuffer)?;
    eprintln!("wrote {IMAGE_OUTPUT_FILE}");
    Ok(())
}

fn write_ppm(path: &str, framebuffer: &[u32]) -> std::io::Result<()> {
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

fn render_row(
    scene: &Scene,
    camera: &Camera,
    sample_count: usize,
    scale: f64,
    y: usize,
    acc_row: &mut [Color],
    fb_row: &mut [u32],
) {
    for (x, (pixel, packed)) in acc_row.iter_mut().zip(fb_row.iter_mut()).enumerate() {
        let mut rng = Rng::for_pixel(x, y, sample_count);
        let u = (x as f64 + rng.next_f64()) / (IMAGE_WIDTH - 1) as f64;
        let v = 1.0 - (y as f64 + rng.next_f64()) / (IMAGE_HEIGHT - 1) as f64;
        *pixel += trace(camera.ray(u, v), scene, MAX_DEPTH, &mut rng);
        *packed = color_to_u32_scaled(*pixel, scale);
    }
}

fn airplane_scene() -> Result<Scene, Box<dyn Error>> {
    let mut scene = Scene::new();

    let ground = Material::lambertian(Color::new(0.18, 0.20, 0.22));
    scene.add_sphere(Sphere::new(Point3::new(0.0, -1000.0, 0.0), 1000.0, ground));

    let mesh = load_obj_mesh(
        Path::new(AIRPLANE_OBJ_PATH),
        Transform::new(
            Point3::new(0.0, 1.05, 0.0),
            0.0026,
            Vec3::new(0.0, -0.25, 0.0),
            Vec3::new(342.674, 0.0, -235.508),
        ),
    )?;
    scene.add_mesh(mesh);

    // Key light: bright warm area light above and to the right
    let key = Material::emissive(Color::new(18.0, 14.0, 9.0));
    scene.add_light(Sphere::new(Point3::new(6.0, 10.0, -4.0), 2.5, key));

    // Fill light: cool blue-purple rim light on the opposite side
    let fill = Material::emissive(Color::new(3.0, 4.0, 9.0));
    scene.add_light(Sphere::new(Point3::new(-8.0, 7.0, 4.0), 1.5, fill));

    Ok(scene)
}

fn add_polygon_car(scene: &mut Scene) {
    let red = Material::lambertian(Color::new(0.78, 0.06, 0.04));
    let glass = Material::dielectric(1.45);
    let rubber = Material::lambertian(Color::new(0.02, 0.02, 0.02));
    let hub = Material::metal(Color::new(0.75, 0.72, 0.66), 0.15);

    add_rounded_ellipsoid_mesh(
        scene,
        Point3::new(0.0, 0.78, 0.0),
        Vec3::new(2.9, 0.55, 1.05),
        red,
        18,
        36,
    );
    add_rounded_ellipsoid_mesh(
        scene,
        Point3::new(-0.35, 1.28, 0.0),
        Vec3::new(1.15, 0.46, 0.78),
        glass,
        12,
        28,
    );

    for x in [-1.75, 1.75] {
        for z in [-0.93, 0.93] {
            add_cylinder_mesh(scene, Point3::new(x, 0.42, z), 0.38, 0.28, rubber, 28);
            add_cylinder_mesh(scene, Point3::new(x, 0.42, z), 0.18, 0.31, hub, 20);
        }
    }

    add_triangle_pair(
        scene,
        Point3::new(-2.75, 0.58, -0.62),
        Point3::new(-2.75, 0.58, 0.62),
        Point3::new(-2.25, 0.82, 0.42),
        Point3::new(-2.25, 0.82, -0.42),
        red,
    );
}

fn add_triangle_pair(
    scene: &mut Scene,
    a: Point3,
    b: Point3,
    c: Point3,
    d: Point3,
    material: Material,
) {
    scene.add_triangle(Triangle::flat(a, b, c, material));
    scene.add_triangle(Triangle::flat(a, c, d, material));
}

fn add_rounded_ellipsoid_mesh(
    scene: &mut Scene,
    center: Point3,
    radii: Vec3,
    material: Material,
    lat_steps: usize,
    lon_steps: usize,
) {
    for lat in 0..lat_steps {
        let theta0 = std::f64::consts::PI * lat as f64 / lat_steps as f64;
        let theta1 = std::f64::consts::PI * (lat + 1) as f64 / lat_steps as f64;

        for lon in 0..lon_steps {
            let phi0 = 2.0 * std::f64::consts::PI * lon as f64 / lon_steps as f64;
            let phi1 = 2.0 * std::f64::consts::PI * (lon + 1) as f64 / lon_steps as f64;
            let (p00, n00) = ellipsoid_vertex(center, radii, theta0, phi0);
            let (p01, n01) = ellipsoid_vertex(center, radii, theta0, phi1);
            let (p10, n10) = ellipsoid_vertex(center, radii, theta1, phi0);
            let (p11, n11) = ellipsoid_vertex(center, radii, theta1, phi1);

            if lat > 0 {
                scene.add_triangle(Triangle::smooth(p00, p10, p11, n00, n10, n11, material));
            }
            if lat + 1 < lat_steps {
                scene.add_triangle(Triangle::smooth(p00, p11, p01, n00, n11, n01, material));
            }
        }
    }
}

fn ellipsoid_vertex(center: Point3, radii: Vec3, theta: f64, phi: f64) -> (Point3, Vec3) {
    let unit = Vec3::new(
        theta.sin() * phi.cos(),
        theta.cos(),
        theta.sin() * phi.sin(),
    );
    let point = center + Vec3::new(unit.x * radii.x, unit.y * radii.y, unit.z * radii.z);
    let normal = Vec3::new(unit.x / radii.x, unit.y / radii.y, unit.z / radii.z).unit();
    (point, normal)
}

fn add_cylinder_mesh(
    scene: &mut Scene,
    center: Point3,
    radius: f64,
    width: f64,
    material: Material,
    segments: usize,
) {
    let half = width * 0.5;
    let left_center = center + Vec3::new(0.0, 0.0, -half);
    let right_center = center + Vec3::new(0.0, 0.0, half);

    for i in 0..segments {
        let a0 = 2.0 * std::f64::consts::PI * i as f64 / segments as f64;
        let a1 = 2.0 * std::f64::consts::PI * (i + 1) as f64 / segments as f64;
        let n0 = Vec3::new(a0.cos(), a0.sin(), 0.0);
        let n1 = Vec3::new(a1.cos(), a1.sin(), 0.0);
        let l0 = left_center + n0 * radius;
        let l1 = left_center + n1 * radius;
        let r0 = right_center + n0 * radius;
        let r1 = right_center + n1 * radius;

        scene.add_triangle(Triangle::smooth(l0, r0, r1, n0, n0, n1, material));
        scene.add_triangle(Triangle::smooth(l0, r1, l1, n0, n1, n1, material));
        scene.add_triangle(Triangle::flat(left_center, l1, l0, material));
        scene.add_triangle(Triangle::flat(right_center, r0, r1, material));
    }
}

#[derive(Clone, Copy)]
struct Transform {
    center: Point3,
    scale: f64,
    rotation: Vec3,
    source_offset: Vec3,
}

impl Transform {
    fn new(center: Point3, scale: f64, rotation: Vec3, source_offset: Vec3) -> Self {
        Self {
            center,
            scale,
            rotation,
            source_offset,
        }
    }

    fn point(self, point: Point3) -> Point3 {
        self.center
            + rotate_xyz(
                remap_obj_axes(point + self.source_offset) * self.scale,
                self.rotation,
            )
    }

    fn normal(self, normal: Vec3) -> Vec3 {
        rotate_xyz(remap_obj_axes(normal), self.rotation).unit()
    }
}

fn remap_obj_axes(value: Vec3) -> Vec3 {
    Vec3::new(value.x, value.z, value.y)
}

fn rotate_xyz(point: Vec3, rotation: Vec3) -> Vec3 {
    let (sin_x, cos_x) = rotation.x.sin_cos();
    let (sin_y, cos_y) = rotation.y.sin_cos();
    let (sin_z, cos_z) = rotation.z.sin_cos();

    let point = Vec3::new(
        point.x,
        point.y * cos_x - point.z * sin_x,
        point.y * sin_x + point.z * cos_x,
    );
    let point = Vec3::new(
        point.x * cos_y + point.z * sin_y,
        point.y,
        -point.x * sin_y + point.z * cos_y,
    );
    Vec3::new(
        point.x * cos_z - point.y * sin_z,
        point.x * sin_z + point.y * cos_z,
        point.z,
    )
}

fn load_obj_mesh(path: &Path, transform: Transform) -> Result<Mesh, Box<dyn Error>> {
    let obj = fs::read_to_string(path)?;
    let materials = load_mtl_materials(path)?;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut texcoords: Vec<[f64; 2]> = Vec::new();
    let mut triangles = Vec::new();
    let mut current_material = Material::lambertian(Color::new(0.78, 0.78, 0.74));

    for line in obj.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let x = parse_f64(parts.next())?;
                let y = parse_f64(parts.next())?;
                let z = parse_f64(parts.next())?;
                vertices.push(Point3::new(x, y, z));
            }
            Some("vn") => {
                let x = parse_f64(parts.next())?;
                let y = parse_f64(parts.next())?;
                let z = parse_f64(parts.next())?;
                normals.push(Vec3::new(x, y, z).unit());
            }
            Some("vt") => {
                let u = parse_f64(parts.next())?;
                let v = parse_f64(parts.next()).unwrap_or(0.0);
                texcoords.push([u, v]);
            }
            Some("usemtl") => {
                if let Some(name) = parts.next() {
                    current_material = materials
                        .get(name)
                        .copied()
                        .unwrap_or_else(|| material_for_name(name));
                }
            }
            Some("f") => {
                let corners = parts
                    .map(|part| {
                        parse_obj_corner(part, vertices.len(), texcoords.len(), normals.len())
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                if corners.len() < 3 {
                    continue;
                }

                for i in 1..corners.len() - 1 {
                    let a = corners[0];
                    let b = corners[i];
                    let c = corners[i + 1];
                    let pa = transform.point(vertices[a.vertex]);
                    let pb = transform.point(vertices[b.vertex]);
                    let pc = transform.point(vertices[c.vertex]);

                    let fallback = (pb - pa).cross(pc - pa).unit();
                    let na = a
                        .normal
                        .and_then(|index| normals.get(index).copied())
                        .map(|normal| transform.normal(normal))
                        .unwrap_or(fallback);
                    let nb = b
                        .normal
                        .and_then(|index| normals.get(index).copied())
                        .map(|normal| transform.normal(normal))
                        .unwrap_or(fallback);
                    let nc = c
                        .normal
                        .and_then(|index| normals.get(index).copied())
                        .map(|normal| transform.normal(normal))
                        .unwrap_or(fallback);

                    let uva = a.texture.and_then(|index| texcoords.get(index).copied());
                    let uvb = b.texture.and_then(|index| texcoords.get(index).copied());
                    let uvc = c.texture.and_then(|index| texcoords.get(index).copied());

                    triangles.push(
                        Triangle::smooth(pa, pb, pc, na, nb, nc, current_material)
                            .with_uvs([uva, uvb, uvc]),
                    );
                }
            }
            _ => {}
        }
    }

    eprintln!(
        "loaded {} vertices, {} normals, {} triangles from {}",
        vertices.len(),
        normals.len(),
        triangles.len(),
        path.display()
    );
    Ok(Mesh::new(triangles))
}

fn parse_f64(value: Option<&str>) -> Result<f64, Box<dyn Error>> {
    Ok(value.ok_or("missing float")?.parse()?)
}

#[derive(Clone, Copy)]
struct ObjCorner {
    vertex: usize,
    texture: Option<usize>,
    normal: Option<usize>,
}

fn parse_obj_corner(
    value: &str,
    vertex_count: usize,
    texture_count: usize,
    normal_count: usize,
) -> Result<ObjCorner, Box<dyn Error>> {
    let mut parts = value.split('/');
    let vertex = parse_obj_index(parts.next().ok_or("missing vertex index")?, vertex_count)?;
    let texture = match parts.next() {
        Some(index) if !index.is_empty() => Some(parse_obj_index(index, texture_count)?),
        _ => None,
    };
    let normal = match parts.next() {
        Some(index) if !index.is_empty() => Some(parse_obj_index(index, normal_count)?),
        _ => None,
    };

    Ok(ObjCorner {
        vertex,
        texture,
        normal,
    })
}

fn parse_obj_index(value: &str, len: usize) -> Result<usize, Box<dyn Error>> {
    let index = value.parse::<isize>()?;
    if index > 0 {
        Ok(index as usize - 1)
    } else if index < 0 {
        Ok((len as isize + index) as usize)
    } else {
        Err("OBJ indices are 1-based".into())
    }
}

fn load_mtl_materials(obj_path: &Path) -> Result<HashMap<String, Material>, Box<dyn Error>> {
    let mut materials = HashMap::new();
    let obj = fs::read_to_string(obj_path)?;
    let parent = obj_path.parent().unwrap_or_else(|| Path::new("."));

    for line in obj.lines() {
        let line = line.trim();
        let mut parts = line.split_whitespace();
        if parts.next() == Some("mtllib") {
            if let Some(mtl_file) = parts.next() {
                read_mtl(parent.join(mtl_file), &mut materials)?;
            }
        }
    }

    Ok(materials)
}

fn read_mtl(
    path: PathBuf,
    materials: &mut HashMap<String, Material>,
) -> Result<(), Box<dyn Error>> {
    let mtl = fs::read_to_string(&path)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut current: Option<String> = None;
    let mut tint: HashMap<String, Color> = HashMap::new();

    for line in mtl.lines() {
        let line = line.trim();
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("newmtl") => {
                let name = parts.next().map(str::to_owned);
                if let Some(name) = &name {
                    materials.insert(
                        name.clone(),
                        Material::lambertian(material_color_for_name(name)),
                    );
                }
                current = name;
            }
            Some("Kd") => {
                if let Some(name) = &current {
                    let r = parse_f64(parts.next())?;
                    let g = parse_f64(parts.next())?;
                    let b = parse_f64(parts.next())?;
                    let color = if r > 0.95 && g > 0.95 && b > 0.95 {
                        Color::new(1.0, 1.0, 1.0)
                    } else {
                        Color::new(r, g, b)
                    };
                    tint.insert(name.clone(), color);
                    if !matches!(materials.get(name), Some(Material::TexturedLambertian { .. })) {
                        let solid = if color.x >= 0.999 && color.y >= 0.999 && color.z >= 0.999 {
                            material_color_for_name(name)
                        } else {
                            color
                        };
                        materials.insert(name.clone(), Material::lambertian(solid));
                    }
                }
            }
            Some("map_Kd") => {
                if let Some(name) = &current {
                    let texture_file = parts.last().ok_or("missing texture path")?;
                    let texture_path = parent.join(texture_file);
                    match Texture::load(&texture_path) {
                        Ok(texture) => {
                            let texture: &'static Texture = Box::leak(Box::new(texture));
                            let tint_color =
                                tint.get(name).copied().unwrap_or(Color::new(1.0, 1.0, 1.0));
                            materials.insert(
                                name.clone(),
                                Material::TexturedLambertian {
                                    texture,
                                    tint: tint_color,
                                },
                            );
                            eprintln!(
                                "loaded texture {} ({}x{}) for material {}",
                                texture_path.display(),
                                texture.width,
                                texture.height,
                                name
                            );
                        }
                        Err(err) => {
                            eprintln!(
                                "failed to load texture {}: {err}",
                                texture_path.display()
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn material_for_name(name: &str) -> Material {
    Material::lambertian(material_color_for_name(name))
}

fn material_color_for_name(name: &str) -> Color {
    let color = if name.contains("body") {
        Color::new(0.86, 0.84, 0.78)
    } else if name.contains("tail") {
        Color::new(0.75, 0.16, 0.12)
    } else if name.contains("wing") {
        Color::new(0.66, 0.7, 0.74)
    } else {
        Color::new(0.75, 0.75, 0.72)
    };
    color
}

fn onb(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let bitangent = normal.cross(up).unit();
    let tangent = bitangent.cross(normal);
    (tangent, bitangent)
}

fn sample_lights(scene: &Scene, hit: &Hit, rng: &mut Rng) -> Color {
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

        if dist <= light.radius {
            continue;
        }

        let sin_theta_max_sq = (light.radius * light.radius / dist_sq).min(1.0);
        let cos_theta_max = (1.0 - sin_theta_max_sq).sqrt();
        let solid_angle = 2.0 * std::f64::consts::PI * (1.0 - cos_theta_max);

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

        let shadow_max = dist - light.radius - 0.001;
        if shadow_max > 0.001
            && scene
                .hit(Ray::new(hit.point, sample_dir), Interval::new(0.001, shadow_max))
                .is_some()
        {
            continue;
        }

        total += emission * (cos_i * solid_angle / std::f64::consts::PI);
    }

    total
}

fn trace(initial_ray: Ray, scene: &Scene, max_depth: usize, rng: &mut Rng) -> Color {
    let mut ray = initial_ray;
    let mut throughput = Color::new(1.0, 1.0, 1.0);
    let mut color = Color::ZERO;
    let mut specular_bounce = true;

    for depth in 0..max_depth {
        if let Some(hit) = scene.hit(ray, Interval::new(0.001, f64::INFINITY)) {
            // Emission visible directly or through specular bounces (avoids double-counting with NEE)
            if specular_bounce {
                color += throughput * hit.material.emission();
            }

            if let Some((scattered, attenuation)) = hit.material.scatter(ray, &hit, rng) {
                if hit.material.is_specular() {
                    specular_bounce = true;
                } else {
                    // NEE: sample lights explicitly for diffuse hits
                    color += throughput * attenuation * sample_lights(scene, &hit, rng);
                    specular_bounce = false;
                }

                throughput = throughput * attenuation;

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
                return color;
            }
        } else {
            let direction = ray.direction.unit();
            let y = direction.y;
            let sky = if y < 0.0 {
                Color::new(0.01, 0.008, 0.015)
            } else if y < 0.25 {
                let t = y / 0.25;
                Color::new(0.62, 0.20, 0.06) * (1.0 - t) + Color::new(0.22, 0.08, 0.20) * t
            } else {
                let t = (y - 0.25) / 0.75;
                Color::new(0.22, 0.08, 0.20) * (1.0 - t) + Color::new(0.02, 0.03, 0.14) * t
            };
            color += throughput * sky;
            return color;
        }
    }

    color
}

fn color_to_u32(pixel: Color, samples: usize) -> u32 {
    color_to_u32_scaled(pixel, 1.0 / samples as f64)
}

fn color_to_u32_scaled(pixel: Color, scale: f64) -> u32 {
    let r = to_u8(linear_to_gamma(pixel.x * scale)) as u32;
    let g = to_u8(linear_to_gamma(pixel.y * scale)) as u32;
    let b = to_u8(linear_to_gamma(pixel.z * scale)) as u32;
    (r << 16) | (g << 8) | b
}

fn linear_to_gamma(value: f64) -> f64 {
    value.max(0.0).sqrt()
}

fn to_u8(value: f64) -> u8 {
    (256.0 * value.clamp(0.0, 0.999)) as u8
}

fn hash_u64(a: u64, b: u64, salt: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= b.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= salt.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

type Point3 = Vec3;
type Color = Vec3;

impl Vec3 {
    const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    fn length_squared(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    fn axis(self, axis: usize) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    fn unit(self) -> Self {
        self / self.length()
    }

    fn near_zero(self) -> bool {
        const EPSILON: f64 = 1e-8;
        self.x.abs() < EPSILON && self.y.abs() < EPSILON && self.z.abs() < EPSILON
    }

    fn reflect(self, normal: Self) -> Self {
        self - normal * 2.0 * self.dot(normal)
    }

    fn refract(self, normal: Self, eta_ratio: f64) -> Self {
        let cos_theta = (-self).dot(normal).min(1.0);
        let perpendicular = (self + normal * cos_theta) * eta_ratio;
        let parallel = normal * -(1.0 - perpendicular.length_squared()).abs().sqrt();
        perpendicular + parallel
    }

    fn random_range(rng: &mut Rng, min: f64, max: f64) -> Self {
        Self::new(
            rng.range_f64(min, max),
            rng.range_f64(min, max),
            rng.range_f64(min, max),
        )
    }

    fn random_unit(rng: &mut Rng) -> Self {
        loop {
            let candidate = Self::random_range(rng, -1.0, 1.0);
            let length_squared = candidate.length_squared();
            if 1e-160 < length_squared && length_squared <= 1.0 {
                return candidate / length_squared.sqrt();
            }
        }
    }

    fn max_component(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }

    fn random_on_hemisphere(rng: &mut Rng, normal: Self) -> Self {
        let on_unit_sphere = Self::random_unit(rng);
        if on_unit_sphere.dot(normal) > 0.0 {
            on_unit_sphere
        } else {
            -on_unit_sphere
        }
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul for Vec3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy, Debug)]
struct Ray {
    origin: Point3,
    direction: Vec3,
}

impl Ray {
    fn new(origin: Point3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    fn at(self, t: f64) -> Point3 {
        self.origin + self.direction * t
    }
}

struct Camera {
    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    lower_left_corner: Point3,
}

impl Camera {
    fn new(
        origin: Point3,
        look_at: Point3,
        up: Vec3,
        vertical_fov: f64,
        aspect_ratio: f64,
    ) -> Self {
        let theta = vertical_fov.to_radians();
        let height = (theta / 2.0).tan() * 2.0;
        let viewport_height = height;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (origin - look_at).unit();
        let u = up.cross(w).unit();
        let v = w.cross(u);

        let horizontal = u * viewport_width;
        let vertical = v * viewport_height;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w;

        Self {
            origin,
            horizontal,
            vertical,
            lower_left_corner,
        }
    }

    fn ray(&self, u: f64, v: f64) -> Ray {
        Ray::new(
            self.origin,
            self.lower_left_corner + self.horizontal * u + self.vertical * v - self.origin,
        )
    }
}

struct CameraController {
    position: Point3,
    yaw: f64,
    pitch: f64,
    previous_mouse: Option<(f32, f32)>,
}

impl CameraController {
    fn new(position: Point3, look_at: Point3) -> Self {
        let direction = (look_at - position).unit();
        let yaw = direction.x.atan2(direction.z);
        let pitch = direction.y.asin();

        Self {
            position,
            yaw,
            pitch,
            previous_mouse: None,
        }
    }

    fn camera(&self) -> Camera {
        Camera::new(
            self.position,
            self.position + self.forward(),
            Vec3::new(0.0, 1.0, 0.0),
            20.0,
            ASPECT_RATIO,
        )
    }

    fn update(&mut self, window: &Window) -> bool {
        let previous_position = self.position;
        let previous_yaw = self.yaw;
        let previous_pitch = self.pitch;

        if window.is_key_down(Key::R) {
            *self = Self::new(Point3::new(13.0, 2.0, 3.0), Point3::new(0.0, 0.0, 0.0));
            return true;
        }

        if window.is_key_down(Key::Left) {
            self.yaw += LOOK_SPEED;
        }
        if window.is_key_down(Key::Right) {
            self.yaw -= LOOK_SPEED;
        }
        if window.is_key_down(Key::Up) {
            self.pitch += LOOK_SPEED;
        }
        if window.is_key_down(Key::Down) {
            self.pitch -= LOOK_SPEED;
        }

        if window.get_mouse_down(MouseButton::Left) {
            if let Some(mouse) = window.get_mouse_pos(MouseMode::Discard) {
                if let Some(previous_mouse) = self.previous_mouse {
                    let delta_x = mouse.0 - previous_mouse.0;
                    let delta_y = mouse.1 - previous_mouse.1;

                    self.yaw -= delta_x as f64 * MOUSE_LOOK_SPEED;
                    self.pitch -= delta_y as f64 * MOUSE_LOOK_SPEED;
                }

                self.previous_mouse = Some(mouse);
            }
        } else {
            self.previous_mouse = None;
        }

        let max_pitch = std::f64::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);

        let forward = self.forward();
        let right = forward.cross(Vec3::new(0.0, 1.0, 0.0)).unit();

        if window.is_key_down(Key::W) {
            self.position += forward * MOVE_SPEED;
        }
        if window.is_key_down(Key::S) {
            self.position += forward * -MOVE_SPEED;
        }
        if window.is_key_down(Key::D) {
            self.position += right * MOVE_SPEED;
        }
        if window.is_key_down(Key::A) {
            self.position += right * -MOVE_SPEED;
        }
        if window.is_key_down(Key::Space) {
            self.position += Vec3::new(0.0, MOVE_SPEED, 0.0);
        }
        // if window.is_key_down(Key::LeftShift) || window.is_key_down(Key::RightShift) {
        //     self.position += Vec3::new(0.0, -MOVE_SPEED, 0.0);
        // }

        (self.position - previous_position).length_squared() > 0.0
            || self.yaw != previous_yaw
            || self.pitch != previous_pitch
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
        .unit()
    }
}

#[derive(Clone, Copy)]
struct Interval {
    min: f64,
    max: f64,
}

impl Interval {
    fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    fn surrounds(self, value: f64) -> bool {
        self.min < value && value < self.max
    }
}

struct Hit {
    point: Point3,
    normal: Vec3,
    t: f64,
    front_face: bool,
    material: Material,
    uv: Option<[f64; 2]>,
}

impl Hit {
    fn new(ray: Ray, outward_normal: Vec3, t: f64, material: Material) -> Self {
        let front_face = ray.direction.dot(outward_normal) < 0.0;
        let normal = if front_face {
            outward_normal
        } else {
            -outward_normal
        };

        Self {
            point: ray.at(t),
            normal,
            t,
            front_face,
            material,
            uv: None,
        }
    }

    fn with_uv(mut self, uv: Option<[f64; 2]>) -> Self {
        self.uv = uv;
        self
    }
}

#[derive(Clone, Copy)]
struct Sphere {
    center: Point3,
    radius: f64,
    material: Material,
}

impl Sphere {
    fn new(center: Point3, radius: f64, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }

    fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
        let oc = self.center - ray.origin;
        let a = ray.direction.length_squared();
        let h = ray.direction.dot(oc);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = h * h - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrt_discriminant = discriminant.sqrt();
        let mut root = (h - sqrt_discriminant) / a;
        if !interval.surrounds(root) {
            root = (h + sqrt_discriminant) / a;
            if !interval.surrounds(root) {
                return None;
            }
        }

        let outward_normal = (ray.at(root) - self.center) / self.radius;
        Some(Hit::new(ray, outward_normal, root, self.material))
    }
}

#[derive(Clone, Copy)]
struct Triangle {
    vertices: [Point3; 3],
    normals: [Vec3; 3],
    uvs: [Option<[f64; 2]>; 3],
    material: Material,
}

impl Triangle {
    fn flat(a: Point3, b: Point3, c: Point3, material: Material) -> Self {
        let normal = (b - a).cross(c - a).unit();
        Self::smooth(a, b, c, normal, normal, normal, material)
    }

    fn smooth(
        a: Point3,
        b: Point3,
        c: Point3,
        normal_a: Vec3,
        normal_b: Vec3,
        normal_c: Vec3,
        material: Material,
    ) -> Self {
        Self {
            vertices: [a, b, c],
            normals: [normal_a.unit(), normal_b.unit(), normal_c.unit()],
            uvs: [None; 3],
            material,
        }
    }

    fn with_uvs(mut self, uvs: [Option<[f64; 2]>; 3]) -> Self {
        self.uvs = uvs;
        self
    }

    fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
        let edge1 = self.vertices[1] - self.vertices[0];
        let edge2 = self.vertices[2] - self.vertices[0];
        let ray_cross_edge2 = ray.direction.cross(edge2);
        let determinant = edge1.dot(ray_cross_edge2);

        if determinant.abs() < 1e-9 {
            return None;
        }

        let inverse_determinant = 1.0 / determinant;
        let origin_to_vertex = ray.origin - self.vertices[0];
        let u = origin_to_vertex.dot(ray_cross_edge2) * inverse_determinant;

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let origin_cross_edge1 = origin_to_vertex.cross(edge1);
        let v = ray.direction.dot(origin_cross_edge1) * inverse_determinant;

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = edge2.dot(origin_cross_edge1) * inverse_determinant;
        if !interval.surrounds(t) {
            return None;
        }

        let w = 1.0 - u - v;
        let normal = (self.normals[0] * w + self.normals[1] * u + self.normals[2] * v).unit();
        let uv = match (self.uvs[0], self.uvs[1], self.uvs[2]) {
            (Some(uv0), Some(uv1), Some(uv2)) => Some([
                uv0[0] * w + uv1[0] * u + uv2[0] * v,
                uv0[1] * w + uv1[1] * u + uv2[1] * v,
            ]),
            _ => None,
        };
        Some(Hit::new(ray, normal, t, self.material).with_uv(uv))
    }

    fn bounds(&self) -> Aabb {
        Aabb::from_points(self.vertices[0], self.vertices[1]).with_point(self.vertices[2])
    }

    fn centroid(&self) -> Point3 {
        (self.vertices[0] + self.vertices[1] + self.vertices[2]) / 3.0
    }
}

#[derive(Clone, Copy)]
struct Aabb {
    min: Point3,
    max: Point3,
}

impl Aabb {
    fn empty() -> Self {
        Self {
            min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    fn from_points(a: Point3, b: Point3) -> Self {
        Self {
            min: Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)),
            max: Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)),
        }
        .pad()
    }

    fn with_point(self, point: Point3) -> Self {
        Self {
            min: Point3::new(
                self.min.x.min(point.x),
                self.min.y.min(point.y),
                self.min.z.min(point.z),
            ),
            max: Point3::new(
                self.max.x.max(point.x),
                self.max.y.max(point.y),
                self.max.z.max(point.z),
            ),
        }
        .pad()
    }

    fn union(self, rhs: Self) -> Self {
        Self {
            min: Point3::new(
                self.min.x.min(rhs.min.x),
                self.min.y.min(rhs.min.y),
                self.min.z.min(rhs.min.z),
            ),
            max: Point3::new(
                self.max.x.max(rhs.max.x),
                self.max.y.max(rhs.max.y),
                self.max.z.max(rhs.max.z),
            ),
        }
    }

    fn longest_axis(self) -> usize {
        let span = self.max - self.min;
        if span.x > span.y && span.x > span.z {
            0
        } else if span.y > span.z {
            1
        } else {
            2
        }
    }

    fn hit(self, ray: Ray, inv_dir: Vec3, mut interval: Interval) -> bool {
        for axis in 0..3 {
            let inv = inv_dir.axis(axis);
            let mut t0 = (self.min.axis(axis) - ray.origin.axis(axis)) * inv;
            let mut t1 = (self.max.axis(axis) - ray.origin.axis(axis)) * inv;

            if inv < 0.0 {
                std::mem::swap(&mut t0, &mut t1);
            }

            interval.min = interval.min.max(t0);
            interval.max = interval.max.min(t1);

            if interval.max <= interval.min {
                return false;
            }
        }

        true
    }

    fn pad(self) -> Self {
        let delta = 0.0001;
        let x = if self.max.x - self.min.x < delta {
            delta
        } else {
            0.0
        };
        let y = if self.max.y - self.min.y < delta {
            delta
        } else {
            0.0
        };
        let z = if self.max.z - self.min.z < delta {
            delta
        } else {
            0.0
        };

        Self {
            min: self.min - Vec3::new(x, y, z),
            max: self.max + Vec3::new(x, y, z),
        }
    }
}

struct Mesh {
    triangles: Vec<Triangle>,
    nodes: Vec<BvhNode>,
}

impl Mesh {
    fn new(mut triangles: Vec<Triangle>) -> Self {
        let mut nodes = Vec::new();
        if !triangles.is_empty() {
            let len = triangles.len();
            build_bvh(&mut triangles, &mut nodes, 0, len);
        }

        eprintln!("built BVH with {} nodes", nodes.len());
        Self { triangles, nodes }
    }

    fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        let inv_dir = Vec3::new(
            1.0 / ray.direction.x,
            1.0 / ray.direction.y,
            1.0 / ray.direction.z,
        );

        let mut stack = [0usize; 64];
        let mut stack_top = 1usize;
        let mut nearest = interval.max;
        let mut result = None;

        while stack_top > 0 {
            stack_top -= 1;
            let node = &self.nodes[stack[stack_top]];

            if !node.bounds.hit(ray, inv_dir, Interval::new(interval.min, nearest)) {
                continue;
            }

            if let Some((start, end)) = node.range {
                for tri in &self.triangles[start..end] {
                    if let Some(hit) = tri.hit(ray, Interval::new(interval.min, nearest)) {
                        nearest = hit.t;
                        result = Some(hit);
                    }
                }
            } else {
                if let Some(right) = node.right {
                    stack[stack_top] = right;
                    stack_top += 1;
                }
                if let Some(left) = node.left {
                    stack[stack_top] = left;
                    stack_top += 1;
                }
            }
        }

        result
    }
}

#[derive(Clone, Copy)]
struct BvhNode {
    bounds: Aabb,
    left: Option<usize>,
    right: Option<usize>,
    range: Option<(usize, usize)>,
}

fn build_bvh(
    triangles: &mut [Triangle],
    nodes: &mut Vec<BvhNode>,
    start: usize,
    end: usize,
) -> usize {
    let node_index = nodes.len();
    nodes.push(BvhNode {
        bounds: triangle_bounds(&triangles[start..end]),
        left: None,
        right: None,
        range: None,
    });

    let count = end - start;
    if count <= 8 {
        nodes[node_index].range = Some((start, end));
        return node_index;
    }

    let centroid_bounds = centroid_bounds(&triangles[start..end]);
    let axis = centroid_bounds.longest_axis();
    triangles[start..end]
        .sort_by(|a, b| a.centroid().axis(axis).total_cmp(&b.centroid().axis(axis)));

    let mid = start + count / 2;
    let left = build_bvh(triangles, nodes, start, mid);
    let right = build_bvh(triangles, nodes, mid, end);
    nodes[node_index].left = Some(left);
    nodes[node_index].right = Some(right);
    node_index
}

fn triangle_bounds(triangles: &[Triangle]) -> Aabb {
    triangles.iter().fold(Aabb::empty(), |bounds, triangle| {
        bounds.union(triangle.bounds())
    })
}

fn centroid_bounds(triangles: &[Triangle]) -> Aabb {
    triangles.iter().fold(Aabb::empty(), |bounds, triangle| {
        bounds.with_point(triangle.centroid())
    })
}

enum Object {
    Sphere(Sphere),
    Triangle(Triangle),
    Mesh(Mesh),
}

impl Object {
    fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
        match self {
            Self::Sphere(sphere) => sphere.hit(ray, interval),
            Self::Triangle(triangle) => triangle.hit(ray, interval),
            Self::Mesh(mesh) => mesh.hit(ray, interval),
        }
    }
}

struct Scene {
    objects: Vec<Object>,
    lights: Vec<Sphere>,
}

impl Scene {
    fn new() -> Self {
        Self {
            objects: Vec::new(),
            lights: Vec::new(),
        }
    }

    fn add_sphere(&mut self, sphere: Sphere) {
        self.objects.push(Object::Sphere(sphere));
    }

    fn add_light(&mut self, sphere: Sphere) {
        self.lights.push(sphere);
        self.objects.push(Object::Sphere(sphere));
    }

    fn add_triangle(&mut self, triangle: Triangle) {
        self.objects.push(Object::Triangle(triangle));
    }

    fn add_mesh(&mut self, mesh: Mesh) {
        self.objects.push(Object::Mesh(mesh));
    }

    fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
        let mut nearest = interval.max;
        let mut result = None;

        for object in &self.objects {
            if let Some(hit) = object.hit(ray, Interval::new(interval.min, nearest)) {
                nearest = hit.t;
                result = Some(hit);
            }
        }

        result
    }
}

struct Texture {
    width: u32,
    height: u32,
    pixels: Vec<Color>,
}

impl Texture {
    fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let mut decoder = jpeg_decoder::Decoder::new(BufReader::new(file));
        let pixels = decoder.decode()?;
        let info = decoder.info().ok_or("missing JPEG info")?;
        let width = info.width as u32;
        let height = info.height as u32;

        let linear = match info.pixel_format {
            jpeg_decoder::PixelFormat::RGB24 => pixels
                .chunks_exact(3)
                .map(|p| Color::new(srgb_to_linear(p[0]), srgb_to_linear(p[1]), srgb_to_linear(p[2])))
                .collect::<Vec<_>>(),
            jpeg_decoder::PixelFormat::L8 => pixels
                .iter()
                .map(|&p| {
                    let v = srgb_to_linear(p);
                    Color::new(v, v, v)
                })
                .collect::<Vec<_>>(),
            other => return Err(format!("unsupported JPEG pixel format: {other:?}").into()),
        };

        Ok(Self {
            width,
            height,
            pixels: linear,
        })
    }

    fn sample(&self, uv: [f64; 2]) -> Color {
        let u = uv[0] - uv[0].floor();
        let v = uv[1] - uv[1].floor();
        let x = ((u * self.width as f64) as i64).clamp(0, self.width as i64 - 1) as u32;
        let y_flipped = 1.0 - v;
        let y = ((y_flipped * self.height as f64) as i64).clamp(0, self.height as i64 - 1) as u32;
        self.pixels[(y * self.width + x) as usize]
    }
}

fn srgb_to_linear(value: u8) -> f64 {
    let v = value as f64 / 255.0;
    v * v
}

#[derive(Clone, Copy)]
enum Material {
    Lambertian {
        albedo: Color,
    },
    TexturedLambertian {
        texture: &'static Texture,
        tint: Color,
    },
    Metal {
        albedo: Color,
        fuzz: f64,
    },
    Dielectric {
        refraction_index: f64,
    },
    Emissive {
        emission: Color,
    },
}

impl Material {
    fn lambertian(albedo: Color) -> Self {
        Self::Lambertian { albedo }
    }

    fn metal(albedo: Color, fuzz: f64) -> Self {
        Self::Metal {
            albedo,
            fuzz: fuzz.min(1.0),
        }
    }

    fn dielectric(refraction_index: f64) -> Self {
        Self::Dielectric { refraction_index }
    }

    fn emissive(emission: Color) -> Self {
        Self::Emissive { emission }
    }

    fn emission(self) -> Color {
        match self {
            Self::Emissive { emission } => emission,
            _ => Color::ZERO,
        }
    }

    fn is_specular(self) -> bool {
        matches!(self, Self::Metal { .. } | Self::Dielectric { .. })
    }

    fn scatter(self, ray: Ray, hit: &Hit, rng: &mut Rng) -> Option<(Ray, Color)> {
        match self {
            Self::Lambertian { albedo } => {
                let mut scatter_direction =
                    hit.normal + Vec3::random_on_hemisphere(rng, hit.normal);

                if scatter_direction.near_zero() {
                    scatter_direction = hit.normal;
                }

                Some((Ray::new(hit.point, scatter_direction), albedo))
            }
            Self::TexturedLambertian { texture, tint } => {
                let mut scatter_direction =
                    hit.normal + Vec3::random_on_hemisphere(rng, hit.normal);

                if scatter_direction.near_zero() {
                    scatter_direction = hit.normal;
                }

                let albedo = hit
                    .uv
                    .map(|uv| texture.sample(uv) * tint)
                    .unwrap_or(tint);
                Some((Ray::new(hit.point, scatter_direction), albedo))
            }
            Self::Metal { albedo, fuzz } => {
                let reflected = ray.direction.unit().reflect(hit.normal);
                let scattered = Ray::new(hit.point, reflected + Vec3::random_unit(rng) * fuzz);

                if scattered.direction.dot(hit.normal) > 0.0 {
                    Some((scattered, albedo))
                } else {
                    None
                }
            }
            Self::Dielectric { refraction_index } => {
                let attenuation = Color::new(1.0, 1.0, 1.0);
                let eta_ratio = if hit.front_face {
                    1.0 / refraction_index
                } else {
                    refraction_index
                };

                let unit_direction = ray.direction.unit();
                let cos_theta = (-unit_direction).dot(hit.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let cannot_refract = eta_ratio * sin_theta > 1.0;
                let direction =
                    if cannot_refract || reflectance(cos_theta, eta_ratio) > rng.next_f64() {
                        unit_direction.reflect(hit.normal)
                    } else {
                        unit_direction.refract(hit.normal, eta_ratio)
                    };

                Some((Ray::new(hit.point, direction), attenuation))
            }
            Self::Emissive { .. } => None,
        }
    }
}

fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
    let r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn for_pixel(x: usize, y: usize, sample: usize) -> Self {
        Self::new(hash_u64(
            x as u64,
            y as u64,
            sample as u64 ^ 0x7E57_5EED_F00D_BAAD,
        ))
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }
    
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }
}


