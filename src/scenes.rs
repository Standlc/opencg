//! Pre-built scene constructors and small procedural-mesh helpers.
//!
//! `airplane_scene` / `car_scene` are the two scenes the binary can launch.
//! The procedural builders (cylinder, ellipsoid, triangle pair) are left here
//! as reusable utilities for sketching new scenes.

#![allow(dead_code)]

use std::error::Error;
use std::path::Path;

use crate::config::{AIRPLANE_OBJ_PATH, AUDI_GLB_PATH};
use crate::geometry::{Sphere, Triangle};
use crate::loader::obj::load_obj_mesh;
use crate::loader::glb::load_glb_mesh;
use crate::loader::Transform;
use crate::material::Material;
use crate::math::{Color, Point3, Vec3};
use crate::scene::Scene;

/// Builds the airplane test scene: ground sphere + OBJ airplane + warm key light.
pub fn airplane_scene() -> Result<Scene, Box<dyn Error>> {
    let mut scene = Scene::new();

    // Huge sphere standing in for the ground plane.
    let ground = Material::lambertian(Color::new(0.18, 0.20, 0.22));
    scene.add_sphere(Sphere::new(Point3::new(0.0, -1000.0, 0.0), 1000.0, ground));

    // Place the airplane in front of the camera. The transform compensates for
    // the OBJ's source orientation (Z-up) and recenters it on the origin.
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

    // Bright warm area light above and to the right.
    let key = Material::emissive(Color::new(18.0, 14.0, 9.0));
    scene.add_light(Sphere::new(Point3::new(6.0, 10.0, -4.0), 2.5, key));

    Ok(scene)
}

/// Builds the GLB car test scene: dark ground + GLB mesh + key/fill lights.
pub fn car_scene() -> Result<Scene, Box<dyn Error>> {
    let mut scene = Scene::new();

    let ground = Material::lambertian(Color::new(0.12, 0.13, 0.14));
    scene.add_sphere(Sphere::new(Point3::new(0.0, -1000.0, 0.0), 1000.0, ground));

    let mesh = load_glb_mesh(
        Path::new(AUDI_GLB_PATH),
        Point3::new(0.0, 0.0, 0.0),
        1.0,
        Vec3::new(0.0, 0.0, 0.0),
    )?;
    scene.add_mesh(mesh);

    // Warm key from upper right.
    let key = Material::emissive(Color::new(20.0, 16.0, 10.0));
    scene.add_light(Sphere::new(Point3::new(8.0, 12.0, -6.0), 3.0, key));
    // Cool fill from upper left to soften shadows.
    let fill = Material::emissive(Color::new(4.0, 5.0, 10.0));
    scene.add_light(Sphere::new(Point3::new(-8.0, 6.0, 8.0), 2.0, fill));

    Ok(scene)
}

// --- Procedural mesh helpers ---------------------------------------------
//
// None of these are currently invoked by `airplane_scene` / `car_scene`, but
// they're useful when sketching new scenes by hand.

/// Adds two coplanar triangles forming a quad (a, b, c, d) to the scene.
pub fn add_triangle_pair(
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

/// Tessellates an axis-aligned ellipsoid into smooth-shaded triangles using
/// `lat_steps * lon_steps` sampling. The two polar rings degenerate into
/// triangles instead of quads, which is why we skip one side at the poles.
pub fn add_rounded_ellipsoid_mesh(
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

/// One ellipsoid vertex (and its normalized surface normal) for the given
/// spherical angles.
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

/// Tessellates an axis-aligned (Z-direction) cylinder with smooth side faces
/// and flat triangular endcaps.
pub fn add_cylinder_mesh(
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

        // Side: two smooth-shaded triangles sharing the slant normals.
        scene.add_triangle(Triangle::smooth(l0, r0, r1, n0, n0, n1, material));
        scene.add_triangle(Triangle::smooth(l0, r1, l1, n0, n1, n1, material));
        // Endcaps: flat fans from each centre.
        scene.add_triangle(Triangle::flat(left_center, l1, l0, material));
        scene.add_triangle(Triangle::flat(right_center, r0, r1, material));
    }
}
