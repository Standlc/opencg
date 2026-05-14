//! The scene: a heterogenous collection of objects plus a separate list of
//! emissive spheres used as explicit light sources for next-event estimation.

use crate::geometry::{Hit, Sphere, Triangle};
use crate::math::{Interval, Ray};
use crate::mesh::Mesh;

/// A renderable object. We dispatch over an enum rather than `Box<dyn Hit>`
/// so the inner shapes can stay `Copy` and stack-allocated.
pub enum Object {
    Sphere(Sphere),
    Triangle(Triangle),
    Mesh(Mesh),
}

impl Object {
    pub fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
        match self {
            Self::Sphere(sphere) => sphere.hit(ray, interval),
            Self::Triangle(triangle) => triangle.hit(ray, interval),
            Self::Mesh(mesh) => mesh.hit(ray, interval),
        }
    }
}

/// World container. Lights are duplicated into `objects` so they appear in
/// camera rays, and into `lights` so the tracer can sample them directly.
pub struct Scene {
    pub objects: Vec<Object>,
    pub lights: Vec<Sphere>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            lights: Vec::new(),
        }
    }

    pub fn add_sphere(&mut self, sphere: Sphere) {
        self.objects.push(Object::Sphere(sphere));
    }

    /// Adds an emissive sphere — both as a visible object and as a sampleable
    /// light for next-event estimation.
    pub fn add_light(&mut self, sphere: Sphere) {
        self.lights.push(sphere);
        self.objects.push(Object::Sphere(sphere));
    }

    pub fn add_triangle(&mut self, triangle: Triangle) {
        self.objects.push(Object::Triangle(triangle));
    }

    pub fn add_mesh(&mut self, mesh: Mesh) {
        self.objects.push(Object::Mesh(mesh));
    }

    /// Returns the nearest hit on any object inside `interval`.
    pub fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
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
