//! Primitive shapes and their ray-intersection logic.

use crate::material::Material;
use crate::math::{Interval, Point3, Ray, Vec3};

/// Records a single ray-surface intersection.
pub struct Hit {
    pub point: Point3,
    pub normal: Vec3,
    pub t: f64,
    /// True when the ray hit the outside of the surface (as opposed to exiting
    /// from inside a closed mesh). Needed for correct dielectric behaviour.
    pub front_face: bool,
    pub material: Material,
    /// Interpolated UV at the hit, if the primitive carried texture coords.
    pub uv: Option<[f64; 2]>,
}

impl Hit {
    /// Builds a hit and flips the normal to point against the incoming ray.
    pub fn new(ray: Ray, outward_normal: Vec3, t: f64, material: Material) -> Self {
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

    pub fn with_uv(mut self, uv: Option<[f64; 2]>) -> Self {
        self.uv = uv;
        self
    }
}

/// Analytic sphere primitive (also used as area lights).
#[derive(Clone, Copy)]
pub struct Sphere {
    pub center: Point3,
    pub radius: f64,
    pub material: Material,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }

    /// Standard ray-sphere quadratic intersection — returns the nearest root
    /// inside `interval`, if any.
    pub fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
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

/// Triangle with per-vertex normals and optional per-vertex UVs.
#[derive(Clone, Copy)]
pub struct Triangle {
    pub vertices: [Point3; 3],
    pub normals: [Vec3; 3],
    pub uvs: [Option<[f64; 2]>; 3],
    pub material: Material,
}

impl Triangle {
    /// Flat-shaded triangle: a single face normal is used at all three corners.
    pub fn flat(a: Point3, b: Point3, c: Point3, material: Material) -> Self {
        let normal = (b - a).cross(c - a).unit();
        Self::smooth(a, b, c, normal, normal, normal, material)
    }

    /// Smooth-shaded triangle with three explicit vertex normals.
    pub fn smooth(
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

    /// Attaches per-vertex UVs (used for textured materials).
    pub fn with_uvs(mut self, uvs: [Option<[f64; 2]>; 3]) -> Self {
        self.uvs = uvs;
        self
    }

    /// Möller–Trumbore ray-triangle intersection. Computes barycentric (u, v)
    /// and uses them to interpolate normal and UV at the hit point.
    pub fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
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

    /// Tight axis-aligned bounding box around the triangle's three vertices.
    pub fn bounds(&self) -> Aabb {
        Aabb::from_points(self.vertices[0], self.vertices[1]).with_point(self.vertices[2])
    }

    /// Geometric centroid — used as the splitting key during BVH construction.
    pub fn centroid(&self) -> Point3 {
        (self.vertices[0] + self.vertices[1] + self.vertices[2]) / 3.0
    }
}

/// Axis-aligned bounding box. Padded slightly to avoid zero-thickness boxes.
#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Point3,
    pub max: Point3,
}

impl Aabb {
    /// An "inverted" empty box that becomes a valid box after any `union` /
    /// `with_point` call.
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    /// Smallest box enclosing two points (padded so it has nonzero volume).
    pub fn from_points(a: Point3, b: Point3) -> Self {
        Self {
            min: Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)),
            max: Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)),
        }
        .pad()
    }

    /// Returns a box expanded to also enclose `point`.
    pub fn with_point(self, point: Point3) -> Self {
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

    /// Smallest box enclosing both `self` and `rhs`.
    pub fn union(self, rhs: Self) -> Self {
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

    /// Index of the longest axis (0/1/2). Used by BVH heuristics.
    #[allow(dead_code)]
    pub fn longest_axis(self) -> usize {
        let span = self.max - self.min;
        if span.x > span.y && span.x > span.z {
            0
        } else if span.y > span.z {
            1
        } else {
            2
        }
    }

    /// Slab-test ray-AABB intersection. `inv_dir` is `1/ray.direction`
    /// precomputed by the caller (we reuse it across many AABBs).
    pub fn hit(self, ray: Ray, inv_dir: Vec3, mut interval: Interval) -> bool {
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

    /// Adds a tiny epsilon along any flat axis so the box has nonzero thickness
    /// everywhere (prevents degenerate slab tests).
    pub fn pad(self) -> Self {
        let delta = 0.0001;
        let x = if self.max.x - self.min.x < delta { delta } else { 0.0 };
        let y = if self.max.y - self.min.y < delta { delta } else { 0.0 };
        let z = if self.max.z - self.min.z < delta { delta } else { 0.0 };
        Self {
            min: self.min - Vec3::new(x, y, z),
            max: self.max + Vec3::new(x, y, z),
        }
    }
}
