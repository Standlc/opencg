//! Vector / matrix / ray math shared by the rest of the renderer.
//!
//! `Vec3` doubles as `Point3` (a position) and `Color` (an RGB triple) — they
//! all share the same storage and operators, so a single implementation
//! covers every use.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::rng::Rng;

/// 3D vector. Also used as a point and as an RGB color via the `Point3` /
/// `Color` aliases below.
#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub type Point3 = Vec3;
pub type Color = Vec3;

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Euclidean length.
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Squared length — avoid a sqrt when you only need to compare distances.
    pub fn length_squared(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Component lookup by axis index (0=x, 1=y, 2=z). Useful in BVH/AABB code
    /// that loops over axes.
    pub fn axis(self, axis: usize) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    /// Returns a unit-length vector pointing the same direction.
    pub fn unit(self) -> Self {
        self / self.length()
    }

    /// True if the vector is close enough to zero that we should treat it as
    /// degenerate (e.g. scatter directions that cancel out).
    pub fn near_zero(self) -> bool {
        const EPSILON: f64 = 1e-8;
        self.x.abs() < EPSILON && self.y.abs() < EPSILON && self.z.abs() < EPSILON
    }

    /// Mirror reflection of `self` about `normal`.
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * 2.0 * self.dot(normal)
    }

    /// Snell's-law refraction. `eta_ratio` is η_in / η_out.
    pub fn refract(self, normal: Self, eta_ratio: f64) -> Self {
        let cos_theta = (-self).dot(normal).min(1.0);
        let perpendicular = (self + normal * cos_theta) * eta_ratio;
        let parallel = normal * -(1.0 - perpendicular.length_squared()).abs().sqrt();
        perpendicular + parallel
    }

    /// Random vector with each component in `[min, max)`.
    pub fn random_range(rng: &mut Rng, min: f64, max: f64) -> Self {
        Self::new(
            rng.range_f64(min, max),
            rng.range_f64(min, max),
            rng.range_f64(min, max),
        )
    }

    /// Uniformly samples a direction on the unit sphere via rejection sampling.
    pub fn random_unit(rng: &mut Rng) -> Self {
        loop {
            let candidate = Self::random_range(rng, -1.0, 1.0);
            let length_squared = candidate.length_squared();
            if 1e-160 < length_squared && length_squared <= 1.0 {
                return candidate / length_squared.sqrt();
            }
        }
    }

    pub fn max_component(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }

    /// Random direction in the hemisphere centred on `normal`.
    pub fn random_on_hemisphere(rng: &mut Rng, normal: Self) -> Self {
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

/// A semi-open interval `(min, max)` used for ray-hit acceptance bounds.
#[derive(Clone, Copy)]
pub struct Interval {
    pub min: f64,
    pub max: f64,
}

impl Interval {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    /// Strict containment — used to reject t-values at the very edges so we
    /// don't immediately re-hit the surface we just scattered from.
    pub fn surrounds(self, value: f64) -> bool {
        self.min < value && value < self.max
    }
}

/// A ray with an origin point and a (non-normalized) direction vector.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// Point along the ray at parameter `t`.
    pub fn at(self, t: f64) -> Point3 {
        self.origin + self.direction * t
    }
}

/// 4×4 matrix stored row-major. Used for glTF node transforms.
#[derive(Clone)]
pub struct Mat4(pub [f64; 16]);

impl Mat4 {
    pub fn identity() -> Self {
        Self([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// Builds a matrix from a column-major array (the layout glTF uses).
    pub fn from_columns(cols: &[f64; 16]) -> Self {
        let mut m = [0.0f64; 16];
        for r in 0..4 {
            for c in 0..4 {
                m[r * 4 + c] = cols[c * 4 + r];
            }
        }
        Self(m)
    }

    /// Composes translation + rotation (quaternion `[x,y,z,w]`) + per-axis scale.
    pub fn from_trs(t: [f64; 3], r: [f64; 4], s: [f64; 3]) -> Self {
        let [qx, qy, qz, qw] = r;
        let r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
        let r01 = 2.0 * (qx * qy - qw * qz);
        let r02 = 2.0 * (qx * qz + qw * qy);
        let r10 = 2.0 * (qx * qy + qw * qz);
        let r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
        let r12 = 2.0 * (qy * qz - qw * qx);
        let r20 = 2.0 * (qx * qz - qw * qy);
        let r21 = 2.0 * (qy * qz + qw * qx);
        let r22 = 1.0 - 2.0 * (qx * qx + qy * qy);
        Self([
            r00 * s[0], r01 * s[1], r02 * s[2], t[0],
            r10 * s[0], r11 * s[1], r12 * s[2], t[1],
            r20 * s[0], r21 * s[1], r22 * s[2], t[2],
            0.0,        0.0,        0.0,         1.0,
        ])
    }

    /// Matrix product `self * rhs`.
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut m = [0.0f64; 16];
        for r in 0..4 {
            for c in 0..4 {
                for k in 0..4 {
                    m[r * 4 + c] += self.0[r * 4 + k] * rhs.0[k * 4 + c];
                }
            }
        }
        Self(m)
    }

    /// Transforms a position (applies translation + linear part, then perspective divide).
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let m = &self.0;
        let x = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3];
        let y = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7];
        let z = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11];
        let w = m[12] * p.x + m[13] * p.y + m[14] * p.z + m[15];
        Vec3::new(x / w, y / w, z / w)
    }

    /// Transforms a direction (ignores translation, no perspective divide).
    pub fn transform_dir(&self, v: Vec3) -> Vec3 {
        let m = &self.0;
        Vec3::new(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[4] * v.x + m[5] * v.y + m[6] * v.z,
            m[8] * v.x + m[9] * v.y + m[10] * v.z,
        )
    }
}

/// Builds an orthonormal basis (tangent, bitangent) for a surface normal.
/// Used when sampling directions in a local frame around a hit normal.
pub fn onb(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let bitangent = normal.cross(up).unit();
    let tangent = bitangent.cross(normal);
    (tangent, bitangent)
}

// --- Color / gamma helpers -----------------------------------------------

/// Cheap gamma 2.0 approximation (sqrt). Good enough for a viewer.
pub fn linear_to_gamma(value: f64) -> f64 {
    value.max(0.0).sqrt()
}

/// Clamps a 0..1 float to a 0..255 byte (display range).
pub fn to_u8(value: f64) -> u8 {
    (256.0 * value.clamp(0.0, 0.999)) as u8
}

/// Cheap sRGB→linear (gamma 2.0 approximation) for 8-bit texture decoding.
pub fn srgb_to_linear(value: u8) -> f64 {
    let v = value as f64 / 255.0;
    v * v
}

/// Packs a linear-space color into an ARGB-ish u32 after applying a scale
/// (e.g. `1 / sample_count`) and gamma correction.
pub fn color_to_u32_scaled(pixel: Color, scale: f64) -> u32 {
    let r = to_u8(linear_to_gamma(pixel.x * scale)) as u32;
    let g = to_u8(linear_to_gamma(pixel.y * scale)) as u32;
    let b = to_u8(linear_to_gamma(pixel.z * scale)) as u32;
    (r << 16) | (g << 8) | b
}

/// Schlick's approximation of Fresnel reflectance — used by dielectrics to
/// decide between reflection and refraction probabilistically.
pub fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
    let r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}
