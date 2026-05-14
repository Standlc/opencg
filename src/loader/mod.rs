//! Mesh loaders (OBJ + glTF/GLB) and the common asset-space → world-space
//! `Transform` they use.

pub mod glb;
pub mod obj;

use crate::math::{Point3, Vec3};

/// Applies "place this asset into the world" math.
///
/// Asset files (especially the OBJ used here) often have a different up axis
/// and origin than our scene. `Transform` captures the four operations needed
/// to remap them:
///
///   world = center + rotate( axis_remap(asset + source_offset) * scale )
#[derive(Clone, Copy)]
pub struct Transform {
    /// Target world-space position for the asset's origin.
    pub center: Point3,
    /// Uniform scale applied after the axis remap.
    pub scale: f64,
    /// Euler rotation (radians, XYZ order) applied after scaling.
    pub rotation: Vec3,
    /// Translation applied in the asset's own coordinate system before remap.
    pub source_offset: Vec3,
}

impl Transform {
    pub fn new(center: Point3, scale: f64, rotation: Vec3, source_offset: Vec3) -> Self {
        Self {
            center,
            scale,
            rotation,
            source_offset,
        }
    }

    /// Transforms a position from asset space to world space.
    pub fn point(self, point: Point3) -> Point3 {
        self.center
            + rotate_xyz(
                remap_obj_axes(point + self.source_offset) * self.scale,
                self.rotation,
            )
    }

    /// Transforms a normal — same as `point` but without translation, and
    /// normalised at the end.
    pub fn normal(self, normal: Vec3) -> Vec3 {
        rotate_xyz(remap_obj_axes(normal), self.rotation).unit()
    }
}

/// Swaps the Y and Z components — OBJ uses Z-up, we use Y-up.
pub fn remap_obj_axes(value: Vec3) -> Vec3 {
    Vec3::new(value.x, value.z, value.y)
}

/// Applies an Euler rotation (X, then Y, then Z, all in radians) to a vector.
pub fn rotate_xyz(point: Vec3, rotation: Vec3) -> Vec3 {
    let (sin_x, cos_x) = rotation.x.sin_cos();
    let (sin_y, cos_y) = rotation.y.sin_cos();
    let (sin_z, cos_z) = rotation.z.sin_cos();

    // X-axis rotation.
    let point = Vec3::new(
        point.x,
        point.y * cos_x - point.z * sin_x,
        point.y * sin_x + point.z * cos_x,
    );
    // Y-axis rotation.
    let point = Vec3::new(
        point.x * cos_y + point.z * sin_y,
        point.y,
        -point.x * sin_y + point.z * cos_y,
    );
    // Z-axis rotation.
    Vec3::new(
        point.x * cos_z - point.y * sin_z,
        point.x * sin_z + point.y * cos_z,
        point.z,
    )
}
