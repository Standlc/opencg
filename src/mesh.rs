//! Indexed triangle mesh accelerated by a BVH.
//!
//! The mesh owns its triangle list and its BVH. The BVH is built once at
//! construction time and never mutated afterwards.

use crate::bvh::{build_bvh, BvhNode};
use crate::geometry::{Hit, Triangle};
use crate::math::{Interval, Ray, Vec3};

/// A renderable triangle mesh.
pub struct Mesh {
    pub triangles: Vec<Triangle>,
    pub nodes: Vec<BvhNode>,
}

impl Mesh {
    /// Builds a mesh and its BVH from a list of triangles. The list is
    /// reordered in place to match BVH leaf ranges.
    pub fn new(mut triangles: Vec<Triangle>) -> Self {
        let mut nodes = Vec::new();
        if !triangles.is_empty() {
            let len = triangles.len();
            build_bvh(&mut triangles, &mut nodes, 0, len);
        }

        eprintln!("built BVH with {} nodes", nodes.len());
        Self { triangles, nodes }
    }

    /// Returns the nearest ray-triangle hit in `interval`, traversing the BVH
    /// with a small fixed-size stack. The "near child first" ordering uses the
    /// ray-direction sign on the split axis to short-circuit faster.
    pub fn hit(&self, ray: Ray, interval: Interval) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }

        // Precompute 1/dir so AABB slab tests can use a single multiply per axis.
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

            // Skip whole subtrees that can't beat the current nearest hit.
            if !node
                .bounds
                .hit(ray, inv_dir, Interval::new(interval.min, nearest))
            {
                continue;
            }

            if let Some((start, end)) = node.range {
                // Leaf: test every triangle directly.
                for tri in &self.triangles[start..end] {
                    if let Some(hit) = tri.hit(ray, Interval::new(interval.min, nearest)) {
                        nearest = hit.t;
                        result = Some(hit);
                    }
                }
            } else {
                // Internal: visit the near child first so we can tighten
                // `nearest` before testing the far child.
                let axis = node.split_axis as usize;
                let dir_neg = ray.direction.axis(axis) < 0.0;
                let (first, second) = if dir_neg {
                    (node.right, node.left)
                } else {
                    (node.left, node.right)
                };
                if let Some(s) = second {
                    stack[stack_top] = s;
                    stack_top += 1;
                }
                if let Some(f) = first {
                    stack[stack_top] = f;
                    stack_top += 1;
                }
            }
        }

        result
    }
}
