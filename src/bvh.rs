//! Bounding-Volume Hierarchy over a triangle list.
//!
//! Built once per mesh and queried per ray. We use a SAH (Surface Area
//! Heuristic) binned-split builder, which keeps build time linear-ish while
//! producing high-quality trees for path tracing.

use crate::geometry::{Aabb, Triangle};

/// One node in the BVH. Either an internal split (with `left`/`right` indices
/// and a `split_axis`) or a leaf containing a range of triangles.
#[derive(Clone, Copy)]
pub struct BvhNode {
    pub bounds: Aabb,
    pub left: Option<usize>,
    pub right: Option<usize>,
    /// `Some((start, end))` on leaves — these are indices into the triangle
    /// list the BVH was built over.
    pub range: Option<(usize, usize)>,
    pub split_axis: u8,
}

/// Stop subdividing once a node holds this few triangles or fewer.
pub const BVH_LEAF_SIZE: usize = 4;

/// Number of bins to use when evaluating SAH splits along an axis.
pub const BVH_BINS: usize = 16;

/// Surface area of an AABB (used as the SAH cost weight).
pub fn surface_area(b: Aabb) -> f64 {
    let d = b.max - b.min;
    let dx = d.x.max(0.0);
    let dy = d.y.max(0.0);
    let dz = d.z.max(0.0);
    2.0 * (dx * dy + dy * dz + dz * dx)
}

/// Marks node `idx` as a leaf covering `triangles[start..end]`.
fn make_leaf(nodes: &mut [BvhNode], idx: usize, start: usize, end: usize) {
    nodes[idx].range = Some((start, end));
}

/// Recursively builds the BVH for `triangles[start..end]`, appending nodes to
/// `nodes` and partitioning the slice in place. Returns the new node's index.
pub fn build_bvh(
    triangles: &mut [Triangle],
    nodes: &mut Vec<BvhNode>,
    start: usize,
    end: usize,
) -> usize {
    let node_index = nodes.len();
    let bounds = triangle_bounds(&triangles[start..end]);
    nodes.push(BvhNode {
        bounds,
        left: None,
        right: None,
        range: None,
        split_axis: 0,
    });

    let count = end - start;
    if count <= BVH_LEAF_SIZE {
        make_leaf(nodes, node_index, start, end);
        return node_index;
    }

    // Choose the longest centroid-axis as the splitting axis.
    let cb = centroid_bounds(&triangles[start..end]);
    let extent = cb.max - cb.min;
    let axis = if extent.x > extent.y && extent.x > extent.z {
        0
    } else if extent.y > extent.z {
        1
    } else {
        2
    };
    let axis_min = cb.min.axis(axis);
    let axis_max = cb.max.axis(axis);

    if axis_max - axis_min < 1e-12 {
        // All centroids collapse to a single line — no meaningful split.
        make_leaf(nodes, node_index, start, end);
        return node_index;
    }

    // Bin triangles by centroid position along the chosen axis.
    let mut bin_counts = [0usize; BVH_BINS];
    let mut bin_bounds = [Aabb::empty(); BVH_BINS];
    let scale = BVH_BINS as f64 / (axis_max - axis_min);

    for tri in &triangles[start..end] {
        let c = tri.centroid().axis(axis);
        let mut b = ((c - axis_min) * scale) as usize;
        if b >= BVH_BINS {
            b = BVH_BINS - 1;
        }
        bin_counts[b] += 1;
        bin_bounds[b] = bin_bounds[b].union(tri.bounds());
    }

    // Sweep left-to-right and right-to-left to compute cumulative SAH costs.
    let mut left_count = [0usize; BVH_BINS - 1];
    let mut left_area = [0.0f64; BVH_BINS - 1];
    let mut right_count = [0usize; BVH_BINS - 1];
    let mut right_area = [0.0f64; BVH_BINS - 1];

    let mut acc_bounds = Aabb::empty();
    let mut acc_count = 0usize;
    for i in 0..BVH_BINS - 1 {
        acc_bounds = acc_bounds.union(bin_bounds[i]);
        acc_count += bin_counts[i];
        left_count[i] = acc_count;
        left_area[i] = if acc_count == 0 { 0.0 } else { surface_area(acc_bounds) };
    }
    acc_bounds = Aabb::empty();
    acc_count = 0;
    for i in (1..BVH_BINS).rev() {
        acc_bounds = acc_bounds.union(bin_bounds[i]);
        acc_count += bin_counts[i];
        right_count[i - 1] = acc_count;
        right_area[i - 1] = if acc_count == 0 { 0.0 } else { surface_area(acc_bounds) };
    }

    // Pick the cheapest split position.
    let parent_area = surface_area(bounds).max(1e-12);
    let traversal_cost = 0.5;
    let leaf_cost = count as f64;
    let mut best_cost = f64::INFINITY;
    let mut best_split = usize::MAX;
    for i in 0..BVH_BINS - 1 {
        if left_count[i] == 0 || right_count[i] == 0 {
            continue;
        }
        let cost = traversal_cost
            + (left_count[i] as f64 * left_area[i] + right_count[i] as f64 * right_area[i])
                / parent_area;
        if cost < best_cost {
            best_cost = cost;
            best_split = i;
        }
    }

    // If splitting wouldn't help (and the node is small enough), just leaf it.
    let should_leaf = best_split == usize::MAX || (best_cost >= leaf_cost && count <= 16);
    if should_leaf {
        make_leaf(nodes, node_index, start, end);
        return node_index;
    }

    // Partition the triangle slice around the chosen split plane.
    let split_pos = axis_min + (best_split + 1) as f64 / BVH_BINS as f64 * (axis_max - axis_min);
    let mid = {
        let slice = &mut triangles[start..end];
        let mut i = 0usize;
        let mut j = slice.len();
        while i < j {
            if slice[i].centroid().axis(axis) < split_pos {
                i += 1;
            } else {
                j -= 1;
                slice.swap(i, j);
            }
        }
        start + i
    };

    // Fallback: if partition produced an empty half, fall back to a median split.
    let (left_idx, right_idx) = if mid == start || mid == end {
        triangles[start..end]
            .sort_by(|a, b| a.centroid().axis(axis).total_cmp(&b.centroid().axis(axis)));
        let mid2 = start + count / 2;
        let l = build_bvh(triangles, nodes, start, mid2);
        let r = build_bvh(triangles, nodes, mid2, end);
        (l, r)
    } else {
        let l = build_bvh(triangles, nodes, start, mid);
        let r = build_bvh(triangles, nodes, mid, end);
        (l, r)
    };

    nodes[node_index].left = Some(left_idx);
    nodes[node_index].right = Some(right_idx);
    nodes[node_index].split_axis = axis as u8;
    node_index
}

/// Union of triangle bounds — the world-space AABB of the slice.
pub fn triangle_bounds(triangles: &[Triangle]) -> Aabb {
    triangles
        .iter()
        .fold(Aabb::empty(), |bounds, triangle| bounds.union(triangle.bounds()))
}

/// AABB enclosing the centroids of all triangles in the slice. Used to pick
/// the split axis (centroid spread rather than triangle spread).
pub fn centroid_bounds(triangles: &[Triangle]) -> Aabb {
    triangles
        .iter()
        .fold(Aabb::empty(), |bounds, triangle| bounds.with_point(triangle.centroid()))
}
