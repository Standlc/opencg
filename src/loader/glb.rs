//! Minimal binary glTF (.glb) loader.
//!
//! We only consume the slices of the spec needed to render the test assets:
//! the embedded JSON + BIN chunks, POSITION/NORMAL/TEXCOORD_0 attributes,
//! indexed triangle primitives, baseColor + texture materials, and the
//! `KHR_materials_transmission` / `KHR_materials_ior` extensions for glass.

use std::error::Error;
use std::fs;
use std::path::Path;

use super::rotate_xyz;
use crate::geometry::Triangle;
use crate::material::{Material, Texture};
use crate::math::{Color, Mat4, Point3, Vec3};
use crate::mesh::Mesh;

/// Loads a `.glb`, walks the node tree, and returns a single built mesh in
/// world space. `center`, `scale`, `rotation` apply on top of the file's own
/// node transforms.
pub fn load_glb_mesh(
    path: &Path,
    center: Point3,
    scale: f64,
    rotation: Vec3,
) -> Result<Mesh, Box<dyn Error>> {
    let data = fs::read(path)?;

    // GLB header: 12-byte magic + version + length.
    if data.len() < 12 || &data[..4] != b"glTF" {
        return Err("not a GLB file".into());
    }

    // Chunk 0 is the JSON manifest.
    let chunk0_len = u32::from_le_bytes(data[12..16].try_into()?) as usize;
    if &data[16..20] != b"JSON" {
        return Err("expected JSON chunk first".into());
    }
    let gltf: serde_json::Value = serde_json::from_slice(&data[20..20 + chunk0_len])?;

    // Chunk 1 (optional) holds the binary buffer.
    let bin_start = 20 + chunk0_len;
    let bin: &[u8] = if bin_start + 8 <= data.len() {
        let chunk1_len = u32::from_le_bytes(data[bin_start..bin_start + 4].try_into()?) as usize;
        if &data[bin_start + 4..bin_start + 8] == b"BIN\0" {
            &data[bin_start + 8..bin_start + 8 + chunk1_len]
        } else {
            &[]
        }
    } else {
        &[]
    };

    let accessors = gltf["accessors"].as_array().ok_or("no accessors")?;
    let buffer_views = gltf["bufferViews"].as_array().ok_or("no bufferViews")?;
    let gltf_meshes = gltf["meshes"].as_array().ok_or("no meshes")?;
    let gltf_nodes = gltf["nodes"].as_array().ok_or("no nodes")?;
    let empty_arr = vec![];
    let gltf_materials = gltf["materials"].as_array().unwrap_or(&empty_arr);
    let gltf_textures = gltf["textures"].as_array().unwrap_or(&empty_arr);
    let gltf_images = gltf["images"].as_array().unwrap_or(&empty_arr);

    // Decode every image once up front; leak them so materials can hold
    // `&'static Texture` and stay `Copy`.
    let mut decoded_images: Vec<Option<&'static Texture>> = Vec::with_capacity(gltf_images.len());
    for image in gltf_images {
        let mime = image["mimeType"].as_str();
        let bytes_opt: Option<Vec<u8>> = if let Some(bv_idx) = image["bufferView"].as_u64() {
            let bv = &buffer_views[bv_idx as usize];
            let offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
            let length = bv["byteLength"].as_u64().unwrap_or(0) as usize;
            Some(bin[offset..offset + length].to_vec())
        } else if let Some(uri) = image["uri"].as_str() {
            if uri.starts_with("data:") {
                None
            } else {
                let parent = path.parent().unwrap_or_else(|| Path::new("."));
                fs::read(parent.join(uri)).ok()
            }
        } else {
            None
        };
        let tex = bytes_opt.and_then(|bytes| match Texture::from_image_bytes(&bytes, mime) {
            Ok(t) => Some(&*Box::leak(Box::new(t))),
            Err(err) => {
                eprintln!("failed to decode GLB image: {err}");
                None
            }
        });
        decoded_images.push(tex);
    }

    // Helper: resolve a texture index through the textures→source indirection.
    let texture_for_index = |idx: usize| -> Option<&'static Texture> {
        let tex = gltf_textures.get(idx)?;
        let source = tex["source"].as_u64()? as usize;
        decoded_images.get(source).copied().flatten()
    };

    // Find the default scene's root nodes.
    let scene_idx = gltf["scene"].as_u64().unwrap_or(0) as usize;
    let root_nodes: Vec<usize> = gltf["scenes"][scene_idx]["nodes"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default();

    // Iterative DFS over the node hierarchy, accumulating parent transforms.
    let mut triangles = Vec::new();
    let mut stack: Vec<(usize, Mat4)> = root_nodes
        .into_iter()
        .map(|n| (n, Mat4::identity()))
        .collect();

    while let Some((node_idx, parent_world)) = stack.pop() {
        let node = &gltf_nodes[node_idx];
        let local = glb_node_matrix(node);
        let world = parent_world.mul(&local);

        if let Some(mesh_idx) = node["mesh"].as_u64() {
            let gltf_mesh = &gltf_meshes[mesh_idx as usize];
            if let Some(primitives) = gltf_mesh["primitives"].as_array() {
                for prim in primitives {
                    // mode 4 = TRIANGLES. Skip lines/points/strips/fans.
                    if prim["mode"].as_u64().unwrap_or(4) != 4 {
                        continue;
                    }
                    let attrs = &prim["attributes"];

                    let material = prim["material"]
                        .as_u64()
                        .and_then(|mi| gltf_materials.get(mi as usize))
                        .map(|m| glb_material(m, &texture_for_index))
                        .unwrap_or_else(|| Material::lambertian(Color::new(0.75, 0.75, 0.75)));

                    let pos_idx = match attrs["POSITION"].as_u64() {
                        Some(i) => i as usize,
                        None => continue,
                    };
                    let positions = read_vec3_accessor(bin, accessors, buffer_views, pos_idx)?;

                    let normals: Option<Vec<Vec3>> = attrs["NORMAL"]
                        .as_u64()
                        .map(|ni| read_vec3_accessor(bin, accessors, buffer_views, ni as usize))
                        .transpose()?;

                    let uvs: Option<Vec<[f64; 2]>> = attrs["TEXCOORD_0"].as_u64().and_then(|ti| {
                        match read_vec2_accessor(bin, accessors, buffer_views, ti as usize) {
                            Ok(v) => Some(v),
                            Err(err) => {
                                eprintln!("failed to read TEXCOORD_0: {err}");
                                None
                            }
                        }
                    });

                    let indices: Option<Vec<usize>> = prim["indices"]
                        .as_u64()
                        .map(|ii| read_index_accessor(bin, accessors, buffer_views, ii as usize))
                        .transpose()?;

                    let tri_count = indices.as_ref().map(|i| i.len()).unwrap_or(positions.len()) / 3;
                    let get_idx = |i: usize| indices.as_ref().map(|idx| idx[i]).unwrap_or(i);

                    for t in 0..tri_count {
                        let ia = get_idx(t * 3);
                        let ib = get_idx(t * 3 + 1);
                        let ic = get_idx(t * 3 + 2);

                        // Apply node-local transform, then user scale/rotate/translate.
                        let pa = center
                            + rotate_xyz(world.transform_point(positions[ia]) * scale, rotation);
                        let pb = center
                            + rotate_xyz(world.transform_point(positions[ib]) * scale, rotation);
                        let pc = center
                            + rotate_xyz(world.transform_point(positions[ic]) * scale, rotation);

                        let triangle = if let Some(norms) = &normals {
                            let na = rotate_xyz(world.transform_dir(norms[ia]).unit(), rotation).unit();
                            let nb = rotate_xyz(world.transform_dir(norms[ib]).unit(), rotation).unit();
                            let nc = rotate_xyz(world.transform_dir(norms[ic]).unit(), rotation).unit();
                            Triangle::smooth(pa, pb, pc, na, nb, nc, material)
                        } else {
                            Triangle::flat(pa, pb, pc, material)
                        };

                        let triangle = if let Some(uv_arr) = &uvs {
                            triangle.with_uvs([
                                Some(uv_arr[ia]),
                                Some(uv_arr[ib]),
                                Some(uv_arr[ic]),
                            ])
                        } else {
                            triangle
                        };

                        triangles.push(triangle);
                    }
                }
            }
        }

        if let Some(children) = node["children"].as_array() {
            for child in children {
                if let Some(child_idx) = child.as_u64() {
                    stack.push((child_idx as usize, world.clone()));
                }
            }
        }
    }

    eprintln!("loaded {} triangles from {}", triangles.len(), path.display());
    Ok(Mesh::new(triangles))
}

/// Builds a node's local transform: either an explicit 4×4 `matrix`, or a TRS
/// triple (`translation` / `rotation` / `scale`).
fn glb_node_matrix(node: &serde_json::Value) -> Mat4 {
    if let Some(matrix) = node["matrix"].as_array() {
        let mut cols = [0.0f64; 16];
        for (i, v) in matrix.iter().enumerate().take(16) {
            cols[i] = v.as_f64().unwrap_or(if i % 5 == 0 { 1.0 } else { 0.0 });
        }
        return Mat4::from_columns(&cols);
    }

    let t = glb_f64_array3(node["translation"].as_array(), [0.0, 0.0, 0.0]);
    let r = glb_f64_array4(node["rotation"].as_array(), [0.0, 0.0, 0.0, 1.0]);
    let s = glb_f64_array3(node["scale"].as_array(), [1.0, 1.0, 1.0]);
    Mat4::from_trs(t, r, s)
}

/// Parses an optional 3-element JSON array, falling back to per-component defaults.
fn glb_f64_array3(arr: Option<&Vec<serde_json::Value>>, default: [f64; 3]) -> [f64; 3] {
    arr.map(|a| {
        [
            a.get(0).and_then(|v| v.as_f64()).unwrap_or(default[0]),
            a.get(1).and_then(|v| v.as_f64()).unwrap_or(default[1]),
            a.get(2).and_then(|v| v.as_f64()).unwrap_or(default[2]),
        ]
    })
    .unwrap_or(default)
}

/// Parses an optional 4-element JSON array (used for rotation quaternions).
fn glb_f64_array4(arr: Option<&Vec<serde_json::Value>>, default: [f64; 4]) -> [f64; 4] {
    arr.map(|a| {
        [
            a.get(0).and_then(|v| v.as_f64()).unwrap_or(default[0]),
            a.get(1).and_then(|v| v.as_f64()).unwrap_or(default[1]),
            a.get(2).and_then(|v| v.as_f64()).unwrap_or(default[2]),
            a.get(3).and_then(|v| v.as_f64()).unwrap_or(default[3]),
        ]
    })
    .unwrap_or(default)
}

/// Converts a glTF material JSON to our `Material` enum. Honours
/// `KHR_materials_transmission` / `KHR_materials_ior` for glass surfaces, and
/// turns `baseColorTexture` references into `TexturedLambertian`.
fn glb_material(
    mat: &serde_json::Value,
    texture_for_index: &dyn Fn(usize) -> Option<&'static Texture>,
) -> Material {
    let pbr = &mat["pbrMetallicRoughness"];
    let factor = pbr["baseColorFactor"].as_array().and_then(|arr| {
        Some(Color::new(
            arr.get(0)?.as_f64()?,
            arr.get(1)?.as_f64()?,
            arr.get(2)?.as_f64()?,
        ))
    });
    let alpha_factor = pbr["baseColorFactor"][3].as_f64().unwrap_or(1.0);
    let alpha_mode = mat["alphaMode"].as_str().unwrap_or("OPAQUE");
    let transmission = mat["extensions"]["KHR_materials_transmission"]["transmissionFactor"]
        .as_f64()
        .unwrap_or(0.0);

    let has_transmission = transmission > 0.0;
    let is_blend_alpha = alpha_mode == "BLEND" && alpha_factor < 0.95;
    if has_transmission || is_blend_alpha {
        // For true KHR_materials_transmission glass, use the authored IOR.
        // For BLEND-alpha surfaces (windshields, tinted panels), these are
        // typically flat translucent meshes, not real refractors — use a
        // near-1 IOR so they stay transparent with minimal Fresnel reflection.
        let ior = if has_transmission {
            mat["extensions"]["KHR_materials_ior"]["ior"]
                .as_f64()
                .unwrap_or(1.5)
        } else {
            1.1
        };
        return Material::dielectric(ior);
    }

    if let Some(tex_idx) = pbr["baseColorTexture"]["index"].as_u64() {
        if let Some(texture) = texture_for_index(tex_idx as usize) {
            let tint = factor.unwrap_or(Color::new(1.0, 1.0, 1.0));
            return Material::TexturedLambertian { texture, tint };
        }
    }

    Material::lambertian(factor.unwrap_or(Color::new(0.75, 0.75, 0.75)))
}

/// Reads a `VEC2` accessor (used for UVs). Supports float / u8 / u16 with
/// optional `normalized` flag, and flips V so our textures see bottom-left
/// origin like the OBJ path expects.
fn read_vec2_accessor(
    bin: &[u8],
    accessors: &[serde_json::Value],
    buffer_views: &[serde_json::Value],
    accessor_idx: usize,
) -> Result<Vec<[f64; 2]>, Box<dyn Error>> {
    let acc = &accessors[accessor_idx];
    let bv_idx = acc["bufferView"].as_u64().ok_or("accessor missing bufferView")? as usize;
    let bv = &buffer_views[bv_idx];
    let bv_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let acc_offset = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let count = acc["count"].as_u64().ok_or("accessor missing count")? as usize;
    let component_type = acc["componentType"]
        .as_u64()
        .ok_or("accessor missing componentType")? as u32;
    let normalized = acc["normalized"].as_bool().unwrap_or(false);

    // glTF componentType codes: 5126=float, 5121=u8, 5123=u16.
    let component_size = match component_type {
        5126 => 4,
        5121 => 1,
        5123 => 2,
        _ => return Err(format!("unsupported TEXCOORD component type {component_type}").into()),
    };
    let default_stride = component_size * 2;
    let stride = bv["byteStride"]
        .as_u64()
        .map(|s| s as usize)
        .unwrap_or(default_stride);
    let base = bv_offset + acc_offset;

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let off = base + i * stride;
        let (u, v) = match component_type {
            5126 => (
                f32::from_le_bytes(bin[off..off + 4].try_into()?) as f64,
                f32::from_le_bytes(bin[off + 4..off + 8].try_into()?) as f64,
            ),
            5121 => {
                let u = bin[off] as f64;
                let v = bin[off + 1] as f64;
                if normalized { (u / 255.0, v / 255.0) } else { (u, v) }
            }
            5123 => {
                let u = u16::from_le_bytes(bin[off..off + 2].try_into()?) as f64;
                let v = u16::from_le_bytes(bin[off + 2..off + 4].try_into()?) as f64;
                if normalized { (u / 65535.0, v / 65535.0) } else { (u, v) }
            }
            _ => unreachable!(),
        };
        // glTF V origin is top-left; Texture::sample assumes bottom-left, so flip here.
        result.push([u, 1.0 - v]);
    }
    Ok(result)
}

/// Reads a `VEC3` float accessor (used for positions and normals).
fn read_vec3_accessor(
    bin: &[u8],
    accessors: &[serde_json::Value],
    buffer_views: &[serde_json::Value],
    accessor_idx: usize,
) -> Result<Vec<Vec3>, Box<dyn Error>> {
    let acc = &accessors[accessor_idx];
    let bv_idx = acc["bufferView"].as_u64().ok_or("accessor missing bufferView")? as usize;
    let bv = &buffer_views[bv_idx];
    let bv_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let stride = bv["byteStride"].as_u64().map(|s| s as usize).unwrap_or(12);
    let acc_offset = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let count = acc["count"].as_u64().ok_or("accessor missing count")? as usize;
    let base = bv_offset + acc_offset;

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let off = base + i * stride;
        let x = f32::from_le_bytes(bin[off..off + 4].try_into()?) as f64;
        let y = f32::from_le_bytes(bin[off + 4..off + 8].try_into()?) as f64;
        let z = f32::from_le_bytes(bin[off + 8..off + 12].try_into()?) as f64;
        result.push(Vec3::new(x, y, z));
    }
    Ok(result)
}

/// Reads an index accessor (u8 / u16 / u32 variants).
fn read_index_accessor(
    bin: &[u8],
    accessors: &[serde_json::Value],
    buffer_views: &[serde_json::Value],
    accessor_idx: usize,
) -> Result<Vec<usize>, Box<dyn Error>> {
    let acc = &accessors[accessor_idx];
    let bv_idx = acc["bufferView"].as_u64().ok_or("accessor missing bufferView")? as usize;
    let bv = &buffer_views[bv_idx];
    let bv_offset = bv["byteOffset"].as_u64().unwrap_or(0) as usize;
    let acc_offset = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let count = acc["count"].as_u64().ok_or("accessor missing count")? as usize;
    let component_type = acc["componentType"]
        .as_u64()
        .ok_or("accessor missing componentType")? as u32;

    let byte_size = match component_type {
        5121 => 1,
        5123 => 2,
        5125 => 4,
        _ => return Err(format!("unsupported index component type {component_type}").into()),
    };
    let stride = bv["byteStride"]
        .as_u64()
        .map(|s| s as usize)
        .unwrap_or(byte_size);
    let base = bv_offset + acc_offset;

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let off = base + i * stride;
        let idx = match component_type {
            5121 => bin[off] as usize,
            5123 => u16::from_le_bytes(bin[off..off + 2].try_into()?) as usize,
            5125 => u32::from_le_bytes(bin[off..off + 4].try_into()?) as usize,
            _ => unreachable!(),
        };
        result.push(idx);
    }
    Ok(result)
}
