//! Wavefront OBJ + companion MTL loader.
//!
//! Supports the subset the test assets need: vertex / vertex-normal /
//! texture-coordinate / face / `usemtl` directives in OBJ, and `newmtl` /
//! `Kd` / `map_Kd` in MTL.

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use super::Transform;
use crate::geometry::Triangle;
use crate::material::{Material, Texture};
use crate::math::{Color, Point3, Vec3};
use crate::mesh::Mesh;

/// Loads `path` (OBJ), resolves any companion MTL, and returns a built mesh in
/// world space.
pub fn load_obj_mesh(path: &Path, transform: Transform) -> Result<Mesh, Box<dyn Error>> {
    // print!("loading OBJ {}... ", path.display());
    let obj = fs::read_to_string(path)?;
    let materials = load_mtl_materials(path)?;
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut texcoords: Vec<[f64; 2]> = Vec::new();
    let mut triangles = Vec::new();
    // Default material applied to faces that appear before any `usemtl` line.
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
                // Switch the active material for following faces. Fall back to
                // a name-heuristic colour if the MTL didn't define this name.
                if let Some(name) = parts.next() {
                    current_material = materials
                        .get(name)
                        .copied()
                        .unwrap_or_else(|| material_for_name(name));
                }
            }
            Some("f") => {
                // Faces may be n-gons; we fan-triangulate them.
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

                    // Use the face's geometric normal as a fallback when a
                    // corner doesn't reference an OBJ normal.
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

/// One corner of an OBJ face: vertex index, optional texcoord index, optional normal index.
#[derive(Clone, Copy)]
struct ObjCorner {
    vertex: usize,
    texture: Option<usize>,
    normal: Option<usize>,
}

/// Parses an OBJ face corner of the form `v`, `v/vt`, `v//vn`, or `v/vt/vn`.
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

/// Converts a 1-based OBJ index (which may be negative for "from the end")
/// into a 0-based slice index.
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

/// Parses a float from an `Option<&str>`, returning an error if missing or malformed.
fn parse_f64(value: Option<&str>) -> Result<f64, Box<dyn Error>> {
    Ok(value.ok_or("missing float")?.parse()?)
}

/// Walks the OBJ for `mtllib` directives and loads each referenced MTL file.
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

/// Reads one MTL file, populating `materials` with `Lambertian` /
/// `TexturedLambertian` entries keyed by material name.
fn read_mtl(
    path: PathBuf,
    materials: &mut HashMap<String, Material>,
) -> Result<(), Box<dyn Error>> {
    let mtl = fs::read_to_string(&path)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut current: Option<String> = None;
    // Track the Kd tint per material so a later `map_Kd` line can combine them.
    let mut tint: HashMap<String, Color> = HashMap::new();

    for line in mtl.lines() {
        let line = line.trim();
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("newmtl") => {
                // Start a new material; seed it with a name-based colour
                // until we hit a `Kd` or `map_Kd` directive.
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
                    // Some MTLs use pure-white Kd as a stand-in for "no tint";
                    // clamp those back to a deliberate white.
                    let color = if r > 0.95 && g > 0.95 && b > 0.95 {
                        Color::new(1.0, 1.0, 1.0)
                    } else {
                        Color::new(r, g, b)
                    };
                    tint.insert(name.clone(), color);
                    // Don't overwrite a textured material with a flat colour.
                    if !matches!(
                        materials.get(name),
                        Some(Material::TexturedLambertian { .. })
                    ) {
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
                            // Leak the texture so its address lives forever; the
                            // `Material` only needs to be `Copy`.
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
                            eprintln!("failed to load texture {}: {err}", texture_path.display());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

/// Returns a Lambertian material with a name-heuristic colour. Used as a
/// fallback when the MTL doesn't define a material we reference.
fn material_for_name(name: &str) -> Material {
    Material::lambertian(material_color_for_name(name))
}

/// Picks a plausible base colour by keyword-matching the material name.
fn material_color_for_name(name: &str) -> Color {
    if name.contains("body") {
        Color::new(0.86, 0.84, 0.78)
    } else if name.contains("tail") {
        Color::new(0.75, 0.16, 0.12)
    } else if name.contains("wing") {
        Color::new(0.66, 0.7, 0.74)
    } else {
        Color::new(0.75, 0.75, 0.72)
    }
}
