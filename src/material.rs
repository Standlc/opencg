//! Surface materials and the textures they may sample.
//!
//! `Material` is an enum rather than a trait object so it can be `Copy` and
//! live inline in every `Hit` / `Triangle` / `Sphere` without heap traffic.

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::math::{reflectance, srgb_to_linear, Color, Ray, Vec3};
use crate::rng::Rng;

// `Hit` is defined alongside geometry but is used here too. We `use` it from
// the geometry module to keep material scatter logic close to the data it
// reads from a hit record.
use crate::geometry::Hit;

/// A linear-space image used as a texture map. Sampled with wrapping UVs.
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<Color>,
}

impl Texture {
    /// Loads a JPEG texture from disk (used by the OBJ/MTL pipeline).
    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        Self::from_jpeg_reader(BufReader::new(file))
    }

    /// Decodes a JPEG stream into a linear-space `Texture`.
    pub fn from_jpeg_reader<R: std::io::Read>(reader: R) -> Result<Self, Box<dyn Error>> {
        let mut decoder = jpeg_decoder::Decoder::new(reader);
        let pixels = decoder.decode()?;
        let info = decoder.info().ok_or("missing JPEG info")?;
        let width = info.width as u32;
        let height = info.height as u32;

        let linear = match info.pixel_format {
            jpeg_decoder::PixelFormat::RGB24 => pixels
                .chunks_exact(3)
                .map(|p| {
                    Color::new(
                        srgb_to_linear(p[0]),
                        srgb_to_linear(p[1]),
                        srgb_to_linear(p[2]),
                    )
                })
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

    /// Decodes a PNG byte slice (RGB/RGBA/grayscale variants) into linear space.
    pub fn from_png_bytes(bytes: &[u8]) -> Result<Self, Box<dyn Error>> {
        let mut decoder = png::Decoder::new(bytes);
        decoder.set_transformations(
            png::Transformations::EXPAND | png::Transformations::STRIP_16,
        );
        let mut reader = decoder.read_info()?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf)?;
        let width = info.width;
        let height = info.height;
        let bytes = &buf[..info.buffer_size()];

        let linear: Vec<Color> = match info.color_type {
            png::ColorType::Rgb => bytes
                .chunks_exact(3)
                .map(|p| {
                    Color::new(
                        srgb_to_linear(p[0]),
                        srgb_to_linear(p[1]),
                        srgb_to_linear(p[2]),
                    )
                })
                .collect(),
            png::ColorType::Rgba => bytes
                .chunks_exact(4)
                .map(|p| {
                    Color::new(
                        srgb_to_linear(p[0]),
                        srgb_to_linear(p[1]),
                        srgb_to_linear(p[2]),
                    )
                })
                .collect(),
            png::ColorType::Grayscale => bytes
                .iter()
                .map(|&p| {
                    let v = srgb_to_linear(p);
                    Color::new(v, v, v)
                })
                .collect(),
            png::ColorType::GrayscaleAlpha => bytes
                .chunks_exact(2)
                .map(|p| {
                    let v = srgb_to_linear(p[0]);
                    Color::new(v, v, v)
                })
                .collect(),
            other => return Err(format!("unsupported PNG color type: {other:?}").into()),
        };

        Ok(Self {
            width,
            height,
            pixels: linear,
        })
    }

    /// Auto-detects PNG vs JPEG from the MIME hint or magic bytes and decodes.
    pub fn from_image_bytes(bytes: &[u8], mime_type: Option<&str>) -> Result<Self, Box<dyn Error>> {
        let is_png = match mime_type {
            Some("image/png") => true,
            Some("image/jpeg") => false,
            _ => bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n",
        };
        if is_png {
            Self::from_png_bytes(bytes)
        } else {
            Self::from_jpeg_reader(bytes)
        }
    }

    /// Nearest-neighbour sample with wrap-around UVs. UVs are assumed to be in
    /// the convention where V=0 is at the top of the image.
    pub fn sample(&self, uv: [f64; 2]) -> Color {
        let u = uv[0] - uv[0].floor();
        let v = uv[1] - uv[1].floor();
        let x = ((u * self.width as f64) as i64).clamp(0, self.width as i64 - 1) as u32;
        let y_flipped = 1.0 - v;
        let y = ((y_flipped * self.height as f64) as i64).clamp(0, self.height as i64 - 1) as u32;
        self.pixels[(y * self.width + x) as usize]
    }
}

/// Surface scattering model. Textures are referenced with a `'static` lifetime
/// because we leak them at load time (the renderer holds them for the whole
/// program lifetime, and this keeps `Material: Copy`).
#[derive(Clone, Copy)]
pub enum Material {
    /// Solid-color diffuse.
    Lambertian { albedo: Color },
    /// Diffuse with a base-color texture, optionally tinted.
    TexturedLambertian {
        texture: &'static Texture,
        tint: Color,
    },
    /// Mirror-like surface with optional roughness (`fuzz`).
    #[allow(dead_code)]
    Metal { albedo: Color, fuzz: f64 },
    /// Refractive glass-like surface (IOR-based).
    Dielectric { refraction_index: f64 },
    /// Light source (no scattering, only emission).
    Emissive { emission: Color },
}

impl Material {
    pub fn lambertian(albedo: Color) -> Self {
        Self::Lambertian { albedo }
    }

    #[allow(dead_code)]
    pub fn metal(albedo: Color, fuzz: f64) -> Self {
        Self::Metal {
            albedo,
            fuzz: fuzz.min(1.0),
        }
    }

    pub fn dielectric(refraction_index: f64) -> Self {
        Self::Dielectric { refraction_index }
    }

    pub fn emissive(emission: Color) -> Self {
        Self::Emissive { emission }
    }

    /// Returns the emitted radiance, or zero for non-emissive materials.
    pub fn emission(self) -> Color {
        match self {
            Self::Emissive { emission } => emission,
            _ => Color::ZERO,
        }
    }

    /// Specular surfaces don't use next-event-estimation, so the tracer needs
    /// to know which materials count as specular.
    pub fn is_specular(self) -> bool {
        matches!(self, Self::Metal { .. } | Self::Dielectric { .. })
    }

    /// Given an incoming ray and a hit point, returns the outgoing scattered
    /// ray and the throughput attenuation, or `None` if the surface absorbs
    /// (or emits without scattering).
    pub fn scatter(self, ray: Ray, hit: &Hit, rng: &mut Rng) -> Option<(Ray, Color)> {
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
                let albedo = hit.uv.map(|uv| texture.sample(uv) * tint).unwrap_or(tint);
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
                // Total internal reflection + Fresnel-weighted reflect/refract.
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
