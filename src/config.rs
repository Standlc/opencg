//! Global tuning constants shared across modules.
//!
//! Centralising these keeps render/viewer/camera settings in one place rather
//! than scattered across files.

/// Image aspect ratio (width / height). Used by both viewer and offline render.
pub const ASPECT_RATIO: f64 = 16.0 / 9.0;

/// Output image width in pixels.
pub const IMAGE_WIDTH: usize = 1280;

/// Output image height in pixels, derived from width and aspect ratio.
pub const IMAGE_HEIGHT: usize = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as usize;

/// Maximum number of ray bounces the path tracer is allowed to follow.
pub const MAX_DEPTH: usize = 16;

/// Bounce depth at which Russian-roulette termination starts kicking in.
pub const RR_MIN_DEPTH: usize = 3;

/// If false, the tracer only does direct light sampling (no indirect bounces).
/// Mostly useful as a debugging toggle.
pub const PATH_TRACING: bool = true;

// --- Viewer camera controls ----------------------------------------------

/// Yaw/pitch radians per pixel of mouse drag while orbiting.
pub const ORBIT_SPEED: f64 = 0.005;

/// World-space pan distance per pixel of mouse drag (scaled by camera distance).
pub const PAN_SPEED: f64 = 0.001;

/// Multiplier on scroll-wheel delta when zooming (exponential).
pub const ZOOM_SPEED: f64 = 0.05;

// --- Offline image rendering ---------------------------------------------

/// Number of accumulated samples per pixel when rendering to an image file.
pub const IMAGE_SAMPLES: usize = 12;

/// Output filename for `--image` mode.
pub const IMAGE_OUTPUT_FILE: &str = "models/airplane_scene.ppm";

/// Path to the airplane OBJ asset (loaded when `--airplane` is passed).
pub const AIRPLANE_OBJ_PATH: &str = "models/airplane/airplane.obj";

/// Path to the default GLB asset (used when no flag is given).
pub const AUDI_GLB_PATH: &str = "models/audi_r8.glb";
