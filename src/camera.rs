//! Pinhole camera and the interactive orbit controller that drives it.

use minifb::{Key, MouseButton, MouseMode, Window};

use crate::config::{ASPECT_RATIO, ORBIT_SPEED, PAN_SPEED, ZOOM_SPEED};
use crate::math::{Point3, Ray, Vec3};

/// A simple thin-lens-free pinhole camera. Generates rays for normalized
/// viewport coordinates `(u, v)` in `[0, 1]`.
pub struct Camera {
    origin: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    /// The world-space corner of the viewport at `(u=0, v=0)`. Stored so
    /// `ray()` can be a single addition + scale per pixel.
    lower_left_corner: Point3,
}

impl Camera {
    /// Builds a camera looking from `origin` toward `look_at`, with `up` used
    /// to orient the viewport. `vertical_fov` is in degrees.
    pub fn new(
        origin: Point3,
        look_at: Point3,
        up: Vec3,
        vertical_fov: f64,
        aspect_ratio: f64,
    ) -> Self {
        let theta = vertical_fov.to_radians();
        let viewport_height = (theta / 2.0).tan() * 2.0;
        let viewport_width = aspect_ratio * viewport_height;

        // Right-handed camera basis: w points away from the target.
        let w = (origin - look_at).unit();
        let u = up.cross(w).unit();
        let v = w.cross(u);

        let horizontal = u * viewport_width;
        let vertical = v * viewport_height;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w;

        Self {
            origin,
            horizontal,
            vertical,
            lower_left_corner,
        }
    }

    /// Generates a primary ray for normalized viewport coordinates.
    pub fn ray(&self, u: f64, v: f64) -> Ray {
        Ray::new(
            self.origin,
            self.lower_left_corner + self.horizontal * u + self.vertical * v - self.origin,
        )
    }
}

/// What the current mouse drag is doing.
#[derive(Clone, Copy, PartialEq)]
pub enum DragMode {
    None,
    Orbit,
    Pan,
}

/// Mouse-only orbit camera controller.
///
/// Controls:
/// - Scroll wheel        → zoom (changes distance to focus point exponentially)
/// - Left click + drag   → orbit around the focus point (yaw/pitch)
/// - Right click + drag  → pan focus and camera together in world XY
/// - `R`                 → reset to initial position/look_at
pub struct CameraController {
    /// Remembered for the `R` reset.
    initial_position: Point3,
    initial_look_at: Point3,
    /// The world-space point being orbited around.
    focus: Point3,
    yaw: f64,
    pitch: f64,
    /// Camera-to-focus distance.
    distance: f64,
    previous_mouse: Option<(f32, f32)>,
    drag_mode: DragMode,
}

impl CameraController {
    /// Initialises the controller so the camera starts at `position` looking at
    /// `look_at`. Yaw/pitch/distance are derived from that relationship.
    pub fn new(position: Point3, look_at: Point3) -> Self {
        let offset = position - look_at;
        let distance = offset.length();
        let direction = offset / distance;
        let yaw = direction.x.atan2(direction.z);
        let pitch = direction.y.asin();

        Self {
            initial_position: position,
            initial_look_at: look_at,
            focus: look_at,
            yaw,
            pitch,
            distance,
            previous_mouse: None,
            drag_mode: DragMode::None,
        }
    }

    /// Builds the current `Camera`. Called after each `update` returns true.
    pub fn camera(&self) -> Camera {
        Camera::new(
            self.position(),
            self.focus,
            Vec3::new(0.0, 1.0, 0.0),
            20.0,
            ASPECT_RATIO,
        )
    }

    /// Current world-space camera position (focus + spherical offset).
    fn position(&self) -> Point3 {
        self.focus + self.offset()
    }

    /// Spherical offset from focus to camera, encoded by yaw/pitch/distance.
    fn offset(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        ) * self.distance
    }

    /// Reads one frame of input from `window` and applies it to the camera
    /// state. Returns `true` when anything changed (so the caller knows it
    /// needs to reset accumulation and re-render).
    pub fn update(&mut self, window: &Window) -> bool {
        let previous_focus = self.focus;
        let previous_yaw = self.yaw;
        let previous_pitch = self.pitch;
        let previous_distance = self.distance;

        if window.is_key_down(Key::R) {
            *self = Self::new(self.initial_position, self.initial_look_at);
            return true;
        }

        self.handle_scroll(window);
        self.handle_mouse(window);

        // Don't allow looking exactly straight up/down to avoid gimbal weirdness.
        let max_pitch = std::f64::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-max_pitch, max_pitch);

        (self.focus - previous_focus).length_squared() > 0.0
            || self.yaw != previous_yaw
            || self.pitch != previous_pitch
            || self.distance != previous_distance
    }

    /// Maps scroll-wheel input to an exponential zoom on `distance`.
    fn handle_scroll(&mut self, window: &Window) {
        if let Some((_, sy)) = window.get_scroll_wheel() {
            // Exponential so each tick is a constant percentage change.
            let factor = (-sy as f64 * ZOOM_SPEED).exp();
            self.distance = (self.distance * factor).clamp(0.05, 10_000.0);
        }
    }

    /// Tracks the mouse buttons (left = orbit, right = pan) and accumulates
    /// per-frame deltas while either is held. Right wins if both are pressed.
    fn handle_mouse(&mut self, window: &Window) {
        let left_down = window.get_mouse_down(MouseButton::Left);
        let right_down = window.get_mouse_down(MouseButton::Right);
        let mouse_pos = window.get_mouse_pos(MouseMode::Discard);

        let new_mode = if right_down {
            DragMode::Pan
        } else if left_down {
            DragMode::Orbit
        } else {
            DragMode::None
        };

        // Mode just changed → start a fresh drag from the current cursor
        // position so we don't jump on the first frame.
        if new_mode != self.drag_mode {
            self.drag_mode = new_mode;
            self.previous_mouse = if new_mode == DragMode::None { None } else { mouse_pos };
            return;
        }

        if self.drag_mode == DragMode::None {
            return;
        }

        // Apply the per-frame mouse delta according to the current drag mode.
        if let (Some(mouse), Some(prev)) = (mouse_pos, self.previous_mouse) {
            let dx = (mouse.0 - prev.0) as f64;
            let dy = (mouse.1 - prev.1) as f64;
            match self.drag_mode {
                DragMode::Orbit => self.apply_orbit(dx, dy),
                DragMode::Pan => self.apply_pan(dx, dy),
                DragMode::None => {}
            }
            self.previous_mouse = Some(mouse);
        } else {
            self.previous_mouse = mouse_pos;
        }
    }

    /// Orbit: dragging right rotates yaw; dragging down increases pitch (looks up).
    fn apply_orbit(&mut self, dx: f64, dy: f64) {
        self.yaw -= dx * ORBIT_SPEED;
        self.pitch += dy * ORBIT_SPEED;
    }

    /// Pan: translates `focus` along camera-right (x) and world-up (y). Scale
    /// with distance so the on-screen speed feels constant at any zoom.
    fn apply_pan(&mut self, dx: f64, dy: f64) {
        let forward = (self.focus - self.position()).unit();
        let right = forward.cross(Vec3::new(0.0, 1.0, 0.0)).unit();
        let pan_scale = self.distance * PAN_SPEED;
        self.focus += right * (-dx * pan_scale);
        self.focus += Vec3::new(0.0, 1.0, 0.0) * (dy * pan_scale);
    }
}
