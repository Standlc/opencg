# Rust Path Tracer

A small CPU path tracer that opens an interactive window and progressively renders a 3D scene.

The interactive renderer computes a 1280x720 framebuffer at native 1:1 window scale and traces each progressive sample across available CPU cores using `rayon`. Geometry uses a SAH-binned BVH for acceleration; lighting uses path tracing with next-event estimation and Russian-roulette termination.

The renderer supports:

- Analytic spheres (also used as area lights)
- Flat and smooth-normal triangle meshes
- OBJ + MTL loading (with `map_Kd` textures)
- Binary glTF (`.glb`) loading (with `baseColorTexture` and basic `KHR_materials_transmission` / `KHR_materials_ior`)
- Lambertian, textured-Lambertian, dielectric (glass), and emissive materials

## Run

The default scene loads the McLaren P1 GLB (`models/mclaren_p1.glb`):

```sh
cargo run --release
```

To load the airplane scene (`models/airplane.obj`) instead:

```sh
cargo run --release -- --airplane
```

To render the airplane scene offline to `models/airplane_scene.ppm`:

```sh
cargo run --release -- --image
```

## Assets

Model files live under `models/`:

```
models/airplane.obj      # OBJ + MTL airplane (used by --airplane and --image)
models/audi_r8.glb       # GLB Audi R8 (referenced by AUDI_GLB_PATH in src/config.rs)
models/mclaren_p1.glb    # GLB McLaren P1 (default scene)
```

Change the constants at the top of `src/config.rs` to point at different files.

## Controls

Mouse / trackpad only:

- **Scroll wheel** — zoom (exponential, scaled by current distance to focus)
- **Left click + drag** — orbit around the focus point
- **Right click + drag** — pan the focus point (and the camera with it) in the vertical plane and along the camera's x axis
- **`R`** — reset camera to its initial position
- **`Esc`** — quit

The image accumulates one path-traced sample per frame. Any camera movement clears the accumulation so the view responds immediately, then progressively refines while the camera is still.

## Project layout

```
src/
├── main.rs        — CLI dispatch, viewer loop, --image mode
├── config.rs      — image size, tuning, asset paths
├── math.rs        — Vec3 / Mat4 / Ray / Interval / color helpers
├── rng.rs         — deterministic per-pixel RNG
├── material.rs    — Texture (JPEG/PNG) + Material enum + scatter()
├── geometry.rs    — Hit, Sphere, Triangle, Aabb
├── bvh.rs         — SAH-binned BVH builder
├── mesh.rs        — triangle mesh with BVH traversal
├── scene.rs       — Object enum + Scene container
├── camera.rs      — Camera + orbit CameraController
├── render.rs      — trace(), sample_lights(), render loop, write_ppm
├── scenes.rs      — airplane_scene, car_scene + procedural prim helpers
└── loader/
    ├── mod.rs     — shared Transform + axis-remap helpers
    ├── obj.rs     — OBJ + MTL parser
    └── glb.rs     — binary glTF parser
```
