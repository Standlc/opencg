# Rust Path Tracer

A small CPU path tracer that opens an interactive window and progressively renders an OBJ-backed airplane scene.

The interactive renderer computes a 1280x720 framebuffer at native 1:1 window scale and renders each progressive sample across available CPU cores.

The renderer supports spheres, flat triangle polygons, smooth-normal triangle meshes for rounded polygonal surfaces, and OBJ mesh loading with BVH acceleration.

## Run

```sh
cargo run --release
```

The default scene loads the airplane OBJ from:

```text
/Users/stan/Downloads/Airplane_v1_L1.123c4a6fedec-1680-4a36-a228-b0d440a4f280/11803_Airplane_v1_l1.obj
```

To render the airplane scene to `airplane_scene.ppm`:

```sh
cargo run --release -- --image
```

## Controls

- `W` / `S`: move forward/back
- `A` / `D`: strafe
- `Space` / `Shift`: move up/down
- Left-click + drag: look around
- Arrow keys: look around
- `R`: reset camera
- `Esc`: quit

The image accumulates one path-traced sample per frame. Moving the camera clears the accumulation so the view responds immediately.
