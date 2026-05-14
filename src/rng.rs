//! Tiny deterministic random number generator.
//!
//! Each pixel gets its own seeded `Rng` so renders are reproducible regardless
//! of how rayon schedules work across threads.

/// Mixes three 64-bit values into a well-distributed 64-bit hash. Used to seed
/// a per-pixel RNG from `(x, y, sample_index)`.
pub fn hash_u64(a: u64, b: u64, salt: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= b.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= salt.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

/// A small linear-congruential generator. Not cryptographically secure, but
/// fast and deterministic per seed — perfect for Monte-Carlo path tracing.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates an RNG from a raw seed.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Creates an RNG deterministically seeded by pixel coordinates and a
    /// sample index. Combining all three means two pixels never share a stream.
    pub fn for_pixel(x: usize, y: usize, sample: usize) -> Self {
        Self::new(hash_u64(
            x as u64,
            y as u64,
            sample as u64 ^ 0x7E57_5EED_F00D_BAAD,
        ))
    }

    /// Advances the LCG and returns the new state as the next 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    /// Returns a uniformly distributed float in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Returns a uniformly distributed float in `[min, max)`.
    pub fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }
}
