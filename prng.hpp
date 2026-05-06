#pragma once
// Based on PCG random number generator (https://www.pcg-random.org)
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

#include "base.hpp"

// ── Types ─────────────────────────────────────────────────────────────────────
struct PrngState {
    u64 state;
    u64 inc;
};

// ── Stateful (explicit state) ─────────────────────────────────────────────────
void prng_seed_r   (PrngState& rng, u64 initstate, u64 initseq);
u32  prng_rand_r   (PrngState& rng);
f32  prng_randf_r  (PrngState& rng);

// ── Global convenience wrappers ───────────────────────────────────────────────
void prng_seed (u64 initstate, u64 initseq);
u32  prng_rand ();
f32  prng_randf();
