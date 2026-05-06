#include "prng.hpp"
#include <climits>

// ── Global state (same fixed seed as the C version) ───────────────────────────
static PrngState s_prng_state = {
    0x853c49e6748fea9bULL,
    0xda3e39cb94b95bdbULL
};

// ── Core PCG step ─────────────────────────────────────────────────────────────
void prng_seed_r(PrngState& rng, u64 initstate, u64 initseq) {
    rng.state = 0ULL;
    rng.inc   = (initseq << 1u) | 1u;
    prng_rand_r(rng);
    rng.state += initstate;
    prng_rand_r(rng);
}

u32 prng_rand_r(PrngState& rng) {
    u64 oldstate  = rng.state;
    rng.state     = oldstate * 6364136223846793005ULL + rng.inc;
    u32 xorshifted = static_cast<u32>(((oldstate >> 18u) ^ oldstate) >> 27u);
    u32 rot        = static_cast<u32>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

f32 prng_randf_r(PrngState& rng) {
    return static_cast<f32>(prng_rand_r(rng)) / static_cast<f32>(UINT32_MAX);
}

// ── Global wrappers ───────────────────────────────────────────────────────────
void prng_seed(u64 initstate, u64 initseq) {
    prng_seed_r(s_prng_state, initstate, initseq);
}

u32 prng_rand() {
    return prng_rand_r(s_prng_state);
}

f32 prng_randf() {
    return prng_randf_r(s_prng_state);
}
