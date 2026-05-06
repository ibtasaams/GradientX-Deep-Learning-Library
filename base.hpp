#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <climits>

// ── Primitive type aliases ────────────────────────────────────────────────────
using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using b8  = i8;
using b32 = i32;

using f32 = float;

// ── Size helpers ──────────────────────────────────────────────────────────────
constexpr u64 KiB(u64 n) { return n << 10; }
constexpr u64 MiB(u64 n) { return n << 20; }
constexpr u64 GiB(u64 n) { return n << 30; }

// ── Generic utilities ─────────────────────────────────────────────────────────
template<typename T>
constexpr T dl_min(T a, T b) { return a < b ? a : b; }

template<typename T>
constexpr T dl_max(T a, T b) { return a > b ? a : b; }

constexpr u64 align_up_pow2(u64 n, u64 p) {
    return (n + (p - 1)) & ~(p - 1);
}
