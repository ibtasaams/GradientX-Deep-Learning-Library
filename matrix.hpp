#pragma once
#include "base.hpp"
#include "arena.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Matrix  (row-major, arena-allocated)
// ─────────────────────────────────────────────────────────────────────────────

struct Matrix {
    u32  rows, cols;
    f32* data;           // row-major flat array

    // ── Convenience accessors ────────────────────────────────────────────────
    f32& at(u32 r, u32 c)             { return data[c + r * cols]; }
    f32  at(u32 r, u32 c) const       { return data[c + r * cols]; }
    u64  size()            const      { return static_cast<u64>(rows) * cols; }
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────
Matrix* mat_create(MemArena* arena, u32 rows, u32 cols);
Matrix* mat_load  (MemArena* arena, u32 rows, u32 cols, const char* filename);

// ── Bulk fill / copy ──────────────────────────────────────────────────────────
b32  mat_copy      (Matrix* dst, const Matrix* src);
void mat_clear     (Matrix* mat);
void mat_fill      (Matrix* mat, f32 x);
void mat_fill_rand (Matrix* mat, f32 lower, f32 upper);
void mat_scale     (Matrix* mat, f32 scale);

// ── Reductions ────────────────────────────────────────────────────────────────
f32  mat_sum    (const Matrix* mat);
u64  mat_argmax (const Matrix* mat);

// ── Element-wise ops ──────────────────────────────────────────────────────────
b32 mat_add (Matrix* out, const Matrix* a, const Matrix* b);
b32 mat_sub (Matrix* out, const Matrix* a, const Matrix* b);

// ── Matrix multiply  (zero_out clears out first; transpose_a/b swap dims) ─────
b32 mat_mul(
    Matrix* out, const Matrix* a, const Matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);

// ── Activation / loss forward ─────────────────────────────────────────────────
b32 mat_relu          (Matrix* out, const Matrix* in);
b32 mat_softmax       (Matrix* out, const Matrix* in);
b32 mat_cross_entropy (Matrix* out, const Matrix* p, const Matrix* q);

// ── Activation / loss backward (accumulate into existing grad) ────────────────
b32 mat_relu_add_grad(
    Matrix* out, const Matrix* in, const Matrix* grad
);
b32 mat_softmax_add_grad(
    Matrix* out, const Matrix* softmax_out, const Matrix* grad
);
b32 mat_cross_entropy_add_grad(
    Matrix* p_grad, Matrix* q_grad,
    const Matrix* p, const Matrix* q, const Matrix* grad
);
