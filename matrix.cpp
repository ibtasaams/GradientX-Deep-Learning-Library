#include "matrix.hpp"
#include "prng.hpp"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────────────────────
//  Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

Matrix* mat_create(MemArena* arena, u32 rows, u32 cols) {
    Matrix* mat   = push_struct<Matrix>(arena, /*zero=*/true);
    mat->rows     = rows;
    mat->cols     = cols;
    mat->data     = push_array<f32>(arena, static_cast<u64>(rows) * cols, /*zero=*/true);
    return mat;
}

Matrix* mat_load(MemArena* arena, u32 rows, u32 cols, const char* filename) {
    Matrix* mat = mat_create(arena, rows, cols);

    std::FILE* f = std::fopen(filename, "rb");
    if (!f) {
        std::printf("\n[!] Error: Could not open %s (make sure you ran getmnist.py and are in the correct directory)\n", filename);
        std::exit(1);
    }

    std::fseek(f, 0, SEEK_END);
    u64 file_size = static_cast<u64>(std::ftell(f));
    std::fseek(f, 0, SEEK_SET);

    u64 byte_size = dl_min(file_size, sizeof(f32) * static_cast<u64>(rows) * cols);
    std::fread(mat->data, 1, byte_size, f);
    std::fclose(f);

    return mat;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bulk fill / copy
// ─────────────────────────────────────────────────────────────────────────────

b32 mat_copy(Matrix* dst, const Matrix* src) {
    if (dst->rows != src->rows || dst->cols != src->cols) return 0;
    std::memcpy(dst->data, src->data, sizeof(f32) * dst->size());
    return 1;
}

void mat_clear(Matrix* mat) {
    std::memset(mat->data, 0, sizeof(f32) * mat->size());
}

void mat_fill(Matrix* mat, f32 x) {
    for (u64 i = 0; i < mat->size(); ++i) mat->data[i] = x;
}

void mat_fill_rand(Matrix* mat, f32 lower, f32 upper) {
    for (u64 i = 0; i < mat->size(); ++i)
        mat->data[i] = prng_randf() * (upper - lower) + lower;
}

void mat_scale(Matrix* mat, f32 scale) {
    for (u64 i = 0; i < mat->size(); ++i) mat->data[i] *= scale;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Reductions
// ─────────────────────────────────────────────────────────────────────────────

f32 mat_sum(const Matrix* mat) {
    f32 s = 0.0f;
    for (u64 i = 0; i < mat->size(); ++i) s += mat->data[i];
    return s;
}

u64 mat_argmax(const Matrix* mat) {
    u64 max_i = 0;
    for (u64 i = 1; i < mat->size(); ++i)
        if (mat->data[i] > mat->data[max_i]) max_i = i;
    return max_i;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Element-wise
// ─────────────────────────────────────────────────────────────────────────────

b32 mat_add(Matrix* out, const Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols)         return 0;
    if (out->rows != a->rows || out->cols != a->cols)     return 0;
    for (u64 i = 0; i < out->size(); ++i)
        out->data[i] = a->data[i] + b->data[i];
    return 1;
}

b32 mat_sub(Matrix* out, const Matrix* a, const Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols)         return 0;
    if (out->rows != a->rows || out->cols != a->cols)     return 0;
    for (u64 i = 0; i < out->size(); ++i)
        out->data[i] = a->data[i] - b->data[i];
    return 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Matrix multiply helpers  (NN / NT / TN / TT)
// ─────────────────────────────────────────────────────────────────────────────

static void _mat_mul_nn(Matrix* out, const Matrix* a, const Matrix* b) {
    for (u64 i = 0; i < out->rows; ++i)
        for (u64 k = 0; k < a->cols; ++k)
            for (u64 j = 0; j < out->cols; ++j)
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] *
                    b->data[j + k * b->cols];
}

static void _mat_mul_nt(Matrix* out, const Matrix* a, const Matrix* b) {
    for (u64 i = 0; i < out->rows; ++i)
        for (u64 j = 0; j < out->cols; ++j)
            for (u64 k = 0; k < a->cols; ++k)
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] *
                    b->data[k + j * b->cols];
}

static void _mat_mul_tn(Matrix* out, const Matrix* a, const Matrix* b) {
    for (u64 k = 0; k < a->rows; ++k)
        for (u64 i = 0; i < out->rows; ++i)
            for (u64 j = 0; j < out->cols; ++j)
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] *
                    b->data[j + k * b->cols];
}

static void _mat_mul_tt(Matrix* out, const Matrix* a, const Matrix* b) {
    for (u64 i = 0; i < out->rows; ++i)
        for (u64 j = 0; j < out->cols; ++j)
            for (u64 k = 0; k < a->rows; ++k)
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] *
                    b->data[k + j * b->cols];
}

b32 mat_mul(
    Matrix* out, const Matrix* a, const Matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
) {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows)                            return 0;
    if (out->rows != a_rows || out->cols != b_cols)  return 0;

    if (zero_out) mat_clear(out);

    u32 t = (static_cast<u32>(transpose_a) << 1) | static_cast<u32>(transpose_b);
    switch (t) {
        case 0b00: _mat_mul_nn(out, a, b); break;
        case 0b01: _mat_mul_nt(out, a, b); break;
        case 0b10: _mat_mul_tn(out, a, b); break;
        case 0b11: _mat_mul_tt(out, a, b); break;
    }

    return 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Activations / loss — forward
// ─────────────────────────────────────────────────────────────────────────────

b32 mat_relu(Matrix* out, const Matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) return 0;
    for (u64 i = 0; i < out->size(); ++i)
        out->data[i] = dl_max(0.0f, in->data[i]);
    return 1;
}

b32 mat_softmax(Matrix* out, const Matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) return 0;
    f32 sum = 0.0f;
    for (u64 i = 0; i < out->size(); ++i) {
        out->data[i] = expf(in->data[i]);
        sum += out->data[i];
    }
    mat_scale(out, 1.0f / sum);
    return 1;
}

b32 mat_cross_entropy(Matrix* out, const Matrix* p, const Matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols)         return 0;
    if (out->rows != p->rows || out->cols != p->cols)     return 0;
    for (u64 i = 0; i < out->size(); ++i)
        out->data[i] = (p->data[i] == 0.0f)
            ? 0.0f
            : p->data[i] * -logf(q->data[i]);
    return 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Activations / loss — backward  (accumulate)
// ─────────────────────────────────────────────────────────────────────────────

b32 mat_relu_add_grad(Matrix* out, const Matrix* in, const Matrix* grad) {
    if (out->rows != in->rows   || out->cols != in->cols)   return 0;
    if (out->rows != grad->rows || out->cols != grad->cols) return 0;
    for (u64 i = 0; i < out->size(); ++i)
        out->data[i] += (in->data[i] > 0.0f) ? grad->data[i] : 0.0f;
    return 1;
}

b32 mat_softmax_add_grad(Matrix* out, const Matrix* softmax_out, const Matrix* grad) {
    // requires a vector (either rows==1 or cols==1)
    if (softmax_out->rows != 1 && softmax_out->cols != 1) return 0;

    MemArenaTemp scratch = arena_scratch_get(nullptr, 0);

    u32     sz       = dl_max(softmax_out->rows, softmax_out->cols);
    Matrix* jacobian = mat_create(scratch.arena, sz, sz);

    for (u32 i = 0; i < sz; ++i)
        for (u32 j = 0; j < sz; ++j)
            jacobian->at(i, j) =
                softmax_out->data[i] * (static_cast<f32>(i == j) - softmax_out->data[j]);

    mat_mul(out, jacobian, grad, 0, 0, 0);

    arena_scratch_release(scratch);
    return 1;
}

b32 mat_cross_entropy_add_grad(
    Matrix* p_grad, Matrix* q_grad,
    const Matrix* p, const Matrix* q, const Matrix* grad
) {
    if (p->rows != q->rows || p->cols != q->cols) return 0;

    if (p_grad != nullptr) {
        if (p_grad->rows != p->rows || p_grad->cols != p->cols) return 0;
        for (u64 i = 0; i < p->size(); ++i)
            p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
    }

    if (q_grad != nullptr) {
        if (q_grad->rows != q->rows || q_grad->cols != q->cols) return 0;
        for (u64 i = 0; i < q->size(); ++i)
            q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
    }

    return 1;
}
