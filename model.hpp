#pragma once
#include "base.hpp"
#include "arena.hpp"
#include "matrix.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Flags  (bit-field, stays as plain enum so we can OR bits freely)
// ─────────────────────────────────────────────────────────────────────────────
enum ModelVarFlags : u32 {
    MV_FLAG_NONE           = 0,
    MV_FLAG_REQUIRES_GRAD  = (1u << 0),
    MV_FLAG_PARAMETER      = (1u << 1),
    MV_FLAG_INPUT          = (1u << 2),
    MV_FLAG_OUTPUT         = (1u << 3),
    MV_FLAG_DESIRED_OUTPUT = (1u << 4),
    MV_FLAG_COST           = (1u << 5),
};

// ─────────────────────────────────────────────────────────────────────────────
//  Operations
// ─────────────────────────────────────────────────────────────────────────────
enum class ModelVarOp : u32 {
    Null = 0,
    Create,

    _UnaryStart,
    Relu,
    Softmax,

    _BinaryStart,
    Add,
    Sub,
    MatMul,
    CrossEntropy,
};

inline u32 mv_num_inputs(ModelVarOp op) {
    if (op < ModelVarOp::_UnaryStart)  return 0;
    if (op < ModelVarOp::_BinaryStart) return 1;
    return 2;
}

// ─────────────────────────────────────────────────────────────────────────────
//  ModelVar  —  a node in the computation graph
// ─────────────────────────────────────────────────────────────────────────────
constexpr u32 MODEL_VAR_MAX_INPUTS = 2;

struct ModelVar {
    u32       index;
    u32       flags;

    Matrix*   val;
    Matrix*   grad;    // nullptr if REQUIRES_GRAD is not set

    ModelVarOp             op;
    ModelVar*  inputs[MODEL_VAR_MAX_INPUTS];
};

// ─────────────────────────────────────────────────────────────────────────────
//  Compiled program  (topologically sorted list of vars)
// ─────────────────────────────────────────────────────────────────────────────
struct ModelProgram {
    ModelVar** vars;
    u32        size;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Model context
// ─────────────────────────────────────────────────────────────────────────────
struct ModelContext {
    u32       num_vars;

    ModelVar* input;
    ModelVar* output;
    ModelVar* desired_output;
    ModelVar* cost;

    ModelProgram forward_prog;
    ModelProgram cost_prog;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Training descriptor
// ─────────────────────────────────────────────────────────────────────────────
struct ModelTrainingDesc {
    Matrix* train_images;
    Matrix* train_labels;
    Matrix* test_images;
    Matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Variable constructors  (graph-building API)
// ─────────────────────────────────────────────────────────────────────────────
ModelVar* mv_create(MemArena* arena, ModelContext* model,
                    u32 rows, u32 cols, u32 flags);

ModelVar* mv_relu        (MemArena*, ModelContext*, ModelVar* input,    u32 flags);
ModelVar* mv_softmax     (MemArena*, ModelContext*, ModelVar* input,    u32 flags);
ModelVar* mv_add         (MemArena*, ModelContext*, ModelVar* a, ModelVar* b, u32 flags);
ModelVar* mv_sub         (MemArena*, ModelContext*, ModelVar* a, ModelVar* b, u32 flags);
ModelVar* mv_matmul      (MemArena*, ModelContext*, ModelVar* a, ModelVar* b, u32 flags);
ModelVar* mv_cross_entropy(MemArena*, ModelContext*, ModelVar* p, ModelVar* q, u32 flags);

// ─────────────────────────────────────────────────────────────────────────────
//  Program execution
// ─────────────────────────────────────────────────────────────────────────────
ModelProgram model_prog_create(MemArena* arena, ModelContext* model, ModelVar* out_var);
void         model_prog_compute(ModelProgram* prog);
void         model_prog_compute_grads(ModelProgram* prog);

// ─────────────────────────────────────────────────────────────────────────────
//  High-level model API
// ─────────────────────────────────────────────────────────────────────────────
ModelContext* model_create   (MemArena* arena);
void          model_compile  (MemArena* arena, ModelContext* model);
void          model_feedforward(ModelContext* model);
void          model_train    (ModelContext* model, const ModelTrainingDesc* desc);
