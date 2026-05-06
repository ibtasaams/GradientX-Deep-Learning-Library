#include "model.hpp"
#include "prng.hpp"
#include <cstring>
#include <cstdio>

#define ANSI_RESET   "\x1b[0m"
#define ANSI_CYAN    "\x1b[36m"
#define ANSI_DIM     "\x1b[2m"
#define ANSI_YELLOW  "\x1b[33m"

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static ModelVar* _mv_unary_impl(
    MemArena* arena, ModelContext* model,
    ModelVar* input,
    u32 rows, u32 cols,
    u32 flags, ModelVarOp op)
{
    if (input->flags & MV_FLAG_REQUIRES_GRAD)
        flags |= MV_FLAG_REQUIRES_GRAD;

    ModelVar* out  = mv_create(arena, model, rows, cols, flags);
    out->op        = op;
    out->inputs[0] = input;
    out->inputs[1] = nullptr;
    return out;
}

static ModelVar* _mv_binary_impl(
    MemArena* arena, ModelContext* model,
    ModelVar* a, ModelVar* b,
    u32 rows, u32 cols,
    u32 flags, ModelVarOp op)
{
    if ((a->flags & MV_FLAG_REQUIRES_GRAD) ||
        (b->flags & MV_FLAG_REQUIRES_GRAD))
        flags |= MV_FLAG_REQUIRES_GRAD;

    ModelVar* out  = mv_create(arena, model, rows, cols, flags);
    out->op        = op;
    out->inputs[0] = a;
    out->inputs[1] = b;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Variable constructors
// ─────────────────────────────────────────────────────────────────────────────

ModelVar* mv_create(
    MemArena* arena, ModelContext* model,
    u32 rows, u32 cols, u32 flags)
{
    ModelVar* out       = push_struct<ModelVar>(arena);
    out->index          = model->num_vars++;
    out->flags          = flags;
    out->op             = ModelVarOp::Create;
    out->inputs[0]      = nullptr;
    out->inputs[1]      = nullptr;
    out->val            = mat_create(arena, rows, cols);
    out->grad           = nullptr;

    if (flags & MV_FLAG_REQUIRES_GRAD)
        out->grad = mat_create(arena, rows, cols);

    if (flags & MV_FLAG_INPUT)          model->input          = out;
    if (flags & MV_FLAG_OUTPUT)         model->output         = out;
    if (flags & MV_FLAG_DESIRED_OUTPUT) model->desired_output = out;
    if (flags & MV_FLAG_COST)           model->cost           = out;

    return out;
}

ModelVar* mv_relu(MemArena* arena, ModelContext* model, ModelVar* input, u32 flags) {
    return _mv_unary_impl(arena, model, input,
                          input->val->rows, input->val->cols,
                          flags, ModelVarOp::Relu);
}

ModelVar* mv_softmax(MemArena* arena, ModelContext* model, ModelVar* input, u32 flags) {
    return _mv_unary_impl(arena, model, input,
                          input->val->rows, input->val->cols,
                          flags, ModelVarOp::Softmax);
}

ModelVar* mv_add(MemArena* arena, ModelContext* model,
                 ModelVar* a, ModelVar* b, u32 flags)
{
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols)
        return nullptr;
    return _mv_binary_impl(arena, model, a, b,
                           a->val->rows, a->val->cols,
                           flags, ModelVarOp::Add);
}

ModelVar* mv_sub(MemArena* arena, ModelContext* model,
                 ModelVar* a, ModelVar* b, u32 flags)
{
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols)
        return nullptr;
    return _mv_binary_impl(arena, model, a, b,
                           a->val->rows, a->val->cols,
                           flags, ModelVarOp::Sub);
}

ModelVar* mv_matmul(MemArena* arena, ModelContext* model,
                    ModelVar* a, ModelVar* b, u32 flags)
{
    if (a->val->cols != b->val->rows) return nullptr;
    return _mv_binary_impl(arena, model, a, b,
                           a->val->rows, b->val->cols,
                           flags, ModelVarOp::MatMul);
}

ModelVar* mv_cross_entropy(MemArena* arena, ModelContext* model,
                           ModelVar* p, ModelVar* q, u32 flags)
{
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols)
        return nullptr;
    return _mv_binary_impl(arena, model, p, q,
                           p->val->rows, p->val->cols,
                           flags, ModelVarOp::CrossEntropy);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Program compilation  (iterative DFS → topological sort)
// ─────────────────────────────────────────────────────────────────────────────

ModelProgram model_prog_create(
    MemArena* arena, ModelContext* model, ModelVar* out_var)
{
    MemArenaTemp scratch = arena_scratch_get(&arena, 1);

    b8*        visited   = push_array<b8>       (scratch.arena, model->num_vars);
    ModelVar** stack     = push_array<ModelVar*>(scratch.arena, model->num_vars, false);
    ModelVar** out       = push_array<ModelVar*>(scratch.arena, model->num_vars, false);
    u32 stack_size = 0, out_size = 0;

    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        ModelVar* cur = stack[--stack_size];

        if (cur->index >= model->num_vars) continue;

        if (visited[cur->index]) {
            if (out_size < model->num_vars)
                out[out_size++] = cur;
            continue;
        }

        visited[cur->index] = 1;

        if (stack_size < model->num_vars)
            stack[stack_size++] = cur;

        u32 num_inputs = mv_num_inputs(cur->op);
        for (u32 i = 0; i < num_inputs; ++i) {
            ModelVar* inp = cur->inputs[i];
            if (!inp || inp->index >= model->num_vars || visited[inp->index])
                continue;

            // remove inp from stack if it's already there (re-prioritise)
            for (u32 j = 0; j < stack_size; ++j) {
                if (stack[j] == inp) {
                    for (u32 k = j; k < stack_size - 1; ++k)
                        stack[k] = stack[k + 1];
                    --stack_size;
                    break;
                }
            }

            if (stack_size < model->num_vars)
                stack[stack_size++] = inp;
        }
    }

    ModelProgram prog;
    prog.size = out_size;
    prog.vars = push_array<ModelVar*>(arena, out_size, false);
    std::memcpy(prog.vars, out, sizeof(ModelVar*) * out_size);

    arena_scratch_release(scratch);
    return prog;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Forward pass
// ─────────────────────────────────────────────────────────────────────────────

void model_prog_compute(ModelProgram* prog) {
    for (u32 i = 0; i < prog->size; ++i) {
        ModelVar* cur = prog->vars[i];
        ModelVar* a   = cur->inputs[0];
        ModelVar* b   = cur->inputs[1];

        switch (cur->op) {
            case ModelVarOp::Null:
            case ModelVarOp::Create:
            case ModelVarOp::_UnaryStart:
            case ModelVarOp::_BinaryStart:
                break;

            case ModelVarOp::Relu:
                mat_relu(cur->val, a->val);
                break;

            case ModelVarOp::Softmax:
                mat_softmax(cur->val, a->val);
                break;

            case ModelVarOp::Add:
                mat_add(cur->val, a->val, b->val);
                break;

            case ModelVarOp::Sub:
                mat_sub(cur->val, a->val, b->val);
                break;

            case ModelVarOp::MatMul:
                mat_mul(cur->val, a->val, b->val, 1, 0, 0);
                break;

            case ModelVarOp::CrossEntropy:
                mat_cross_entropy(cur->val, a->val, b->val);
                break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Backward pass  (reverse-mode autodiff, accumulates into grads)
// ─────────────────────────────────────────────────────────────────────────────

void model_prog_compute_grads(ModelProgram* prog) {
    // Zero non-parameter intermediate grads
    for (u32 i = 0; i < prog->size; ++i) {
        ModelVar* cur = prog->vars[i];
        if (!(cur->flags & MV_FLAG_REQUIRES_GRAD)) continue;
        if (cur->flags & MV_FLAG_PARAMETER)        continue;
        mat_clear(cur->grad);
    }

    // Seed: dL/dL = 1
    mat_fill(prog->vars[prog->size - 1]->grad, 1.0f);

    for (i64 i = static_cast<i64>(prog->size) - 1; i >= 0; --i) {
        ModelVar* cur = prog->vars[i];

        if (!(cur->flags & MV_FLAG_REQUIRES_GRAD)) continue;

        ModelVar*  a          = cur->inputs[0];
        ModelVar*  b          = cur->inputs[1];
        u32        num_inputs = mv_num_inputs(cur->op);

        // Skip if no input needs a gradient
        bool a_needs = a && (a->flags & MV_FLAG_REQUIRES_GRAD);
        bool b_needs = b && (b->flags & MV_FLAG_REQUIRES_GRAD);
        if (num_inputs == 1 && !a_needs) continue;
        if (num_inputs == 2 && !a_needs && !b_needs) continue;

        switch (cur->op) {
            case ModelVarOp::Null:
            case ModelVarOp::Create:
            case ModelVarOp::_UnaryStart:
            case ModelVarOp::_BinaryStart:
                break;

            case ModelVarOp::Relu:
                mat_relu_add_grad(a->grad, a->val, cur->grad);
                break;

            case ModelVarOp::Softmax:
                mat_softmax_add_grad(a->grad, cur->val, cur->grad);
                break;

            case ModelVarOp::Add:
                if (a_needs) mat_add(a->grad, a->grad, cur->grad);
                if (b_needs) mat_add(b->grad, b->grad, cur->grad);
                break;

            case ModelVarOp::Sub:
                if (a_needs) mat_add(a->grad, a->grad, cur->grad);
                if (b_needs) mat_sub(b->grad, b->grad, cur->grad);
                break;

            case ModelVarOp::MatMul:
                // dL/dA = dL/dC · B^T
                if (a_needs) mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
                // dL/dB = A^T · dL/dC
                if (b_needs) mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
                break;

            case ModelVarOp::CrossEntropy: {
                ModelVar* p = a;
                ModelVar* q = b;
                mat_cross_entropy_add_grad(
                    a_needs ? p->grad : nullptr,
                    b_needs ? q->grad : nullptr,
                    p->val, q->val, cur->grad
                );
            } break;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  High-level API
// ─────────────────────────────────────────────────────────────────────────────

ModelContext* model_create(MemArena* arena) {
    return push_struct<ModelContext>(arena);
}

void model_compile(MemArena* arena, ModelContext* model) {
    if (model->output)
        model->forward_prog = model_prog_create(arena, model, model->output);
    if (model->cost)
        model->cost_prog = model_prog_create(arena, model, model->cost);
}

void model_feedforward(ModelContext* model) {
    model_prog_compute(&model->forward_prog);
}

void model_train(ModelContext* model, const ModelTrainingDesc* desc) {
    Matrix* train_images = desc->train_images;
    Matrix* train_labels = desc->train_labels;
    Matrix* test_images  = desc->test_images;
    Matrix* test_labels  = desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 input_size   = train_images->cols;
    u32 output_size  = train_labels->cols;
    u32 num_tests    = test_images->rows;
    u32 num_batches  = num_examples / desc->batch_size;

    MemArenaTemp scratch = arena_scratch_get(nullptr, 0);
    u32* order = push_array<u32>(scratch.arena, num_examples, false);
    for (u32 i = 0; i < num_examples; ++i) order[i] = i;

    std::printf(ANSI_DIM "    [Backend] Initializing Fisher-Yates dataset shuffle..." ANSI_RESET "\n");
    std::printf(ANSI_DIM "    [Backend] Preparing Stochastic Gradient Descent (SGD)..." ANSI_RESET "\n");
    std::printf(ANSI_DIM "    [Backend] Configuration: Batch Size: %u | LR: %.4f | Examples: %u" ANSI_RESET "\n\n",
                desc->batch_size, desc->learning_rate, num_examples);

    for (u32 epoch = 0; epoch < desc->epochs; ++epoch) {
        
        std::printf(ANSI_DIM "    [Backend] Epoch %u: Executing Forward Pass & Backpropagation..." ANSI_RESET "\n", epoch + 1);

        // ── Fisher-Yates shuffle ─────────────────────────────────────────────
        for (u32 i = 0; i < num_examples; ++i) {
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;
            u32 tmp = order[b]; order[b] = order[a]; order[a] = tmp;
        }

        for (u32 batch = 0; batch < num_batches; ++batch) {

            // Zero parameter grads before accumulating
            for (u32 i = 0; i < model->cost_prog.size; ++i) {
                ModelVar* v = model->cost_prog.vars[i];
                if (v->flags & MV_FLAG_PARAMETER) mat_clear(v->grad);
            }

            f32 avg_cost = 0.0f;

            for (u32 s = 0; s < desc->batch_size; ++s) {
                u32 idx = order[batch * desc->batch_size + s];

                std::memcpy(
                    model->input->val->data,
                    train_images->data + idx * input_size,
                    sizeof(f32) * input_size);

                std::memcpy(
                    model->desired_output->val->data,
                    train_labels->data + idx * output_size,
                    sizeof(f32) * output_size);

                model_prog_compute(&model->cost_prog);
                model_prog_compute_grads(&model->cost_prog);

                avg_cost += mat_sum(model->cost->val);
            }
            avg_cost /= static_cast<f32>(desc->batch_size);

            // ── SGD update ───────────────────────────────────────────────────
            for (u32 i = 0; i < model->cost_prog.size; ++i) {
                ModelVar* v = model->cost_prog.vars[i];
                if (!(v->flags & MV_FLAG_PARAMETER)) continue;

                mat_scale(v->grad,
                    desc->learning_rate / static_cast<f32>(desc->batch_size));
                mat_sub(v->val, v->val, v->grad);
            }

            std::printf("    " ANSI_CYAN "[Epoch %2u/%2u]" ANSI_RESET " Batch %4u/%4u | Cost: %.4f | Status: Fwd/Bwd... | [",
                epoch + 1, desc->epochs,
                batch + 1, num_batches, avg_cost);
            
            u32 bar_width = 30;
            u32 filled = ((batch + 1) * bar_width) / num_batches;
            for (u32 b = 0; b < bar_width; ++b) {
                if (b < filled) std::printf(ANSI_CYAN "█" ANSI_RESET);
                else std::printf(ANSI_DIM "░" ANSI_RESET);
            }
            std::printf("]   \r");
            std::fflush(stdout);
        }
        std::printf("\n");

        std::printf(ANSI_DIM "    [Backend] Epoch %u completed. Evaluating accuracy on test-set (forward pass only)..." ANSI_RESET "\n", epoch + 1);

        // ── Test-set evaluation ──────────────────────────────────────────────
        u32 num_correct = 0;
        f32 avg_cost    = 0.0f;

        for (u32 i = 0; i < num_tests; ++i) {
            std::memcpy(
                model->input->val->data,
                test_images->data + i * input_size,
                sizeof(f32) * input_size);

            std::memcpy(
                model->desired_output->val->data,
                test_labels->data + i * output_size,
                sizeof(f32) * output_size);

            model_prog_compute(&model->cost_prog);

            avg_cost    += mat_sum(model->cost->val);
            num_correct += (mat_argmax(model->output->val) ==
                            mat_argmax(model->desired_output->val));
        }

        avg_cost /= static_cast<f32>(num_tests);
        std::printf(
            "    " ANSI_YELLOW "=> Accuracy: %5u / %5u (%.2f%%), Average Cost: %.4f" ANSI_RESET "\n\n",
            num_correct, num_tests,
            static_cast<f32>(num_correct) / num_tests * 100.0f,
            avg_cost);
    }

    arena_scratch_release(scratch);
}
