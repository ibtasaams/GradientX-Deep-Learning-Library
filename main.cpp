#include "base.hpp"
#include "arena.hpp"
#include "prng.hpp"
#include "matrix.hpp"
#include "model.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>

#ifdef _WIN32
#include <windows.h>
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Enable ANSI escape codes on Windows Terminal
// ─────────────────────────────────────────────────────────────────────────────
static void enable_ansi_and_utf8() {
#ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        GetConsoleMode(hOut, &mode);
        SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
    SetConsoleOutputCP(65001);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
//  Color helpers for pretty terminal output
// ─────────────────────────────────────────────────────────────────────────────
#define ANSI_RESET   "\x1b[0m"
#define ANSI_BOLD    "\x1b[1m"
#define ANSI_CYAN    "\x1b[36m"
#define ANSI_GREEN   "\x1b[32m"
#define ANSI_YELLOW  "\x1b[33m"
#define ANSI_RED     "\x1b[31m"
#define ANSI_MAGENTA "\x1b[35m"
#define ANSI_DIM     "\x1b[2m"

// ─────────────────────────────────────────────────────────────────────────────
//  Class names helper
// ─────────────────────────────────────────────────────────────────────────────
static char custom_class_names[100][64];
static u32 num_custom_classes = 0;

static void load_custom_classes() {
    std::FILE* f = std::fopen("custom_classes.txt", "r");
    if (!f) return;
    while (std::fscanf(f, "%63s", custom_class_names[num_custom_classes]) == 1) {
        num_custom_classes++;
        if (num_custom_classes >= 100) break;
    }
    std::fclose(f);
}

static const char* get_class_name(int dataset_choice, u64 class_idx) {
    if (dataset_choice == 2) {
        const char* fashion_classes[] = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        };
        return fashion_classes[class_idx];
    } else if (dataset_choice == 3) {
        if (class_idx < num_custom_classes) return custom_class_names[class_idx];
        static char fallback[32];
        std::sprintf(fallback, "Class %u", (unsigned)class_idx);
        return fallback;
    } else {
        const char* digit_classes[] = {
            "0", "1", "2", "3", "4",
            "5", "6", "7", "8", "9"
        };
        return digit_classes[class_idx];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  ASCII-art Image Viewer (uses 24-bit TrueColor)
// ─────────────────────────────────────────────────────────────────────────────
static void draw_image(const f32* data, u32 input_size) {
    u32 side_len = (u32)std::sqrt((float)input_size);

    std::printf("    +");
    for (u32 x = 0; x < side_len * 2; ++x) std::printf("-");
    std::printf("+\n");

    for (u32 y = 0; y < side_len; ++y) {
        std::printf("    |");
        for (u32 x = 0; x < side_len; ++x) {
            f32 v = data[x + y * side_len];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            u32 c = static_cast<u32>(v * 255.0f);
            // Print two spaces with truecolor grayscale background
            std::printf("\x1b[48;2;%u;%u;%um  \x1b[0m", c, c, c);
        }
        std::printf("|\n");
    }

    std::printf("    +");
    for (u32 x = 0; x < side_len * 2; ++x) std::printf("-");
    std::printf("+\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Print a horizontal bar chart for confidence values
// ─────────────────────────────────────────────────────────────────────────────
static void print_confidence_bar(int dataset_choice, u32 class_idx, f32 confidence, u64 predicted) {
    const char* color = (class_idx == predicted) ? ANSI_GREEN ANSI_BOLD : ANSI_DIM;
    u32 bar_len = static_cast<u32>(confidence * 40.0f);

    std::printf("    %s %11s: ", color, get_class_name(dataset_choice, class_idx));
    for (u32 i = 0; i < 40; ++i) {
        if (i < bar_len) std::printf("█");
        else             std::printf("░");
    }
    std::printf(" %6.2f%%" ANSI_RESET "\n", confidence * 100.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Save a matrix to a binary file
// ─────────────────────────────────────────────────────────────────────────────
static void save_matrix_bin(const Matrix* mat, const char* filename) {
    std::FILE* f = std::fopen(filename, "wb");
    if (!f) { std::printf("  [!] Failed to save %s\n", filename); return; }
    // Write rows, cols header
    std::fwrite(&mat->rows, sizeof(u32), 1, f);
    std::fwrite(&mat->cols, sizeof(u32), 1, f);
    // Write data
    std::fwrite(mat->data, sizeof(f32), mat->size(), f);
    std::fclose(f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Network definition — returns weight pointers for saving
// ─────────────────────────────────────────────────────────────────────────────
struct ModelWeights {
    ModelVar *W0, *W1, *W2;
    ModelVar *bias0, *bias1, *bias2;
};

static ModelWeights create_model(MemArena* arena, ModelContext* model, u32 input_size, u32 num_classes) {
    // ── Input ────────────────────────────────────────────────────────────────
    ModelVar* input = mv_create(arena, model, input_size, 1, MV_FLAG_INPUT);

    // Expand hidden layer size for larger datasets
    u32 hidden = (input_size > 784) ? 64 : 16;

    // ── Weight matrices ───────────────────────────────────────────────────────
    ModelVar* W0 = mv_create(arena, model, hidden,  input_size, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* W1 = mv_create(arena, model, hidden,  hidden,  MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* W2 = mv_create(arena, model, num_classes, hidden,  MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // ── Xavier / Glorot uniform init ─────────────────────────────────────────
    f32 b0 = sqrtf(6.0f / (input_size + hidden));
    f32 b1 = sqrtf(6.0f / (hidden + hidden));
    f32 b2 = sqrtf(6.0f / (hidden + num_classes));
    mat_fill_rand(W0->val, -b0, b0);
    mat_fill_rand(W1->val, -b1, b1);
    mat_fill_rand(W2->val, -b2, b2);

    // ── Bias vectors (zero-initialised by arena_push) ─────────────────────────
    ModelVar* bias0 = mv_create(arena, model, hidden, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* bias1 = mv_create(arena, model, hidden, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    ModelVar* bias2 = mv_create(arena, model, num_classes, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    // ── Layer 0  (Dense → ReLU) ───────────────────────────────────────────────
    ModelVar* z0a = mv_matmul(arena, model, W0, input, 0);
    ModelVar* z0b = mv_add   (arena, model, z0a, bias0, 0);
    ModelVar* a0  = mv_relu  (arena, model, z0b, 0);

    // ── Layer 1  (Dense → ReLU → residual add) ───────────────────────────────
    ModelVar* z1a = mv_matmul(arena, model, W1, a0, 0);
    ModelVar* z1b = mv_add   (arena, model, z1a, bias1, 0);
    ModelVar* z1c = mv_relu  (arena, model, z1b, 0);
    ModelVar* a1  = mv_add   (arena, model, a0, z1c, 0);   // skip connection

    // ── Layer 2  (Dense → Softmax) ────────────────────────────────────────────
    ModelVar* z2a   = mv_matmul (arena, model, W2, a1, 0);
    ModelVar* z2b   = mv_add    (arena, model, z2a, bias2, 0);
    ModelVar* output = mv_softmax(arena, model, z2b, MV_FLAG_OUTPUT);

    // ── Desired output and loss ───────────────────────────────────────────────
    ModelVar* y    = mv_create       (arena, model, num_classes, 1, MV_FLAG_DESIRED_OUTPUT);
    ModelVar* cost = mv_cross_entropy(arena, model, y, output, MV_FLAG_COST);
    (void)cost;

    return { W0, W1, W2, bias0, bias1, bias2 };
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
static u64 raw_argmax(const f32* data, u32 n);
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    enable_ansi_and_utf8();

    // Seed PRNG with current time so each run is different
    prng_seed(static_cast<u64>(std::time(nullptr)), 42);

    std::printf("\n");
    std::printf(ANSI_BOLD ANSI_CYAN
        "    ╔══════════════════════════════════════════════════════╗\n"
        "    ║        GradientX  —  Deep Learning Library          ║\n"
        "    ║        Image Classification Neural Network           ║\n"
        "    ╚══════════════════════════════════════════════════════╝\n"
        ANSI_RESET "\n");

    std::printf("    Select Dataset to Train On:\n");
    std::printf("    1) Standard MNIST (Handwritten Digits)\n");
    std::printf("    2) Fashion-MNIST  (Clothing Items)\n");
    std::printf("    3) Custom Dataset (Real-life Photos)\n");
    std::printf(ANSI_CYAN "    > " ANSI_RESET);

    int dataset_choice = 1;
    if (std::scanf("%d", &dataset_choice) != 1) dataset_choice = 1;
    if (dataset_choice < 1 || dataset_choice > 3) dataset_choice = 1;

    const char* prefix = (dataset_choice == 2) ? "fashion_" : (dataset_choice == 3) ? "custom_" : "";
    char train_img_path[256]; std::sprintf(train_img_path, "%strain_images.mat", prefix);
    char test_img_path[256];  std::sprintf(test_img_path,  "%stest_images.mat", prefix);
    char train_lbl_path[256]; std::sprintf(train_lbl_path, "%strain_labels.mat", prefix);
    char test_lbl_path[256];  std::sprintf(test_lbl_path,  "%stest_labels.mat", prefix);

    std::printf("\n");

    MemArena* perm = arena_create(GiB(1), MiB(1));
    
    u32 train_size = 60000;
    u32 test_size  = 10000;
    u32 input_size = 784;
    u32 num_classes = 10;

    if (dataset_choice == 3) {
        std::FILE* meta = std::fopen("custom_meta.txt", "r");
        if (meta) {
            std::fscanf(meta, "%u %u %u %u", &train_size, &test_size, &input_size, &num_classes);
            std::fclose(meta);
        } else {
            std::printf(ANSI_RED "    [!] Error: Could not find custom_meta.txt." ANSI_RESET "\n");
            std::printf("        Please run: python import_custom_dataset.py <your_image_folder>\n");
            std::exit(1);
        }
        load_custom_classes();
    }

    // ── Load raw binary files produced by python scripts ─────────────────────
    std::printf(ANSI_DIM "    Loading dataset..." ANSI_RESET "\n");
    Matrix* train_images = mat_load(perm, train_size, input_size, train_img_path);
    Matrix* test_images  = mat_load(perm, test_size, input_size, test_img_path);
    Matrix* train_labels = mat_create(perm, train_size, num_classes);
    Matrix* test_labels  = mat_create(perm, test_size, num_classes);

    // One-hot encode labels
    {
        Matrix* trl_raw = mat_load(perm, train_size, 1, train_lbl_path);
        Matrix* tel_raw = mat_load(perm, test_size, 1, test_lbl_path);

        for (u32 i = 0; i < train_size; ++i) {
            u32 class_idx = static_cast<u32>(trl_raw->data[i]);
            if (class_idx < num_classes) train_labels->data[i * num_classes + class_idx] = 1.0f;
        }
        for (u32 i = 0; i < test_size; ++i) {
            u32 class_idx = static_cast<u32>(tel_raw->data[i]);
            if (class_idx < num_classes) test_labels->data[i * num_classes + class_idx] = 1.0f;
        }
    }
    std::printf(ANSI_GREEN "    ✓ Dataset loaded: %u train / %u test images" ANSI_RESET "\n\n", train_size, test_size);

    // ── Show a random test image (different each run) ─────────────────────────
    u32 sample_idx = prng_rand() % test_size;
    std::printf(ANSI_BOLD "    Sample image from test set (index %u):" ANSI_RESET "\n", sample_idx);
    draw_image(test_images->data + static_cast<u64>(sample_idx) * input_size, input_size);
    u64 sample_label = raw_argmax(test_labels->data + static_cast<u64>(sample_idx) * num_classes, num_classes);
    std::printf("    Actual class: %s\n\n", get_class_name(dataset_choice, sample_label));

    // ── Build and compile model ───────────────────────────────────────────────
    u32 hidden = (input_size > 784) ? 64 : 16;
    std::printf(ANSI_DIM "    Building neural network: %u → Dense(%u)+ReLU → Residual(%u)+ReLU → Dense(%u)+Softmax" ANSI_RESET "\n", input_size, hidden, hidden, num_classes);
    ModelContext* model = model_create(perm);
    ModelWeights weights = create_model(perm, model, input_size, num_classes);
    model_compile(perm, model);
    std::printf(ANSI_GREEN "    ✓ Model compiled successfully" ANSI_RESET "\n\n");

    // ── Pre-training prediction ───────────────────────────────────────────────
    std::memcpy(model->input->val->data,
                test_images->data + static_cast<u64>(sample_idx) * input_size,
                sizeof(f32) * input_size);
    model_feedforward(model);

    std::printf(ANSI_YELLOW "    Pre-training output:  " ANSI_RESET);
    for (u32 i = 0; i < num_classes; ++i)
        std::printf("%.4f ", model->output->val->data[i]);
    std::printf("\n");
    std::printf(ANSI_YELLOW "    Pre-training prediction: %s" ANSI_RESET "\n\n",
                get_class_name(dataset_choice, mat_argmax(model->output->val)));

    // ── Train ─────────────────────────────────────────────────────────────────
    std::printf(ANSI_BOLD ANSI_MAGENTA
        "    ═══════════════════════ TRAINING ═══════════════════════\n"
        ANSI_RESET "\n");

    ModelTrainingDesc desc{};
    desc.train_images  = train_images;
    desc.train_labels  = train_labels;
    desc.test_images   = test_images;
    desc.test_labels   = test_labels;
    desc.epochs        = 10;
    desc.batch_size    = 50;
    desc.learning_rate = 0.01f;

    model_train(model, &desc);

    std::printf("\n" ANSI_BOLD ANSI_GREEN
        "    ═══════════════════ TRAINING COMPLETE ═══════════════════\n"
        ANSI_RESET "\n");

    // ── Save weights for web presentation ─────────────────────────────────────
    std::printf(ANSI_DIM "    Saving trained weights..." ANSI_RESET "\n");
    char w0_path[256]; std::sprintf(w0_path, "%sw0.bin", prefix);
    char w1_path[256]; std::sprintf(w1_path, "%sw1.bin", prefix);
    char w2_path[256]; std::sprintf(w2_path, "%sw2.bin", prefix);
    char b0_path[256]; std::sprintf(b0_path, "%sb0.bin", prefix);
    char b1_path[256]; std::sprintf(b1_path, "%sb1.bin", prefix);
    char b2_path[256]; std::sprintf(b2_path, "%sb2.bin", prefix);
    save_matrix_bin(weights.W0->val,    w0_path);
    save_matrix_bin(weights.W1->val,    w1_path);
    save_matrix_bin(weights.W2->val,    w2_path);
    save_matrix_bin(weights.bias0->val, b0_path);
    save_matrix_bin(weights.bias1->val, b1_path);
    save_matrix_bin(weights.bias2->val, b2_path);
    std::printf(ANSI_GREEN "    ✓ Weights saved for %s model" ANSI_RESET "\n\n", (dataset_choice == 2) ? "Fashion-MNIST" : "MNIST");

    // ── Post-training prediction on the same sampled test digit ───────────────
    std::memcpy(model->input->val->data,
                test_images->data + static_cast<u64>(sample_idx) * input_size,
                sizeof(f32) * input_size);
    model_feedforward(model);

    std::printf(ANSI_BOLD "    Post-training prediction:\n" ANSI_RESET);
    u64 predicted = mat_argmax(model->output->val);
    u64 actual    = raw_argmax(test_labels->data + static_cast<u64>(sample_idx) * num_classes, num_classes);
    for (u32 d = 0; d < num_classes; ++d)
        print_confidence_bar(dataset_choice, d, model->output->val->data[d], predicted);

    const char* status = (predicted == actual) ? ANSI_GREEN "✓ CORRECT" : ANSI_RED "✗ WRONG";
    std::printf("\n    Predicted: " ANSI_BOLD "%s" ANSI_RESET
                "  |  Actual: " ANSI_BOLD "%s" ANSI_RESET
                "  |  %s" ANSI_RESET "\n",
                get_class_name(dataset_choice, predicted),
                get_class_name(dataset_choice, actual),
                status);

    // ── Interactive single-image prediction loop ──────────────────────────────
    std::printf("\n" ANSI_BOLD ANSI_CYAN
        "    ╔══════════════════════════════════════════════════════╗\n"
        "    ║           Interactive Prediction Mode                ║\n"
        "    ╚══════════════════════════════════════════════════════╝\n"
        ANSI_RESET "\n");
    std::printf("    Enter a test index (0-9999) to predict, or -1 to quit:\n\n");

    int idx = 0;
    while (std::printf(ANSI_CYAN "    > " ANSI_RESET),
           std::fflush(stdout),
           std::scanf("%d", &idx) == 1 && idx >= 0) {

        if (idx >= (int)test_size) {
            std::printf(ANSI_RED "    Index out of range. Try 0-%u." ANSI_RESET "\n", test_size - 1);
            continue;
        }

        const f32* img = test_images->data + static_cast<u64>(idx) * input_size;
        std::printf("\n" ANSI_BOLD "    Test image #%d:" ANSI_RESET "\n", idx);
        draw_image(img, input_size);

        std::memcpy(model->input->val->data, img, sizeof(f32) * input_size);
        model_feedforward(model);

        u64 pred   = mat_argmax(model->output->val);
        u64 actual_class = raw_argmax(test_labels->data + static_cast<u64>(idx) * num_classes, num_classes);

        std::printf(ANSI_BOLD "    Confidence breakdown:\n" ANSI_RESET);
        for (u32 d = 0; d < num_classes; ++d)
            print_confidence_bar(dataset_choice, d, model->output->val->data[d], pred);

        const char* res = (pred == actual_class) ? ANSI_GREEN "✓ CORRECT" : ANSI_RED "✗ WRONG";
        std::printf("\n    Predicted: " ANSI_BOLD "%s" ANSI_RESET
                    "  |  Actual: " ANSI_BOLD "%s" ANSI_RESET
                    "  |  %s" ANSI_RESET "\n\n",
                    get_class_name(dataset_choice, pred),
                    get_class_name(dataset_choice, actual_class),
                    res);
    }

    std::printf("\n" ANSI_DIM "    Goodbye!" ANSI_RESET "\n");
    arena_destroy(perm);
    return 0;
}

// Helper: argmax over a raw array of 'n' elements
static u64 raw_argmax(const f32* data, u32 n) {
    u64 best = 0;
    for (u32 i = 1; i < n; ++i)
        if (data[i] > data[best]) best = i;
    return best;
}
