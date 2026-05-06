cmake_minimum_required(VERSION 3.16)
project(DeepLearnCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Compiler flags ─────────────────────────────────────────────────────────────
add_compile_options(
    -O2
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# ── Sources ───────────────────────────────────────────────────────────────────
add_executable(deeplearn
    main.cpp
    arena.cpp
    prng.cpp
    matrix.cpp
    model.cpp
)

target_link_libraries(deeplearn PRIVATE m)

# ── Install (optional) ────────────────────────────────────────────────────────
install(TARGETS deeplearn DESTINATION bin)
