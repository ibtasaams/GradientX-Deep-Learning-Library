#pragma once
#include "base.hpp"

// ── Constants ─────────────────────────────────────────────────────────────────
constexpr u64 ARENA_ALIGN = sizeof(void*);

// ── Types ─────────────────────────────────────────────────────────────────────
struct MemArena {
    u64 reserve_size;
    u64 commit_size;
    u64 pos;
    u64 commit_pos;
};

constexpr u64 ARENA_BASE_POS = sizeof(MemArena);

struct MemArenaTemp {
    MemArena* arena;
    u64       start_pos;
};

// ── Arena lifetime ────────────────────────────────────────────────────────────
MemArena* arena_create(u64 reserve_size, u64 commit_size);
void      arena_destroy(MemArena* arena);

// ── Push / pop ────────────────────────────────────────────────────────────────
void* arena_push(MemArena* arena, u64 size, b32 non_zero);
void  arena_pop(MemArena* arena, u64 size);
void  arena_pop_to(MemArena* arena, u64 pos);
void  arena_clear(MemArena* arena);

// ── Temporary sub-arenas ──────────────────────────────────────────────────────
MemArenaTemp arena_temp_begin(MemArena* arena);
void         arena_temp_end(MemArenaTemp temp);

MemArenaTemp arena_scratch_get(MemArena** conflicts, u32 num_conflicts);
void         arena_scratch_release(MemArenaTemp scratch);

// ── Typed push helpers (replaces C macros with templates) ─────────────────────
template<typename T>
T* push_struct(MemArena* arena, bool zero = true) {
    return reinterpret_cast<T*>(arena_push(arena, sizeof(T), zero ? 0 : 1));
}

template<typename T>
T* push_array(MemArena* arena, u64 count, bool zero = true) {
    return reinterpret_cast<T*>(arena_push(arena, sizeof(T) * count, zero ? 0 : 1));
}

// ── Platform memory ───────────────────────────────────────────────────────────
u32   plat_get_pagesize();
void* plat_mem_reserve(u64 size);
b32   plat_mem_commit(void* ptr, u64 size);
b32   plat_mem_decommit(void* ptr, u64 size);
b32   plat_mem_release(void* ptr, u64 size);
