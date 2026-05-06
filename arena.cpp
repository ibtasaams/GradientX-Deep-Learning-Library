#include "arena.hpp"
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
//  Arena core
// ─────────────────────────────────────────────────────────────────────────────

MemArena* arena_create(u64 reserve_size, u64 commit_size) {
    u32 pagesize = plat_get_pagesize();

    reserve_size = align_up_pow2(reserve_size, pagesize);
    commit_size  = align_up_pow2(commit_size,  pagesize);

    MemArena* arena = reinterpret_cast<MemArena*>(plat_mem_reserve(reserve_size));
    if (!arena) return nullptr;

    if (!plat_mem_commit(arena, commit_size)) {
        plat_mem_release(arena, reserve_size);
        return nullptr;
    }

    arena->reserve_size = reserve_size;
    arena->commit_size  = commit_size;
    arena->pos          = ARENA_BASE_POS;
    arena->commit_pos   = commit_size;

    return arena;
}

void arena_destroy(MemArena* arena) {
    plat_mem_release(arena, arena->reserve_size);
}

void* arena_push(MemArena* arena, u64 size, b32 non_zero) {
    u64 pos_aligned = align_up_pow2(arena->pos, ARENA_ALIGN);
    u64 new_pos     = pos_aligned + size;

    if (new_pos > arena->reserve_size) return nullptr;

    if (new_pos > arena->commit_pos) {
        u64 new_commit_pos = new_pos;
        new_commit_pos += arena->commit_size - 1;
        new_commit_pos -= new_commit_pos % arena->commit_size;
        new_commit_pos  = dl_min(new_commit_pos, arena->reserve_size);

        u8*  mem         = reinterpret_cast<u8*>(arena) + arena->commit_pos;
        u64  commit_size = new_commit_pos - arena->commit_pos;

        if (!plat_mem_commit(mem, commit_size)) return nullptr;

        arena->commit_pos = new_commit_pos;
    }

    arena->pos = new_pos;
    u8* out    = reinterpret_cast<u8*>(arena) + pos_aligned;

    if (!non_zero) {
        std::memset(out, 0, size);
    }

    return out;
}

void arena_pop(MemArena* arena, u64 size) {
    size       = dl_min(size, arena->pos - ARENA_BASE_POS);
    arena->pos -= size;
}

void arena_pop_to(MemArena* arena, u64 pos) {
    u64 size = (pos < arena->pos) ? (arena->pos - pos) : 0;
    arena_pop(arena, size);
}

void arena_clear(MemArena* arena) {
    arena_pop_to(arena, ARENA_BASE_POS);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Temporary arenas
// ─────────────────────────────────────────────────────────────────────────────

MemArenaTemp arena_temp_begin(MemArena* arena) {
    return MemArenaTemp{ arena, arena->pos };
}

void arena_temp_end(MemArenaTemp temp) {
    arena_pop_to(temp.arena, temp.start_pos);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scratch arenas (two per thread, conflict-aware)
// ─────────────────────────────────────────────────────────────────────────────

static thread_local MemArena* s_scratch_arenas[2] = { nullptr, nullptr };

MemArenaTemp arena_scratch_get(MemArena** conflicts, u32 num_conflicts) {
    i32 scratch_index = -1;

    for (i32 i = 0; i < 2; i++) {
        bool conflict_found = false;

        for (u32 j = 0; j < num_conflicts; j++) {
            if (s_scratch_arenas[i] == conflicts[j]) {
                conflict_found = true;
                break;
            }
        }

        if (!conflict_found) {
            scratch_index = i;
            break;
        }
    }

    if (scratch_index == -1) {
        return MemArenaTemp{ nullptr, 0 };
    }

    MemArena** selected = &s_scratch_arenas[scratch_index];

    if (*selected == nullptr) {
        *selected = arena_create(MiB(64), MiB(1));
    }

    return arena_temp_begin(*selected);
}

void arena_scratch_release(MemArenaTemp scratch) {
    arena_temp_end(scratch);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Platform implementations
// ─────────────────────────────────────────────────────────────────────────────

#if defined(_WIN32)

#include <windows.h>

u32 plat_get_pagesize() {
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    return si.dwPageSize;
}
void* plat_mem_reserve(u64 size) {
    return VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_READWRITE);
}
b32 plat_mem_commit(void* ptr, u64 size) {
    return VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) != nullptr ? 1 : 0;
}
b32 plat_mem_decommit(void* ptr, u64 size) {
    return VirtualFree(ptr, size, MEM_DECOMMIT) ? 1 : 0;
}
b32 plat_mem_release(void* ptr, u64 size) {
    return VirtualFree(ptr, 0, MEM_RELEASE) ? 1 : 0;
}

#elif defined(__linux__)

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include <unistd.h>
#include <sys/mman.h>

u32 plat_get_pagesize() {
    return static_cast<u32>(sysconf(_SC_PAGESIZE));
}
void* plat_mem_reserve(u64 size) {
    void* out = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (out == MAP_FAILED) ? nullptr : out;
}
b32 plat_mem_commit(void* ptr, u64 size) {
    return mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0 ? 1 : 0;
}
b32 plat_mem_decommit(void* ptr, u64 size) {
    if (mprotect(ptr, size, PROT_NONE) != 0) return 0;
    return madvise(ptr, size, MADV_DONTNEED) == 0 ? 1 : 0;
}
b32 plat_mem_release(void* ptr, u64 size) {
    return munmap(ptr, size) == 0 ? 1 : 0;
}

#endif
