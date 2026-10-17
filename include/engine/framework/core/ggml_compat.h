#pragma once

#ifdef AUDIOCPP_GGML_IK_LLAMA

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <ggml.h>
#include <ggml-backend.h>

// 1. gguf.h is baked into ggml.h in ik_llama - no separate include needed.

// 2. ggml_type_traits_t is already the correct name in ik_llama.
//    But ggml_get_type_traits was renamed:


// 3. RoPE type
#define GGML_ROPE_TYPE_NORMAL 0

// 4. Element-wise ops (sin, cos, round, floor), implemented as CPU custom ops.
//    Requirements that the previous version violated:
//      * n_tasks must be GGML_N_TASKS_MAX or > 0 (ggml_map_custom1 GGML_ASSERTs on 0),
//      * the source may be a non-contiguous view/permute -> honour nb[] strides,
//      * the destination is a fresh contiguous tensor (ggml_dup_tensor).
namespace ggml_compat_impl {
enum class unary_kind : intptr_t { sin_op = 1, cos_op, round_op, floor_op };

static inline float unary_apply(unary_kind k, float x) {
    switch (k) {
        case unary_kind::sin_op:   return std::sin(x);
        case unary_kind::cos_op:   return std::cos(x);
        case unary_kind::round_op: return std::round(x);  // half away from zero, like stock ggml
        case unary_kind::floor_op: return std::floor(x);
    }
    return x;
}

static inline float load_f(const char * p, enum ggml_type t) {
    return t == GGML_TYPE_F16 ? ggml_fp16_to_fp32(*(const ggml_fp16_t *) p) : *(const float *) p;
}
static inline void store_f(char * p, enum ggml_type t, float v) {
    if (t == GGML_TYPE_F16) *(ggml_fp16_t *) p = ggml_fp32_to_fp16(v); else *(float *) p = v;
}

static void unary_kernel(struct ggml_tensor * dst, const struct ggml_tensor * src, int ith, int nth, void * ud) {
    const unary_kind k = static_cast<unary_kind>(reinterpret_cast<intptr_t>(ud));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == src->type);
    const int64_t nr = ggml_nrows(src);
    const int64_t rows_per_thread = (nr + nth - 1) / nth;
    const int64_t r0 = rows_per_thread * ith;
    const int64_t r1 = std::min<int64_t>(r0 + rows_per_thread, nr);
    for (int64_t r = r0; r < r1; ++r) {
        const int64_t i3 = r / (src->ne[2] * src->ne[1]);
        const int64_t i2 = (r - i3 * src->ne[2] * src->ne[1]) / src->ne[1];
        const int64_t i1 = r - i3 * src->ne[2] * src->ne[1] - i2 * src->ne[1];
        const char * sp = (const char *) src->data + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3];
        char *       dp = (char *) dst->data       + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3];
        for (int64_t i0 = 0; i0 < src->ne[0]; ++i0) {
            store_f(dp + i0 * dst->nb[0], dst->type, unary_apply(k, load_f(sp + i0 * src->nb[0], src->type)));
        }
    }
}

static inline struct ggml_tensor * unary_op(struct ggml_context * ctx, struct ggml_tensor * a, unary_kind k) {
    return ggml_map_custom1(ctx, a, unary_kernel, GGML_N_TASKS_MAX, reinterpret_cast<void *>(static_cast<intptr_t>(k)));
}
}  // namespace ggml_compat_impl

static inline struct ggml_tensor * ggml_sin_compat(struct ggml_context * ctx, struct ggml_tensor * a) {
    return ggml_compat_impl::unary_op(ctx, a, ggml_compat_impl::unary_kind::sin_op);
}
#define ggml_sin ggml_sin_compat

static inline struct ggml_tensor * ggml_cos_compat(struct ggml_context * ctx, struct ggml_tensor * a) {
    return ggml_compat_impl::unary_op(ctx, a, ggml_compat_impl::unary_kind::cos_op);
}
#define ggml_cos ggml_cos_compat

static inline struct ggml_tensor * ggml_round_compat(struct ggml_context * ctx, struct ggml_tensor * a) {
    return ggml_compat_impl::unary_op(ctx, a, ggml_compat_impl::unary_kind::round_op);
}
#define ggml_round ggml_round_compat

// 5. ggml_pad_ext(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
//    Stock semantics: zero pad `lp` elements before and `rp` after, per dimension.
//    ik_llama only has ggml_pad(), which pads at the END of each dim only (ne + p).
//    Left padding is therefore built from the end-padded tensor:
//        padded = pad(a, lp + rp)            = [ a | 0 x (lp+rp) ]
//        out    = concat( padded[ne .. ne+lp)  (lp zeros),
//                         padded[0  .. ne+rp)  (a followed by rp zeros) )
//    Negative values crop (view) before padding, matching what stock's kernel tolerates.
static inline struct ggml_tensor * ggml_pad_ext_compat(
    struct ggml_context * ctx, struct ggml_tensor * a,
    int lp0, int rp0, int lp1, int rp1, int lp2, int rp2, int lp3, int rp3) {
    const int lp[4] = { lp0, lp1, lp2, lp3 };
    const int rp[4] = { rp0, rp1, rp2, rp3 };
    struct ggml_tensor * cur = a;
    bool touched = false;
    for (int d = 0; d < 4; ++d) {
        const int64_t crop_l = lp[d] < 0 ? -lp[d] : 0;
        const int64_t crop_r = rp[d] < 0 ? -rp[d] : 0;
        const int64_t pad_l  = lp[d] > 0 ?  lp[d] : 0;
        const int64_t pad_r  = rp[d] > 0 ?  rp[d] : 0;
        if (crop_l + crop_r > 0) {
            int64_t ne[4] = { cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3] };
            GGML_ASSERT(ne[d] > crop_l + crop_r);
            ne[d] -= crop_l + crop_r;
            cur = ggml_view_4d(ctx, cur, ne[0], ne[1], ne[2], ne[3],
                               cur->nb[1], cur->nb[2], cur->nb[3], crop_l * cur->nb[d]);
            touched = true;
        }
        if (pad_l + pad_r > 0) {
            if (!ggml_is_contiguous(cur)) cur = ggml_cont(ctx, cur);
            int p[4] = { 0, 0, 0, 0 };
            p[d] = static_cast<int>(pad_l + pad_r);
            struct ggml_tensor * padded = ggml_pad(ctx, cur, p[0], p[1], p[2], p[3]);
            if (pad_l == 0) {
                cur = padded;  // pure end padding
            } else {
                int64_t zne[4] = { padded->ne[0], padded->ne[1], padded->ne[2], padded->ne[3] };
                int64_t bne[4] = { padded->ne[0], padded->ne[1], padded->ne[2], padded->ne[3] };
                zne[d] = pad_l;
                bne[d] = cur->ne[d] + pad_r;
                struct ggml_tensor * zeros = ggml_view_4d(ctx, padded, zne[0], zne[1], zne[2], zne[3],
                    padded->nb[1], padded->nb[2], padded->nb[3], cur->ne[d] * padded->nb[d]);
                struct ggml_tensor * body  = ggml_view_4d(ctx, padded, bne[0], bne[1], bne[2], bne[3],
                    padded->nb[1], padded->nb[2], padded->nb[3], 0);
                cur = ggml_concat(ctx, zeros, body, d);
            }
            touched = true;
        }
    }
    if (!touched) {
        return ggml_pad(ctx, a, 0, 0, 0, 0);  // stock returns a fresh tensor even for a no-op
    }
    return ggml_is_contiguous(cur) ? cur : ggml_cont(ctx, cur);
}
#define ggml_pad_ext ggml_pad_ext_compat

// 6. ggml_flash_attn_ext_with_bias_mask: NOT emulable on ik_llama (its FA requires mask->ne[2]==1,
//    the patched op takes a per-head bias). Call sites are routed to the matmul+soft_max_ext
//    fallback under AUDIOCPP_GGML_IK_LLAMA; if one is ever reached, fail loudly instead of
//    silently dropping the bias as the previous shim did.
static inline struct ggml_tensor * ggml_flash_attn_ext_with_bias_mask_compat(
    struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *,
    struct ggml_tensor *, struct ggml_tensor *, float, float, float) {
    GGML_ABORT("ggml_flash_attn_ext_with_bias_mask is unsupported on ik_llama; use the non-flash path");
    return nullptr;
}
#define ggml_flash_attn_ext_with_bias_mask ggml_flash_attn_ext_with_bias_mask_compat

// 7. gguf_get_tensor_size: intentionally NOT shimmed. ik_llama's gguf API exposes no tensor
//    shape, so any offset-difference emulation is wrong (last tensor => 0, alignment padding
//    included for the rest). Callers use ggml_nbytes() on the tensor from the ggml_context.

// 8. ggml_tallocr_alloc returns void in ik_llama
static inline enum ggml_status ggml_tallocr_alloc_compat(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    ggml_tallocr_alloc(talloc, tensor);
    return GGML_STATUS_SUCCESS;
}
#define ggml_tallocr_alloc ggml_tallocr_alloc_compat

// 9. ggml_backend_view_init returns void in ik_llama
static inline enum ggml_status ggml_backend_view_init_compat(struct ggml_tensor * tensor) {
    ggml_backend_view_init(tensor);
    return GGML_STATUS_SUCCESS;
}
#define ggml_backend_view_init ggml_backend_view_init_compat

// 10. ggml_graph_set_n_nodes
static inline void ggml_graph_set_n_nodes(struct ggml_cgraph * g, int n) { g->n_nodes = n; }

// 11. Backend device/reg API compatibility
// Forward declare types

typedef struct ggml_backend_reg_s { ggml_backend_t backend; size_t index; const char * name; } ggml_backend_reg_s;
typedef ggml_backend_reg_s * ggml_backend_reg_t;

typedef struct ggml_backend_dev_s {
    ggml_backend_t backend;
    size_t index;
    const char * name;
    const char * description;
    int type;
} ggml_backend_dev_s;
typedef ggml_backend_dev_s * ggml_backend_dev_t;

enum ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU = 0,
    GGML_BACKEND_DEVICE_TYPE_GPU = 1,
    GGML_BACKEND_DEVICE_TYPE_IGPU = 2,
    GGML_BACKEND_DEVICE_TYPE_ACCEL = 3,
    GGML_BACKEND_DEVICE_TYPE_META = 4,
};

typedef void (*ggml_backend_set_n_threads_t)(ggml_backend_t backend, int n_threads);

namespace ggml_compat_impl {
static std::vector<ggml_backend_reg_s> s_regs;
static std::vector<ggml_backend_dev_s> s_devs;
static bool s_initialized = false;

static void ensure_initialized() {
    if (s_initialized) return;
    s_initialized = true;
    size_t n = ggml_backend_reg_get_count();
    for (size_t i = 0; i < n; i++) {
        const char * name = ggml_backend_reg_get_name(i);
        if (!name) name = "";
        ggml_backend_dev_s dev;
        dev.backend = nullptr; dev.index = i; dev.name = name; dev.description = name;
        dev.type = (std::strcmp(name, "CPU") == 0) ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU;
        s_devs.push_back(dev);
        ggml_backend_reg_s reg; reg.backend = nullptr; reg.index = i; reg.name = name;
        s_regs.push_back(reg);
    }
}
}

static inline size_t ggml_backend_reg_count(void) {
    ggml_compat_impl::ensure_initialized();
    return ggml_compat_impl::s_regs.size();
}
static inline ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    ggml_compat_impl::ensure_initialized();
    if (index >= ggml_compat_impl::s_regs.size()) return nullptr;
    return &ggml_compat_impl::s_regs[index];
}
static inline const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    return reg ? reg->name : nullptr;
}
static inline size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    return reg ? 1 : 0;
}
static inline ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    ggml_compat_impl::ensure_initialized();
    if (!reg || index >= ggml_compat_impl::s_devs.size()) return nullptr;
    return &ggml_compat_impl::s_devs[reg->index];
}
static inline void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    (void)reg;
    if (!name) return nullptr;
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) return (void*)ggml_backend_cpu_set_n_threads;
    return nullptr;
}
static inline void ggml_backend_load_all(void) { ggml_compat_impl::ensure_initialized(); }
static inline ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    ggml_compat_impl::ensure_initialized();
    if (!backend) return nullptr;
    const char * bname = ggml_backend_name(backend);
    if (!bname) return nullptr;
    for (auto & dev : ggml_compat_impl::s_devs) {
        if (dev.name && std::strcmp(dev.name, bname) == 0) { dev.backend = backend; return &dev; }
    }
    return nullptr;
}
static inline const char * ggml_backend_dev_name(ggml_backend_dev_t d) { return d ? d->name : nullptr; }
static inline const char * ggml_backend_dev_description(ggml_backend_dev_t d) { return d ? d->description : nullptr; }
static inline void ggml_backend_dev_memory(ggml_backend_dev_t d, size_t * free, size_t * total) {
    (void)d; if (free) *free = 0; if (total) *total = 0;
}
static inline enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t d) {
    return d ? static_cast<enum ggml_backend_dev_type>(d->type) : GGML_BACKEND_DEVICE_TYPE_CPU;
}
static inline ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t d) {
    if (!d) return nullptr; ggml_compat_impl::ensure_initialized();
    if (d->index >= ggml_compat_impl::s_regs.size()) return nullptr;
    return &ggml_compat_impl::s_regs[d->index];
}
static inline ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t d, const char * params) {
    if (!d) return nullptr;
    return ggml_backend_reg_init_backend(d->index, params);
}
static inline ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    if (type == GGML_BACKEND_DEVICE_TYPE_CPU) return ggml_backend_cpu_init();
    return ggml_backend_reg_init_backend_from_str(params ? params : "");
}
static inline ggml_backend_t ggml_backend_init_best(void) {
    return ggml_backend_cpu_init();
}


// ggml_get_type_traits: ik_llama returns by value, audio.cpp expects pointer
static inline const ggml_type_traits_t * ggml_get_type_traits_compat(enum ggml_type type) {
    static thread_local ggml_type_traits_t cached;
    cached = ggml_internal_get_type_traits(type);
    return &cached;
}
#define ggml_get_type_traits ggml_get_type_traits_compat

// 12. Custom audio ops missing from ik_llama: ggml_conv_1d_dw, ggml_conv_1d_fast_1d_im2col, ggml_col2im_1d
// These are graph-building wrappers using ggml_im2col + ggml_mul_mat

static inline struct ggml_tensor * ggml_conv_1d_dw_compat(
    struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
    int s0, int p0, int d0) {
    struct ggml_tensor * new_b = ggml_reshape_4d(ctx, b, b->ne[0], 1, b->ne[1], b->ne[2]);
    struct ggml_tensor * im2col = ggml_im2col(ctx, a, new_b, s0, 0, p0, 0, d0, 0, false, a->type);
    struct ggml_tensor * result = ggml_mul_mat(ctx, im2col, a);
    return ggml_reshape_3d(ctx, result, result->ne[0], result->ne[2], 1);
}
#define ggml_conv_1d_dw ggml_conv_1d_dw_compat

static inline struct ggml_tensor * ggml_conv_1d_fast_1d_im2col_compat(
    struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b,
    int s0, int p0, int d0) {
    struct ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, 0, p0, 0, d0, 0, false, a->type);
    struct ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]);
}
#define ggml_conv_1d_fast_1d_im2col ggml_conv_1d_fast_1d_im2col_compat

static inline struct ggml_tensor * ggml_col2im_1d_compat(
    struct ggml_context *, struct ggml_tensor *, int, int, int) {
    // Only reachable via the CUDA/HIP ConvTranspose1d fast path, which is disabled for ik_llama
    // (see conv_modules.cpp). Never hand back an uninitialised tensor.
    GGML_ABORT("ggml_col2im_1d is not implemented for ik_llama");
    return nullptr;
}
#define ggml_col2im_1d ggml_col2im_1d_compat

// ggml_graph_plan: stock takes 3 args (cgraph, n_threads, buffer_type), ik takes 2 (cgraph, n_threads)
static inline struct ggml_cplan ggml_graph_plan_compat(const struct ggml_cgraph * g, int nt, void * buf_type) {
    (void)buf_type;
    return ggml_graph_plan(g, nt);
}
#define ggml_graph_plan ggml_graph_plan_compat


// 13. ggml_custom_4d: arbitrary-shape custom op with up to GGML_MAX_SRC args.
//     ik_llama's ggml_map_customN always produces dup(a)'s shape, so build the node by hand:
//     a new tensor of the requested shape, op = GGML_OP_MAP_CUSTOM1 (dispatch just calls the
//     fun stored in op_params[0..24)), all args in src[], and the caller's fun/userdata kept
//     in op_params bytes [32..48) and invoked through a trampoline.
namespace ggml_compat_impl {
typedef void (*custom_nd_fn)(struct ggml_tensor *, int, int, void *);
struct custom_nd_extra { custom_nd_fn fun; void * userdata; };
static constexpr size_t kCustomExtraOffset = 32;
static_assert(kCustomExtraOffset + sizeof(custom_nd_extra) <= GGML_MAX_OP_PARAMS, "op_params too small");

static void custom_nd_trampoline(struct ggml_tensor * dst, const struct ggml_tensor *, int ith, int nth, void *) {
    custom_nd_extra ex;
    std::memcpy(&ex, reinterpret_cast<const char *>(dst->op_params) + kCustomExtraOffset, sizeof(ex));
    ex.fun(dst, ith, nth, ex.userdata);
}

static inline struct ggml_tensor * make_custom_node(
    struct ggml_context * ctx, enum ggml_type type, const int64_t ne[4],
    struct ggml_tensor ** args, int n_args, custom_nd_fn fun, int n_tasks, void * userdata) {
    GGML_ASSERT(n_args >= 1 && n_args <= GGML_MAX_SRC);
    GGML_ASSERT(n_tasks == GGML_N_TASKS_MAX || n_tasks > 0);
    struct ggml_tensor * result = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
    struct { ggml_custom1_op_t fun; int n_tasks; void * userdata; } p = { custom_nd_trampoline, n_tasks, nullptr };
    std::memcpy(result->op_params, &p, sizeof(p));
    custom_nd_extra ex = { fun, userdata };
    std::memcpy(reinterpret_cast<char *>(result->op_params) + kCustomExtraOffset, &ex, sizeof(ex));
    result->op = GGML_OP_MAP_CUSTOM1;
    result->grad = nullptr;
    for (int i = 0; i < n_args; ++i) result->src[i] = args[i];
    return result;
}
}  // namespace ggml_compat_impl

static inline struct ggml_tensor * ggml_custom_4d_compat(
    struct ggml_context * ctx, enum ggml_type type,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    struct ggml_tensor ** args, int n_args,
    void (*fun)(struct ggml_tensor*, int, int, void*),
    int n_tasks, void * userdata) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    return ggml_compat_impl::make_custom_node(ctx, type, ne, args, n_args, fun, n_tasks, userdata);
}
#define ggml_custom_4d ggml_custom_4d_compat

// 14. ggml_pad_reflect_1d(ctx, a, p0, p1): reflect (no edge repeat) along dim 0, F32.
namespace ggml_compat_impl {
struct reflect_params { int p0; int p1; };
static void pad_reflect_kernel(struct ggml_tensor * dst, int ith, int nth, void * ud) {
    const reflect_params rp = *static_cast<const reflect_params *>(ud);
    const struct ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    const int64_t n = src->ne[0];
    const int64_t nr = ggml_nrows(src);
    for (int64_t r = ith; r < nr; r += nth) {
        const int64_t i3 = r / (src->ne[2] * src->ne[1]);
        const int64_t i2 = (r - i3 * src->ne[2] * src->ne[1]) / src->ne[1];
        const int64_t i1 = r - i3 * src->ne[2] * src->ne[1] - i2 * src->ne[1];
        const char * sp = (const char *) src->data + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3];
        char *       dp = (char *) dst->data       + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3];
        for (int64_t i = 0; i < dst->ne[0]; ++i) {
            int64_t idx = i - rp.p0;
            if (idx < 0) idx = -idx;
            if (idx >= n) idx = 2 * (n - 1) - idx;
            if (idx < 0) idx = 0;  // pad larger than signal: clamp rather than read OOB
            *(float *) (dp + i * dst->nb[0]) = *(const float *) (sp + idx * src->nb[0]);
        }
    }
}
}  // namespace ggml_compat_impl

static inline struct ggml_tensor * ggml_pad_reflect_1d_compat(
    struct ggml_context * ctx, struct ggml_tensor * a, int p0, int p1) {
    GGML_ASSERT(p0 >= 0 && p1 >= 0 && a->type == GGML_TYPE_F32);
    GGML_ASSERT(a->ne[0] > p0 && a->ne[0] > p1);
    // p0/p1 are packed into the userdata pointer value (no allocation, nothing to outlive).
    const int64_t ne[4] = { a->ne[0] + p0 + p1, a->ne[1], a->ne[2], a->ne[3] };
    struct ggml_tensor * args[1] = { a };
    struct ggml_tensor * r = ggml_compat_impl::make_custom_node(ctx, a->type, ne, args, 1,
        [](struct ggml_tensor * dst, int ith, int nth, void * ud) {
            const uintptr_t v = reinterpret_cast<uintptr_t>(ud);
            ggml_compat_impl::reflect_params rp = { int(v >> 32), int(v & 0xffffffffu) };
            ggml_compat_impl::pad_reflect_kernel(dst, ith, nth, &rp);
        },
        GGML_N_TASKS_MAX,
        reinterpret_cast<void *>((uintptr_t(uint32_t(p0)) << 32) | uintptr_t(uint32_t(p1))));
    return r;
}
#define ggml_pad_reflect_1d ggml_pad_reflect_1d_compat

// 15. ggml_mul_mat_pack4 -> ggml_mul_mat (lossy fallback)
#define ggml_mul_mat_pack4 ggml_mul_mat


// ggml_backend_tensor_copy: ik_llama takes non-const src, stock may take const
static inline void ggml_backend_tensor_copy_compat(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_tensor_copy(const_cast<struct ggml_tensor*>(src), dst);
}
#define ggml_backend_tensor_copy ggml_backend_tensor_copy_compat

static inline struct ggml_tensor * ggml_floor_compat(struct ggml_context * ctx, struct ggml_tensor * a) {
    return ggml_compat_impl::unary_op(ctx, a, ggml_compat_impl::unary_kind::floor_op);
}
#define ggml_floor ggml_floor_compat

// Disable IQK flash attention (use generic fallback) - IQK kernels crash on our shapes
static inline void ggml_flash_attn_disable_iqk(struct ggml_tensor * a) {
    if (a) a->op_params[4] = GGML_FLASH_ATTN_EXT_IQK_DISABLED;
}

// Manual flash attention replacement for ik_llama (its FA doesn't support F16 K/V)
// Input: q [HD, T, H], k [HD, T, H], v [HD, T, H], mask [T, T] (optional)
// Output: [HD, T, H] -> caller reshapes to [D, T]
static inline struct ggml_tensor * ggml_manual_attn_ext_impl(
    struct ggml_context * ctx,
    struct ggml_tensor * q,  // [head_dim, n_batch, n_heads]
    struct ggml_tensor * kk,  // [head_dim, n_kv, n_heads]
    struct ggml_tensor * v,  // [head_dim, n_kv, n_heads]
    struct ggml_tensor * mask,  // [n_kv, n_batch] or null
    float scale, float max_bias, float softcap) {
    // QK^T: [HD, T_q, H] @ [HD, T_k, H] -> need to compute per-head
    // For simplicity, compute scores as: (q^T @ k) / sqrt(d) per head
    // q: [HD, T_q, H], k: [HD, T_k, H]
    // Transpose k: [T_k, HD, H], then mul_mat q^T @ k^T... 
    // Actually, let's use the standard approach:
    // scores[i,j,h] = sum_d q[d,i,h] * k[d,j,h] * scale
    // This is a batched matmul. GGML can do it via:
    // transpose k -> [T_k, HD, H], then q @ k^T per head
    // But GGML doesn't have per-head batched matmul natively.
    // 
    // Simpler: use ggml_mul_mat with reshaping:
    // q: [HD, T_q, H] -> reshape to [HD, T_q*H] 
    // k: [HD, T_k, H] -> reshape to [HD, T_k*H]
    // mul_mat(q, k) = [T_q*H, T_k*H] - NOT what we want
    //
    // Correct approach: 
    // k_t = transpose(k) = [T_k, HD, H]
    // scores = mul_mat(q_t, k) where q_t = [T_q, HD, H]
    // = [T_q, T_k, H] - still 3D
    // Actually GGML mul_mat is 2D: [m, k] @ [k, n] = [m, n]
    // So reshape q to [HD, T_q*H], k to [HD, T_k*H]
    // mul_mat gives [T_q*H, T_k*H] which is wrong (cross-head mixing)
    //
    // The correct approach for per-head attention:
    // Use ggml_mul_mat per head via a loop or ggml_map_custom3
    // For now, use ggml_flash_attn_ext with a fallback:
    // If the types are F16, the only working approach is to use the 
    // stock ggml's flash attention. Since we can't mix forks,
    // implement a simple dense attention using mul_mat + softmax.
    
    (void)max_bias; (void)softcap;
    
    // k^T: [T_k, HD, H] -> [T_k, HD*H]
    // Actually, let's just do the standard approach:
    // scores = Q @ K^T / sqrt(d) where Q=[T,HD], K=[T_k,HD] per head
    // We need to handle heads. Reshape:
    // q: [HD, T_q, H] -> view as [T_q, HD*H] (transpose + reshape)
    // k: [HD, T_k, H] -> view as [T_k, HD*H]
    // This gives scores = q @ k^T = [T_q, T_k] but with cross-head mixing
    
    // For correct per-head attention we need to separate heads first.
    // The simplest working approach: use the einsum-like decomposition
    // that GGML supports via mul_mat with proper reshaping.
    
    // q: [HD, T_q, H], k: [HD, T_k, H]
    // Step 1: k_t = ggml_cont(ggml_transpose(ctx, kk)) = [T_k, HD, H]
    // Step 2: scores = ggml_mul_mat(q, k_t) = [T_k, T_q, H] NO - mul_mat is 2D
    // 
    // Let me use the correct GGML approach:
    // scores[i,j] = sum_d q[d,i,h] * k[d,j,h] for each head h
    // This is a batched matmul with batch=H.
    // GGML mul_mat with 3D tensors: if q=[HD,T_q,H] and k_t=[T_k,HD,H],
    // then mul_mat(q, k_t) = [T_k, T_q, H] - YES! GGML mul_mat handles 3D
    // as batched 2D matmul over the remaining dims.
    // 
    // Wait, GGML mul_mat is defined as: dst = src0^T @ src1 (for 2D)
    // For 3D: dst[b] = src0[b]^T @ src1[b]
    // So q^T @ k_t = [T_q, HD, H] @ [T_k, HD, H]
    // q^T = [HD, T_q, H], k_t = [T_k, HD, H]
    // mul_mat(q, k_t) = q^T @ k_t = [T_q, HD, H] @ [T_k, HD, H]
    // = [T_q, T_k, H] - YES!
    
    auto * k_t = ggml_cont(ctx, ggml_transpose(ctx, kk));  // [T_k, HD, H]
    auto * scores = ggml_mul_mat(ctx, q, k_t);  // [T_q, T_k, H]
    // Apply scale
    scores = ggml_scale(ctx, scores, scale);
    // Apply mask
    if (mask) {
        // mask: [T_k, T_q] or [T_k, mask_pad]
        // scores: [T_q, T_k, H]
        // Need to broadcast mask over heads: repeat mask to [T_q, T_k, H]
        auto * mask_rep = ggml_repeat(ctx, mask, ggml_reshape_3d(ctx, scores, scores->ne[0], scores->ne[1], scores->ne[2]));
        scores = ggml_add(ctx, scores, mask_rep);
    }
    // Softmax over T_k (dim 1)
    scores = ggml_soft_max(ctx, scores);
    // scores: [T_q, T_k, H], v: [HD, T_k, H]
    // output = v @ scores = [HD, T_k, H] @ [T_q, T_k, H]
    // v^T @ scores... no, we want sum_t scores[n,t,h] * v[d,t,h]
    // = v @ scores^T... 
    // scores^T = [T_k, T_q, H]
    // v @ scores^T = [HD, T_k, H] @ [T_k, T_q, H] = [HD, T_q, H]
    auto * scores_t = ggml_cont(ctx, ggml_transpose(ctx, scores));  // [T_k, T_q, H]
    auto * out = ggml_mul_mat(ctx, v, scores_t);  // [HD, T_q, H]
    return out;
}
#endif // AUDIOCPP_GGML_IK_LLAMA
