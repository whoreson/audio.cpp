// xtts_v2 vocoder -- HiFi-GAN + ResNet speaker encoder
// Ported from CrispASR xtts_voc.cpp for audio.cpp framework.
#include "engine/framework/core/ggml_compat.h"

#include "engine/community_models/xtts_v2/vocoder.h"
#include "engine/framework/core/backend.h"

#include "ggml-backend.h"
#ifndef AUDIOCPP_GGML_IK_LLAMA
#include "ggml-cpu.h"
#endif
#include "ggml.h"
#ifndef AUDIOCPP_GGML_IK_LLAMA
#include "gguf.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace engine::models::xtts_v2 {
namespace {

static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return (it != m.end()) ? it->second : nullptr;
}

static ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
    ggml_tensor* b, int stride, int padding, int dilation) {
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]));
    return y;
}

static ggml_tensor* conv_t1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
    ggml_tensor* b, int stride, int padding) {
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    if (padding > 0) {
        int64_t Tt = y->ne[0] - 2 * padding;
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, Tt, y->ne[1], y->nb[1], padding * ggml_element_size(y)));
    }
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]));
    return y;
}

static ggml_tensor* resblock(ggml_context* ctx, ggml_tensor* x,
    const std::map<std::string, ggml_tensor*>& ts, const std::string& p,
    int ks, const std::vector<int>& dil, float sl) {
    for (int d = 0; d < (int)dil.size(); d++) {
        ggml_tensor* r = x;
        x = ggml_leaky_relu(ctx, x, sl, false);
        x = conv1d(ctx, x, T(ts, p + ".c1." + std::to_string(d)),
                   T(ts, p + ".c1." + std::to_string(d) + ".bias"),
                   1, dil[d] * (ks - 1) / 2, dil[d]);
        x = ggml_leaky_relu(ctx, x, sl, false);
        x = conv1d(ctx, x, T(ts, p + ".c2." + std::to_string(d)),
                   T(ts, p + ".c2." + std::to_string(d) + ".bias"), 1, (ks - 1) / 2, 1);
        x = ggml_add(ctx, x, r);
    }
    return x;
}

static ggml_tensor* bn4d(ggml_context* ctx, ggml_tensor* x, const std::string& pfx) {
    // placeholder - actual BN inline
    return x;
}

// HiFi-GAN graph
static ggml_cgraph* build_hifigan(ggml_context* ctx0,
    const std::map<std::string, ggml_tensor*>& ts, int T_mel, bool use_spk, ggml_tensor* spk_in,
    bool trace) {
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, 1024);
    ggml_set_name(x, "voc_input"); ggml_set_input(x);
    x = conv1d(ctx0, x, T(ts, "hifigan_decoder.dec.conv_pre.weight"),
               T(ts, "hifigan_decoder.dec.conv_pre.bias"), 1, 3, 1);
    if (trace) { ggml_set_name(x, "voc_conv_pre"); ggml_set_output(x); ggml_build_forward_expand(gf, x); }
    if (use_spk) {
        auto* c = T(ts, "hifigan_decoder.dec.cond_layer.weight");
        auto* s = spk_in;
        auto* p = ggml_cont(ctx0, ggml_transpose(ctx0,
            ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, c, (int)c->ne[1], (int)c->ne[2]),
                         ggml_reshape_2d(ctx0, s, 512, 1))));
        if (auto* cb = T(ts, "hifigan_decoder.dec.cond_layer.bias"))
            p = ggml_add(ctx0, p, ggml_reshape_2d(ctx0, cb, 1, (int)ggml_nelements(cb)));
        x = ggml_add(ctx0, x, ggml_repeat(ctx0, p, x));
    }
    int rates[4] = {8,8,2,2}, ks[4] = {16,16,4,4}, rks[3] = {3,7,11};
    std::vector<int> dil = {1,3,5};
    for (int i = 0; i < 4; i++) {
        x = ggml_leaky_relu(ctx0, x, 0.1f, false);
        std::string u = "hifigan_decoder.dec.ups." + std::to_string(i);
        x = conv_t1d(ctx0, x, T(ts, u), T(ts, u + ".bias"), rates[i], (ks[i] - rates[i]) / 2);
        if (use_spk) {
            std::string cw = "hifigan_decoder.dec.conds." + std::to_string(i) + ".weight";
            auto* w = T(ts, cw);
            auto* s = spk_in;
            auto* p = ggml_cont(ctx0, ggml_transpose(ctx0,
                ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, w, (int)w->ne[1], (int)w->ne[2]),
                             ggml_reshape_2d(ctx0, s, 512, 1))));
            if (auto* cb = T(ts, cw + ".bias"))
                p = ggml_add(ctx0, p, ggml_reshape_2d(ctx0, cb, 1, (int)ggml_nelements(cb)));
            x = ggml_add(ctx0, x, ggml_repeat(ctx0, p, x));
        }
        ggml_tensor* rs = nullptr;
        for (int j = 0; j < 3; j++) {
            std::string rp = "hifigan_decoder.dec.resb." + std::to_string(i * 3 + j);
            auto* rb = resblock(ctx0, x, ts, rp, rks[j], dil, 0.1f);
            rs = rs ? ggml_add(ctx0, rs, rb) : rb;
        }
        x = ggml_scale(ctx0, rs, 1.0f / 3);
        if (trace) { char nm[32]; snprintf(nm, sizeof(nm), "voc_ups%d", i); ggml_set_name(x, nm); ggml_set_output(x); ggml_build_forward_expand(gf, x); }
    }
    x = ggml_leaky_relu(ctx0, x, 0.1f, false);
    x = conv1d(ctx0, x, T(ts, "hifigan_decoder.dec.conv_post.weight"),
               T(ts, "hifigan_decoder.dec.conv_post.bias"), 1, 3, 1);
    if (trace) { ggml_set_name(x, "voc_post"); ggml_set_output(x); ggml_build_forward_expand(gf, x); }
    x = ggml_tanh(ctx0, x);
    ggml_set_name(x, "voc_output"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// BN apply for conv2d: (x - rm) / sqrt(rv) * w + b
static ggml_tensor* apply_bn4d(ggml_context* ctx, ggml_tensor* x,
    ggml_tensor* rm, ggml_tensor* rv, ggml_tensor* w, ggml_tensor* b) {
    auto* rm4 = ggml_reshape_4d(ctx, rm, 1, 1, (int)x->ne[2], 1);
    auto* rv4 = ggml_reshape_4d(ctx, rv, 1, 1, (int)x->ne[2], 1);
    auto* w4 = ggml_reshape_4d(ctx, w, 1, 1, (int)x->ne[2], 1);
    auto* b4 = ggml_reshape_4d(ctx, b, 1, 1, (int)x->ne[2], 1);
    auto* m = ggml_repeat(ctx, rm4, x);
    // BN eps (1e-5) is pre-folded into .rv tensors at load time
    auto* v = ggml_repeat(ctx, ggml_sqrt(ctx, rv4), x);
    return ggml_add(ctx, ggml_mul(ctx, ggml_div(ctx, ggml_sub(ctx, x, m), v),
                   ggml_repeat(ctx, w4, x)), ggml_repeat(ctx, b4, x));
}

static ggml_tensor* se_blk(ggml_context* ctx, ggml_tensor* x, int C,
    ggml_tensor* f0w, ggml_tensor* f0b, ggml_tensor* f2w, ggml_tensor* f2b) {
    int64_t sp = x->ne[0] * x->ne[1];
    ggml_tensor* xr = ggml_reshape_3d(ctx, x, sp, C, 1);
    ggml_tensor* mn = ggml_scale(ctx, ggml_sum_rows(ctx, xr), 1.0f / sp);
    ggml_tensor* mT = ggml_cont(ctx, ggml_transpose(ctx, mn)); // [C,1,1]
    ggml_tensor* h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, f0w, mT), f0b));
    ggml_tensor* g = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, f2w, h), f2b));
    ggml_tensor* g4 = ggml_reshape_4d(ctx, g, 1, 1, C, 1);
    return ggml_mul(ctx, x, ggml_repeat(ctx, g4, x));
}

// ---- Host-side speaker encoder (bypasses ggml conv_2d which is broken on MinGW/ik_llama) ----

static std::vector<float> read_tensor_f32(ggml_tensor* t) {
    std::vector<float> v(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(v.data(), t->data, ggml_nbytes(t));
    } else {
        ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    }
    return v;
}

// conv2d host: input [W,H,C] (contiguous x[w + W*(h + H*c)]),
// weight [KW,KH,IC,OC] contiguous w[kw + KW*(kh + KH*(ic + KH*IC*oc))],
// output [OW,OH,OC] contiguous out[ow + OW*(oh + OH*oc)]
#if 0  // host-side speaker encoder (keep for debugging ggml conv_2d issues)
static void host_conv2d_simple(const float* __restrict inp, int W, int H, int C,
    const float* __restrict w, int KW, int KH, int IC, int OC,
    const float* __restrict bias,
    float* __restrict out, int s0, int s1, int p0, int p1) {
    int OW = (W + 2*p0 - KW)/s0 + 1;
    int OH = (H + 2*p1 - KH)/s1 + 1;
    memset(out, 0, OW*OH*OC*sizeof(float));
    // GGUF weight stride: oc_step = KW*KH*IC
    int oc_step = KW * KH * IC;
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int oc = 0; oc < OC; oc++) {
                float s = bias ? bias[oc] : 0.0f;
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        for (int ic = 0; ic < IC; ic++) {
                            int iw = ow*s0 + kw - p0;
                            int ih = oh*s1 + kh - p1;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                float ip = inp[iw + W*(ih + H*ic)];
                                float wp = w[kw + KW*(kh + KH*ic) + oc*oc_step];
                                s += ip * wp;
                            }
                        }
                out[ow + OW*(oh + OH*oc)] = s;
            }
}

// batchnorm2d: per-channel over [W,H], eps=1e-5 folded into rv
static void host_bn2d(float* __restrict x, int W, int H, int C,
    const float* __restrict rm, const float* __restrict rv,
    const float* __restrict wt, const float* __restrict bt) {
    for (int c = 0; c < C; c++) {
        float m = rm[c], v = rv[c], w = wt[c], b = bt[c];
        float inv = 1.0f / sqrtf(v + 1e-5f);
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
                float* px = x + w + W*(h + H*c);
                *px = (*px - m) * inv * w + b;
            }
    }
}

static void host_relu(float* x, int N) {
    for (int i = 0; i < N; i++) if (x[i] < 0) x[i] = 0.0f;
}

// Squeeze-and-Excitation block
static void host_se_blk(float* __restrict x, int W, int H, int C,
    const float* f0w, const float* f0b, int r,  // fc.0: [r, C]
    const float* f2w, const float* f2b) {       // fc.2: [C, r]
    // global avg pool -> [C]
    float mean[256];
    memset(mean, 0, C*sizeof(float));
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                mean[c] += x[w + W*(h + H*c)];
    for (int c = 0; c < C; c++) mean[c] /= (W*H);
    // fc.0: relu(r x C @ C) = relu(mean * f0w + f0b)
    float h[256];
    for (int i = 0; i < r; i++) {
        float s = f0b[i];
        for (int c = 0; c < C; c++) s += mean[c] * f0w[c + C*i];
        h[i] = s > 0 ? s : 0.0f;
    }
    // fc.2: sigmoid(C x r @ r) = sigmoid(h * f2w + f2b)
    float g[256];
    for (int c = 0; c < C; c++) {
        float s = f2b[c];
        for (int i = 0; i < r; i++) s += h[i] * f2w[i + r*c];
        g[c] = 1.0f / (1.0f + expf(-s));
    }
    // apply
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                x[w + W*(h + H*c)] *= g[c];
}
// se_basic block (host)
static std::vector<float> host_spk_enc(const float* mel, int Tm,
    const std::map<std::string, ggml_tensor*>& ts) {
    // mel layout: [ch * Tm + t]  ->  as 4D [W=Tm, H=64, C=1, N=1]: x[w + W*(h + H*c)]
    // But mel is channel-major: mel[ch*Tm + t]. Map to x[t + Tm*(ch + 64*0)] = mel[ch*Tm + t].
    // So x[w + Tm*h] = mel[h*Tm + w]. Already matches!

    // conv1: [3,3,1,32] F32, bias [32] F32
    auto c1w = read_tensor_f32(ts.at("hifigan_decoder.se.conv1.weight"));
    auto c1b = read_tensor_f32(ts.at("hifigan_decoder.se.conv1.bias"));
    std::vector<float> x(Tm*64);
    memcpy(x.data(), mel, Tm*64*sizeof(float)); // [W=Tm, H=64, C=1]

    // conv1 -> bias -> relu -> bn1
    std::vector<float> h(Tm*64*32);
    host_conv2d_simple(x.data(), Tm, 64, 1, c1w.data(), 3, 3, 1, 32, c1b.data(), h.data(), 1, 1, 1, 1);
    host_relu(h.data(), Tm*64*32);
    host_bn2d(h.data(), Tm, 64, 32,
        read_tensor_f32(ts.at("hifigan_decoder.se.bn1.rm")).data(),
        read_tensor_f32(ts.at("hifigan_decoder.se.bn1.rv")).data(),
        read_tensor_f32(ts.at("hifigan_decoder.se.bn1.weight")).data(),
        read_tensor_f32(ts.at("hifigan_decoder.se.bn1.bias")).data());

    // Layers: lf[4]={32,64,128,256}, lb[4]={3,4,6,3}
    int lf[4] = {32,64,128,256}, lb[4] = {3,4,6,3};
    int curW = Tm, curH = 64, curC = 32;
    std::vector<float> buf;

    for (int ly = 0; ly < 4; ly++) {
        bool first = (ly > 0); // downsample at first block of each layer after layer0
        for (int bl = 0; bl < lb[ly]; bl++) {
            bool ds = first && (bl == 0);
            std::string pfx = "hifigan_decoder.se.layer" + std::to_string(ly+1) + "." + std::to_string(bl);
            auto p_c1w = read_tensor_f32(ts.at(pfx + ".conv1.weight"));
            auto p_c2w = read_tensor_f32(ts.at(pfx + ".conv2.weight"));
            auto p_bn1_rm = read_tensor_f32(ts.at(pfx + ".bn1.rm"));
            auto p_bn1_rv = read_tensor_f32(ts.at(pfx + ".bn1.rv"));
            auto p_bn1_wt = read_tensor_f32(ts.at(pfx + ".bn1.weight"));
            auto p_bn1_bt = read_tensor_f32(ts.at(pfx + ".bn1.bias"));
            auto p_bn2_rm = read_tensor_f32(ts.at(pfx + ".bn2.rm"));
            auto p_bn2_rv = read_tensor_f32(ts.at(pfx + ".bn2.rv"));
            auto p_bn2_wt = read_tensor_f32(ts.at(pfx + ".bn2.weight"));
            auto p_bn2_bt = read_tensor_f32(ts.at(pfx + ".bn2.bias"));
            auto p_se0w = read_tensor_f32(ts.at(pfx + ".se.fc.0.weight"));
            auto p_se0b = read_tensor_f32(ts.at(pfx + ".se.fc.0.bias"));
            auto p_se2w = read_tensor_f32(ts.at(pfx + ".se.fc.2.weight"));
            auto p_se2b = read_tensor_f32(ts.at(pfx + ".se.fc.2.bias"));

            // SE reduction ratio r = C/8
            int r = curC / 8;

            std::vector<float> ds_w, ds_rm, ds_rv, ds_wt, ds_bt;
            if (ds) {
                ds_w = read_tensor_f32(ts.at(pfx + ".downsample.0.weight"));
                ds_rm = read_tensor_f32(ts.at(pfx + ".downsample.1.rm"));
                ds_rv = read_tensor_f32(ts.at(pfx + ".downsample.1.rv"));
                ds_wt = read_tensor_f32(ts.at(pfx + ".downsample.1.weight"));
                ds_bt = read_tensor_f32(ts.at(pfx + ".downsample.1.bias"));
            }

            int newW = ds ? (curW - 1)/2 + 1 : curW;
            int newH = ds ? (curH - 1)/2 + 1 : curH;
            int newC = lf[ly];

            std::vector<float> out(curW * curH * curC);
            memcpy(out.data(), h.data(), h.size()*sizeof(float));

            // conv1 -> relu -> bn1
            std::vector<float> h1(newW*newH*newC);
            host_conv2d_simple(out.data(), curW, curH, curC, p_c1w.data(), 3, 3, curC, newC, nullptr, h1.data(), ds?2:1, ds?2:1, 1, 1);
            host_relu(h1.data(), newW*newH*newC);
            host_bn2d(h1.data(), newW, newH, newC, p_bn1_rm.data(), p_bn1_rv.data(), p_bn1_wt.data(), p_bn1_bt.data());

            // conv2 -> bn2
            host_conv2d_simple(h1.data(), newW, newH, newC, p_c2w.data(), 3, 3, newC, newC, nullptr, h1.data(), 1, 1, 1, 1);
            host_bn2d(h1.data(), newW, newH, newC, p_bn2_rm.data(), p_bn2_rv.data(), p_bn2_wt.data(), p_bn2_bt.data());

            // se_blk
            host_se_blk(h1.data(), newW, newH, newC, p_se0w.data(), p_se0b.data(), r, p_se2w.data(), p_se2b.data());

            // residual
            if (ds) {
                // downsample: conv2d(2x2,stride2,no_bias) + bn
                std::vector<float> res(newW*newH*newC);
                host_conv2d_simple(out.data(), curW, curH, curC, ds_w.data(), 1, 1, curC, newC, nullptr, res.data(), 2, 2, 0, 0);
                host_bn2d(res.data(), newW, newH, newC, ds_rm.data(), ds_rv.data(), ds_wt.data(), ds_bt.data());
                h.resize(newW*newH*newC);
                for (size_t i = 0; i < res.size(); i++) res[i] += h1[i];
                memcpy(h.data(), res.data(), res.size()*sizeof(float));
            } else {
                h.resize(newW*newH*newC);
                for (size_t i = 0; i < h1.size(); i++) h1[i] += out[i];
                memcpy(h.data(), h1.data(), h1.size()*sizeof(float));
            }
            host_relu(h.data(), newW*newH*newC);

            curW = newW; curH = newH; curC = newC;
        }
    }

    // Now h has [W=1, H=1, C=256] -> spatial is 1x1 after 4 layers of downsampling
    // Actually: Tm=2941, H=64 -> layer0: 2941x64x32 -> layer1(ds): 1470x32x32 -> layer2(ds): 735x16x64
    // -> layer3(ds): 367x8x128 -> layer4(ds): 183x4x256. Not 1x1!
    // The reference flattens to [Ws, Cs*Hs] then does ASP.

    int Ws = curW, Hs = curH, Cs = curC;
    // Flatten to [Ws, Cs*Hs]: row r = c*Hs + h
    int CH = Cs * Hs;
    std::vector<float> xf(Ws * CH);
    for (int w = 0; w < Ws; w++)
        for (int c = 0; c < Cs; c++)
            for (int hh = 0; hh < Hs; hh++)
                xf[w + Ws*(c*Hs + hh)] = h[w + Ws*(hh + Hs*c)];

    // ASP: Conv1d(CH, 128, 1) -> relu -> BN1d -> Conv1d(128, CH, 1) -> softmax(time)
    auto asp0w = read_tensor_f32(ts.at("hifigan_decoder.se.attention.0.weight"));
    auto asp0b = read_tensor_f32(ts.at("hifigan_decoder.se.attention.0.bias"));
    auto asp2_rm = read_tensor_f32(ts.at("hifigan_decoder.se.attention.2.rm"));
    auto asp2_rv = read_tensor_f32(ts.at("hifigan_decoder.se.attention.2.rv"));
    auto asp2_wt = read_tensor_f32(ts.at("hifigan_decoder.se.attention.2.weight"));
    auto asp2_bt = read_tensor_f32(ts.at("hifigan_decoder.se.attention.2.bias"));
    auto asp3w = read_tensor_f32(ts.at("hifigan_decoder.se.attention.3.weight"));
    auto asp3b = read_tensor_f32(ts.at("hifigan_decoder.se.attention.3.bias"));

    // Conv1d: a1[w][ch] = sum_in xf[w][in] * asp0w[in + in_ch*ch] + asp0b[ch]
    // weight layout [in, out] (GGUF linear: ne[0]=in, ne[1]=out), data [out][in]
    int in_ch = CH, out_ch = 128;
    std::vector<float> a1(Ws * out_ch);
    memset(a1.data(), 0, a1.size()*sizeof(float));
    for (int w = 0; w < Ws; w++)
        for (int ch = 0; ch < out_ch; ch++)
            for (int in = 0; in < in_ch; in++)
                a1[w + Ws*ch] += xf[w + Ws*in] * asp0w[in + in_ch*ch];
    for (int ch = 0; ch < out_ch; ch++)
        for (int w = 0; w < Ws; w++)
            a1[w + Ws*ch] += asp0b[ch];
    // relu
    host_relu(a1.data(), Ws*out_ch);
    // BN1d: per channel over time
    for (int ch = 0; ch < out_ch; ch++) {
        float m = asp2_rm[ch], v = asp2_rv[ch], w2 = asp2_wt[ch], b2 = asp2_bt[ch];
        float inv = 1.0f / sqrtf(v + 1e-5f);
        for (int w = 0; w < Ws; w++) {
            float& v2 = a1[w + Ws*ch];
            v2 = (v2 - m) * inv * w2 + b2;
        }
    }
    // Conv1d: a2[w][ch] = sum_in a1[w][in] * asp3w[in + 128*ch] + asp3b[ch]
    std::vector<float> a2(Ws * CH);
    memset(a2.data(), 0, a2.size()*sizeof(float));
    for (int w = 0; w < Ws; w++)
        for (int ch = 0; ch < CH; ch++)
            for (int in = 0; in < 128; in++)
                a2[w + Ws*ch] += a1[w + Ws*in] * asp3w[in + 128*ch];
    for (int ch = 0; ch < CH; ch++)
        for (int w = 0; w < Ws; w++)
            a2[w + Ws*ch] += asp3b[ch];
    // softmax over time (per channel)
    for (int ch = 0; ch < CH; ch++) {
        float mx = -1e30f;
        for (int w = 0; w < Ws; w++) mx = std::max(mx, a2[w + Ws*ch]);
        float s = 0;
        for (int w = 0; w < Ws; w++) { a2[w + Ws*ch] = expf(a2[w + Ws*ch] - mx); s += a2[w + Ws*ch]; }
        for (int w = 0; w < Ws; w++) a2[w + Ws*ch] /= s;
    }

    // mu = sum_w xf[w] * a2[w], var = sum_w xf[w]^2 * a2[w] - mu^2
    std::vector<float> mu(CH), sg(CH);
    memset(mu.data(), 0, CH*sizeof(float));
    for (int ch = 0; ch < CH; ch++)
        for (int w = 0; w < Ws; w++)
            mu[ch] += xf[w + Ws*ch] * a2[w + Ws*ch];
    for (int ch = 0; ch < CH; ch++) {
        float s2 = 0;
        for (int w = 0; w < Ws; w++) s2 += xf[w + Ws*ch] * xf[w + Ws*ch] * a2[w + Ws*ch];
        sg[ch] = sqrtf(std::max(s2 - mu[ch]*mu[ch], 1e-5f));
    }

    // asp = [mu; sg] = [4096]
    std::vector<float> asp(2*CH);
    memcpy(asp.data(), mu.data(), CH*sizeof(float));
    memcpy(asp.data() + CH, sg.data(), CH*sizeof(float));

    // fc: [512, 4096] x [4096] + bias -> [512]
    auto fcw = read_tensor_f32(ts.at("hifigan_decoder.se.fc.weight"));
    std::vector<float> emb(512);
    memset(emb.data(), 0, 512*sizeof(float));
    for (int o = 0; o < 512; o++)
        for (int i = 0; i < 2*CH; i++)
            emb[o] += asp[i] * fcw[i + (2*CH)*o];
    auto fcb = read_tensor_f32(ts.at("hifigan_decoder.se.fc.bias"));
    for (int o = 0; o < 512; o++) emb[o] += fcb[o];

    // L2 normalize
    float nm = 0;
    for (int o = 0; o < 512; o++) nm += emb[o]*emb[o];
    nm = sqrtf(nm);
    for (int o = 0; o < 512; o++) emb[o] /= nm;

    return emb;
}


// Speaker encoder graph
#endif
static ggml_tensor* se_basic(ggml_context* ctx, ggml_tensor* x,
    const std::map<std::string, ggml_tensor*>& ts, const std::string& p,
    int C, bool ds) {
    ggml_tensor* res = x;
    // reference: conv1 -> relu -> bn1
    ggml_tensor* h = ggml_conv_2d(ctx, T(ts, p + ".conv1.weight"), x,
        ds ? 2 : 1, ds ? 2 : 1, 1, 1, 1, 1);
    h = ggml_relu(ctx, h);
    h = apply_bn4d(ctx, h, T(ts, p + ".bn1.rm"), T(ts, p + ".bn1.rv"),
                   T(ts, p + ".bn1.weight"), T(ts, p + ".bn1.bias"));
    h = ggml_conv_2d(ctx, T(ts, p + ".conv2.weight"), h, 1, 1, 1, 1, 1, 1);
    h = apply_bn4d(ctx, h, T(ts, p + ".bn2.rm"), T(ts, p + ".bn2.rv"),
                   T(ts, p + ".bn2.weight"), T(ts, p + ".bn2.bias"));
    h = se_blk(ctx, h, C, T(ts, p + ".se.fc.0.weight"), T(ts, p + ".se.fc.0.bias"),
               T(ts, p + ".se.fc.2.weight"), T(ts, p + ".se.fc.2.bias"));
    if (ds) {
        ggml_tensor* dc = ggml_conv_2d(ctx, T(ts, p + ".downsample.0.weight"), res, 2, 2, 0, 0, 1, 1);
        if (T(ts, p + ".downsample.1.weight")) {
            dc = apply_bn4d(ctx, dc, T(ts, p + ".downsample.1.rm"), T(ts, p + ".downsample.1.rv"),
                           T(ts, p + ".downsample.1.weight"), T(ts, p + ".downsample.1.bias"));
        }
        res = dc;
    }
    // reference: out += residual; out = relu(out)
    return ggml_relu(ctx, ggml_add(ctx, h, res));
}
static ggml_cgraph* build_spk_enc(ggml_context* ctx0,
    const std::map<std::string, ggml_tensor*>& ts, int T_mel) {
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    // GGML conv_2d expects feature maps as ne=[W(time), H(mels), C, N]
    // data layout: element (t, ch) at t + ch*T_mel -> matches mel[ch*Tm + t]
    ggml_tensor* x = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T_mel, 64, 1, 1);
    ggml_set_name(x, "speaker_mel"); ggml_set_input(x);
    // reference stem: conv1 -> relu -> bn1
    ggml_tensor* h = ggml_conv_2d(ctx0, T(ts, "hifigan_decoder.se.conv1.weight"), x, 1, 1, 1, 1, 1, 1);
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, ggml_reshape_4d(ctx0, T(ts, "hifigan_decoder.se.conv1.bias"), 1, 1, (int)h->ne[2], 1), h));
    if (getenv("XTTS_DUMP_SPK_NET")) { ggml_set_name(h, "spk_net_conv1"); ggml_set_output(h); ggml_build_forward_expand(gf, h); }
    h = ggml_relu(ctx0, h);
    h = apply_bn4d(ctx0, h, T(ts, "hifigan_decoder.se.bn1.rm"), T(ts, "hifigan_decoder.se.bn1.rv"),
                   T(ts, "hifigan_decoder.se.bn1.weight"), T(ts, "hifigan_decoder.se.bn1.bias"));
    if (getenv("XTTS_DUMP_SPK_NET")) { ggml_set_name(h, "spk_net_stem"); ggml_set_output(h); ggml_build_forward_expand(gf, h); }
    int lf[4] = {32,64,128,256}, lb[4] = {3,4,6,3};
    for (int ly = 0; ly < 4; ly++) {
        for (int bl = 0; bl < lb[ly]; bl++) {
            std::string pfx = "hifigan_decoder.se.layer" + std::to_string(ly + 1) + "." + std::to_string(bl);
            h = se_basic(ctx0, h, ts, pfx, lf[ly], (bl == 0 && ly > 0));
        }
        if (getenv("XTTS_DUMP_SPK_NET")) {
            char nm[64];
            snprintf(nm, sizeof(nm), "spk_net_layer%d", ly + 1);
            ggml_set_name(h, nm); ggml_set_output(h);
            ggml_build_forward_expand(gf, h);
        }
    }
    int Ws = (int)h->ne[0], Hs = (int)h->ne[1], Cs = (int)h->ne[2];
    // Flatten conv output to [Ws, Cs*Hs]: row r = c*Hs + h (PyTorch reshape(B, -1, T) order).
    // GGML element (w,h,c) at w + h*Ws + c*Ws*Hs == 2D view (w, r=c*Hs+h) at w + r*Ws. Exact match.
    ggml_tensor* xf = ggml_reshape_2d(ctx0, h, Ws, Cs * Hs);
    // ASP attention: Conv1d(2048,128,1) -> ReLU -> BN1d(128) -> Conv1d(128,2048,1) -> softmax(time)
    ggml_tensor* a1 = conv1d(ctx0, xf, T(ts, "hifigan_decoder.se.attention.0.weight"),
                             T(ts, "hifigan_decoder.se.attention.0.bias"), 1, 0, 1);
    a1 = ggml_relu(ctx0, a1);
    // BN1d over channels: a1 is [Ws, 128]; normalize per channel over time
    {
        auto* rm = T(ts, "hifigan_decoder.se.attention.2.rm");
        auto* rv = T(ts, "hifigan_decoder.se.attention.2.rv");
        auto* w  = T(ts, "hifigan_decoder.se.attention.2.weight");
        auto* b  = T(ts, "hifigan_decoder.se.attention.2.bias");
        auto* rm2 = ggml_reshape_2d(ctx0, rm, 1, (int)a1->ne[1]);
        auto* rv2 = ggml_reshape_2d(ctx0, rv, 1, (int)a1->ne[1]);
        auto* w2  = ggml_reshape_2d(ctx0, w, 1, (int)a1->ne[1]);
        auto* b2  = ggml_reshape_2d(ctx0, b, 1, (int)a1->ne[1]);
        auto* mn = ggml_repeat(ctx0, rm2, a1);
        auto* sd = ggml_repeat(ctx0, ggml_sqrt(ctx0, rv2), a1);
        a1 = ggml_add(ctx0, ggml_mul(ctx0, ggml_div(ctx0, ggml_sub(ctx0, a1, mn), sd), w2), b2);
    }
    ggml_tensor* a2 = conv1d(ctx0, a1, T(ts, "hifigan_decoder.se.attention.3.weight"),
                             T(ts, "hifigan_decoder.se.attention.3.bias"), 1, 0, 1);
    // softmax over time (ne[0]); w sums to 1 per channel row
    a2 = ggml_soft_max(ctx0, a2);
    // mu = sum_t x*w  [2048], sg = sqrt(clamp(sum_t x^2*w - mu^2))  [2048]
    ggml_tensor* xw = ggml_mul(ctx0, xf, a2);
    int CH = Cs * Hs;
    ggml_tensor* mu = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, xw), CH, 1);   // [2048, 1]
    ggml_tensor* x2w = ggml_mul(ctx0, ggml_mul(ctx0, xf, xf), a2);
    ggml_tensor* var = ggml_sub(ctx0, ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, x2w), CH, 1),
                                ggml_mul(ctx0, mu, mu));
    ggml_tensor* sg = ggml_sqrt(ctx0, ggml_clamp(ctx0, var, 1e-5f, 1e30f));
    ggml_tensor* asp = ggml_concat(ctx0, mu, sg, 0);                 // [4096, 1]
    ggml_tensor* emb = ggml_mul_mat(ctx0, T(ts, "hifigan_decoder.se.fc.weight"), asp);
    if (T(ts, "hifigan_decoder.se.fc.bias"))
        emb = ggml_add(ctx0, emb, T(ts, "hifigan_decoder.se.fc.bias"));
    // L2 normalize
    ggml_tensor* nm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, emb, emb)));
    emb = ggml_div(ctx0, emb, nm);
    ggml_set_name(emb, "speaker_embedding");
    ggml_build_forward_expand(gf, emb);
    return gf;
}

// GGUF weight loader
struct WL { ggml_context* ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
            std::map<std::string, ggml_tensor*> tensors; };

static bool load_gguf(const char* path, ggml_backend_t be, WL& wl) {
    gguf_init_params gp = {true, &wl.ctx};
    gguf_context* gu = gguf_init_from_file(path, gp);
    if (!gu || !wl.ctx) { if (gu) gguf_free(gu); return false; }
    auto buf = ggml_backend_alloc_ctx_tensors(wl.ctx, be);
    if (!buf) { gguf_free(gu); ggml_free(wl.ctx); wl.ctx = nullptr; return false; }
    wl.buf = buf;
    FILE* fp = fopen(path, "rb");
    if (!fp) { ggml_backend_buffer_free(buf); gguf_free(gu); ggml_free(wl.ctx); wl.ctx = nullptr; return false; }
    size_t doff = gguf_get_data_offset(gu);
    for (ggml_tensor* t = ggml_get_first_tensor(wl.ctx); t; t = ggml_get_next_tensor(wl.ctx, t)) {
        int64_t tid = gguf_find_tensor(gu, ggml_get_name(t));
        if (tid < 0) continue;
        size_t off = gguf_get_tensor_offset(gu, tid), nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tbuf(nbytes);
        if (fseek(fp, doff + off, SEEK_SET) != 0 || fread(tbuf.data(), 1, nbytes, fp) != nbytes) break;
        ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
    }
    fclose(fp); gguf_free(gu);
    // Fold BatchNorm eps (1e-5) into running_var tensors: PyTorch eval does sqrt(var + eps)
    for (ggml_tensor* t = ggml_get_first_tensor(wl.ctx); t; t = ggml_get_next_tensor(wl.ctx, t)) {
        std::string nm = ggml_get_name(t);
        if (nm.size() > 3 && nm.compare(nm.size() - 3, 3, ".rv") == 0 && t->type == GGML_TYPE_F32) {
            int64_t n = ggml_nelements(t);
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(float));
            for (int64_t i = 0; i < n; i++) tmp[i] += 1e-5f;
            ggml_backend_tensor_set(t, tmp.data(), 0, n * sizeof(float));
        }
    }
    for (ggml_tensor* t = ggml_get_first_tensor(wl.ctx); t; t = ggml_get_next_tensor(wl.ctx, t))
        wl.tensors[ggml_get_name(t)] = t;
    return true;
}

// Mel spectrogram (simple DFT-based, for conditioning only)
// Speaker-encoder mel: hamming window, centered frames, power mel, NO log
// (matches torchaudio MelSpectrogram with hamming_window + PreEmphasis(0.97) applied upstream)
static std::vector<float> compute_mel_hamming(const float* pcm, int ns, int sr, int nfft,
    int hop, int win, int nm, float fmin, float fmax) {
    // torch.hamming_window(win) default: PERIODIC (denominator win-1)
    std::vector<float> ham(win);
    for (int i = 0; i < win; i++) ham[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (win - 1));
    // torchaudio center=True, pad_mode="reflect": reflect padding of nfft/2 on both sides
    int pad = nfft / 2;
    std::vector<float> padded(2 * pad + ns, 0);
    for (int i = 1; i <= pad; i++) {
        int li = i, ri = i;
        while (li >= ns) li = 2 * (ns - 1) - li;
        while (ri >= ns) ri = 2 * (ns - 1) - ri;
        padded[pad - i] = pcm[li];
        padded[pad + ns - 1 + i] = pcm[ns - 1 - ri];
    }
    std::memcpy(padded.data() + pad, pcm, ns * sizeof(float));
    int T = (int)((padded.size() - nfft) / hop) + 1;
    if (T <= 0) return {};
    int nf = nfft / 2 + 1;
    auto mscl = [](float f){ return 1127.01048f * logf(1.0f + f / 700.0f); };
    float fmin_m = mscl(fmin), fmax_m = mscl(fmax);
    std::vector<float> mpts(nm + 2);
    for (int i = 0; i <= nm + 1; i++)
        mpts[i] = 700.0f * (expf((fmin_m + (fmax_m - fmin_m) * i / (nm + 1)) / 1127.01048f) - 1.0f);
    std::vector<float> fb(nm * nf, 0);
    for (int m = 0; m < nm; m++) for (int f = 0; f < nf; f++) {
        float freq = (float)sr * f / nfft;
        if (freq >= mpts[m] && freq <= mpts[m + 2])
            fb[m * nf + f] = freq <= mpts[m + 1] ?
                (freq - mpts[m]) / (mpts[m + 1] - mpts[m] + 1e-12f) :
                (mpts[m + 2] - freq) / (mpts[m + 2] - mpts[m + 1] + 1e-12f);
    }
    std::vector<float> mel(nm * T, 0);
    std::vector<float> frame(nfft);
    // torchaudio centers the (short) window inside the n_fft frame
    const int wpad = (nfft - win) / 2;
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < nfft; n++) {
            int wi = n - wpad;
            frame[n] = padded[(size_t)t * hop + n] * ((wi >= 0 && wi < win) ? ham[wi] : 0.0f);
        }
        // radix-2 FFT (nfft=512), iterative Cooley-Tukey: requires bit-reversed input order
        std::vector<float> re(frame), im(nfft, 0);
        for (int i = 1, j = 0; i < nfft; i++) {
            int bit = nfft >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
        }
        for (int len = 2; len <= nfft; len <<= 1) {
            int half = len >> 1;
            float wr = cosf(-2.0f * (float)M_PI / len), wi = sinf(-2.0f * (float)M_PI / len);
            for (int i = 0; i < nfft; i += len) {
                float cr = 1, ci = 0;
                for (int j = 0; j < half; j++) {
                    int a = i + j, b = i + j + half;
                    float tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - tr; im[b] = im[a] - ti;
                    re[a] += tr; im[a] += ti;
                    float nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
                }
            }
        }
        for (int m = 0; m < nm; m++) {
            float s = 0;
            for (int f = 0; f < nf; f++) s += fb[m * nf + f] * (re[f] * re[f] + im[f] * im[f]);
            mel[m * T + t] = s;
        }
    }
    return mel;
}

static std::vector<float> compute_mel(const float* pcm, int ns, int sr, int nfft,
    int hop, int win, int nm, float fmin, float fmax, bool log_flag) {
    std::vector<float> hann(win);
    for (int i = 0; i < win; i++) hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win));
    int nf = nfft / 2 + 1;
    auto mscl = [](float f){ return 1127.01048f * logf(1.0f + f / 700.0f); };
    float fmin_m = mscl(fmin), fmax_m = mscl(fmax);
    std::vector<float> mpts(nm + 2);
    for (int i = 0; i <= nm + 1; i++)
        mpts[i] = 700.0f * (expf((fmin_m + (fmax_m - fmin_m) * i / (nm + 1)) / 1127.01048f) - 1.0f);
    std::vector<float> fb(nm * nf, 0);
    for (int m = 0; m < nm; m++) for (int f = 0; f < nf; f++) {
        float freq = (float)sr * f / nfft;
        if (freq >= mpts[m] && freq <= mpts[m + 2])
            fb[m * nf + f] = freq <= mpts[m + 1] ?
                (freq - mpts[m]) / (mpts[m + 1] - mpts[m] + 1e-12f) :
                (mpts[m + 2] - freq) / (mpts[m + 2] - mpts[m + 1] + 1e-12f);
    }
    std::vector<float> padded(ns + nfft - 1, 0);
    std::memcpy(padded.data() + (nfft - 1) / 2, pcm, ns * sizeof(float));
    int T = (int)((padded.size() - nfft) / hop) + 1;
    if (T <= 0) return {};
    std::vector<float> mel(nm * T, 0);
    for (int t = 0; t < T; t++) {
        std::vector<float> re(nf, 0), im(nf, 0);
        for (int k = 0; k < nf; k++) for (int n = 0; n < nfft; n++) {
            float a = -2.0f * (float)M_PI * k * n / nfft;
            re[k] += padded[t * hop + n] * (n < win ? hann[n] : 0.0f) * cosf(a);
            im[k] += padded[t * hop + n] * (n < win ? hann[n] : 0.0f) * sinf(a);
        }
        for (int m = 0; m < nm; m++) {
            float s = 0;
            for (int f = 0; f < nf; f++) s += fb[m * nf + f] * (re[f] * re[f] + im[f] * im[f]);
            mel[m * T + t] = log_flag ? logf(s + 1e-7f) : s;
        }
    }
    return mel;
}

} // namespace

struct XttSv2Vocoder::Impl {
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;
    int n_threads = 4;
    ~Impl() {
        if (ctx_w) ggml_free(ctx_w);
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (backend_cpu) ggml_backend_free(backend_cpu);
    }
};

XttSv2Vocoder::~XttSv2Vocoder() {
    if (impl_) { delete impl_; impl_ = nullptr; }
}

bool XttSv2Vocoder::load(const std::filesystem::path& path, const core::BackendConfig& backend_config) {
    n_threads_ = backend_config.threads;
    if (impl_) { delete impl_; impl_ = nullptr; }
    impl_ = new Impl();
    auto& c = *impl_;
    c.backend_cpu = core::init_backend(backend_config);
    if (!c.backend_cpu) return false;
    core::set_backend_threads(c.backend_cpu, backend_config.threads);
    WL wl;
    if (!load_gguf(path.string().c_str(), c.backend_cpu, wl)) {
        fprintf(stderr, "xtts_v2: failed to load vocoder GGUF: %s\n", path.string().c_str());
        return false;
    }
    c.ctx_w = wl.ctx;
    c.tensors = std::move(wl.tensors);
    { gguf_context* g = gguf_init_from_file(path.string().c_str(), {true, 0});
      if (g) {
          int ki = gguf_find_key(g, "xtts.mel_stats");
          if (ki >= 0 && gguf_get_kv_type(g, ki) == GGUF_TYPE_ARRAY) {
              int n = gguf_get_arr_n(g, ki);
              const float* d = static_cast<const float*>(gguf_get_arr_data(g, ki));
              mel_stats_.assign(d, d + n);
          }
          gguf_free(g);
      }
    }
    fprintf(stderr, "xtts_v2: vocoder loaded from %s (%zu tensors)\n",
            path.string().c_str(), c.tensors.size());
    return true;
}

std::vector<float> XttSv2Vocoder::generate(const float* latent, int T, const float* spk_emb) const {
    if (!impl_ || !latent || T <= 0) return {};
    auto& c = *impl_;
    static int voc_call = 0;
    auto path_for = [](const char* base) {
        static char buf[512];
        snprintf(buf, sizeof(buf), "%s.%d", base, voc_call);
        return buf;
    };
    // Reference HifiDecoder.forward: TWO chained linear interpolates (align_corners=False):
    //   Stage 1) scale_factor = ar_mel_length_compression / output_hop_length = 1024/256 = 4
    //   Stage 2) scale_factor = output_sample_rate / input_sample_rate = 24000/22050
    // PyTorch align_corners=False: output len = floor(input * scale),
    //   src_pos = (dst + 0.5) / scale - 0.5
    // These must NOT be fused into one: the intermediate discrete length matters.
    const double s1 = 4.0;
    const double s2 = 24000.0 / 22050.0;
    int T1 = (int)(T * s1);
    std::vector<float> stage1(1024 * T1, 0);
    for (int ch = 0; ch < 1024; ch++)
        for (int to = 0; to < T1; to++) {
            float pos = (float)((to + 0.5) / s1 - 0.5);
            if (pos < 0) pos = 0;
            if (pos > T - 1) pos = T - 1;
            int t0 = (int)pos, t1 = std::min(t0 + 1, T - 1);
            float f = pos - t0;
            stage1[(size_t)ch * T1 + to] = latent[(size_t)ch * T + t0] * (1.0f - f) + latent[(size_t)ch * T + t1] * f;
        }
    int Tu = (int)(T1 * s2);
#if defined(__x86_64__) || defined(_M_X64)
    if (getenv("XTTS_DEBUG")) {
        unsigned mxcsr = _mm_getcsr();
        fprintf(stderr, "xtts_dbg: vocoder mxcsr=0x%08x (FTZ=%d DAZ=%d)\n",
                mxcsr, (mxcsr >> 15) & 1, (mxcsr >> 6) & 1);
    }
#endif
    std::vector<float> ups;
    // Trace hook: inject the interpolated vocoder input latents [1024 x Tu].
    // The file is authoritative: Tu is derived from its size, so a dump produced
    // with a different (stochastic) mel code count still replays exactly.
    bool injected = false;
    if (const char* p = getenv("XTTS_INJECT_VOC_LAT")) {
        FILE* f = fopen(p, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            if (sz >= (long)(1024 * sizeof(float)) && sz % (long)(1024 * sizeof(float)) == 0) {
                Tu = (int)(sz / (1024 * (long)sizeof(float)));
                fprintf(stderr, "xtts_dbg: injected vocoder latents (%d frames, from file)\n", Tu);
                injected = true;
                ups.resize((size_t)1024 * Tu);
                if (fread(ups.data(), 4, ups.size(), f) != ups.size()) {
                    fprintf(stderr, "xtts_dbg: XTTS_INJECT_VOC_LAT short read\n");
                    return {};
                }
            } else {
                fprintf(stderr, "xtts_dbg: XTTS_INJECT_VOC_LAT bad size: file=%ld (not multiple of 4096)\n", sz);
            }
            fclose(f);
        } else fprintf(stderr, "xtts_dbg: XTTS_INJECT_VOC_LAT could not open %s\n", p);
    }
    if (!injected) {
        ups.assign((size_t)1024 * Tu, 0.f);
        for (int ch = 0; ch < 1024; ch++)
            for (int to = 0; to < Tu; to++) {
                float pos = (float)((to + 0.5) / s2 - 0.5);
                if (pos < 0) pos = 0;
                if (pos > T1 - 1) pos = T1 - 1;
                int t0 = (int)pos, t1 = std::min(t0 + 1, T1 - 1);
                float f = pos - t0;
                ups[(size_t)ch * Tu + to] = stage1[(size_t)ch * T1 + t0] * (1.0f - f) + stage1[(size_t)ch * T1 + t1] * f;
            }
        if (const char* p = getenv("XTTS_DUMP_VOC_LAT")) {
            const char* dp = path_for(p);
            FILE* f = fopen(dp, "wb");
            if (f) { fwrite(ups.data(), 4, ups.size(), f); fclose(f);
                     fprintf(stderr, "xtts_dbg: dumped vocoder latents to %s (%d frames)\n", dp, Tu); }
        }
    }
    // Trace hook: dump or inject the speaker embedding [512]
    static thread_local std::vector<float> spk_trace;
    if (spk_emb) {
        if (const char* p = getenv("XTTS_INJECT_VOC_SPK")) {
            FILE* f = fopen(p, "rb");
            if (f) { spk_trace.resize(512);
                     if (fread(spk_trace.data(), 4, 512, f) == 512)
                         fprintf(stderr, "xtts_dbg: injected speaker embedding\n");
                     fclose(f); }
        } else {
            if (const char* p = getenv("XTTS_DUMP_VOC_SPK")) {
                const char* dp = path_for(p);
                FILE* f = fopen(dp, "wb");
                if (f) { fwrite(spk_emb, 4, 512, f); fclose(f);
                         fprintf(stderr, "xtts_dbg: dumped speaker embedding to %s\n", dp); }
            }
        }
    }
    size_t need = 128 * 1024 * 1024;
    if (c.compute_meta.size() < need) c.compute_meta.resize(need);
    ggml_context* ctx0 = ggml_init({need, c.compute_meta.data(), true});
    // Single shared speaker_emb input tensor (used by cond_layer + 4 conds.N blocks)
    ggml_tensor* spk_t = nullptr;
    if (spk_emb) {
        spk_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 512);
        ggml_set_name(spk_t, "speaker_emb"); ggml_set_input(spk_t);
    }
    const bool voc_net_trace = getenv("XTTS_DUMP_VOC_NET") != nullptr;
    ggml_cgraph* gf = build_hifigan(ctx0, c.tensors, Tu, spk_emb != nullptr, spk_t, voc_net_trace);
    // Allocate input tensors (tracked for cleanup)
    ggml_backend_buffer_t vin_buf = nullptr;
    ggml_backend_buffer_t spk_buf = nullptr;
    { ggml_tensor* vin = ggml_graph_get_tensor(gf, "voc_input");
      vin_buf = ggml_backend_alloc_buffer(c.backend_cpu, ggml_nbytes(vin));
      vin->buffer = vin_buf;
      vin->data = (void*)ggml_backend_buffer_get_base(vin_buf);
      ggml_backend_tensor_set(vin, ups.data(), 0, ggml_nbytes(vin)); }
    if (spk_t) {
        spk_buf = ggml_backend_alloc_buffer(c.backend_cpu, ggml_nbytes(spk_t));
        spk_t->buffer = spk_buf;
        spk_t->data = (void*)ggml_backend_buffer_get_base(spk_buf);
        const float* spk_data = spk_emb;
        if (spk_trace.size() == 512) spk_data = spk_trace.data();
        ggml_backend_tensor_set(spk_t, spk_data, 0, 512 * sizeof(float));
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c.backend_cpu));
    if (!ga || !ggml_gallocr_reserve(ga, gf)) {
        ggml_backend_buffer_free(vin_buf); ggml_backend_buffer_free(spk_buf);
        if (ga) ggml_gallocr_free(ga); ggml_free(ctx0); return {}; }
    if (!ggml_gallocr_alloc_graph(ga, gf) ||
        ggml_backend_graph_compute(c.backend_cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(vin_buf); ggml_backend_buffer_free(spk_buf);
        ggml_gallocr_free(ga); ggml_free(ctx0); return {};
    }
    ggml_tensor* out = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    int64_t Ta = out->ne[0];
    if (Ta <= 0) {
        ggml_backend_buffer_free(vin_buf); ggml_backend_buffer_free(spk_buf);
        ggml_gallocr_free(ga); ggml_free(ctx0); return {}; }
    std::vector<float> result(Ta);
    ggml_backend_tensor_get(out, result.data(), 0, Ta * sizeof(float));
    // Trace hook: dump vocoder output floats
    if (const char* p = getenv("XTTS_DUMP_VOC_OUT")) {
        const char* dp = path_for(p);
        FILE* f = fopen(dp, "wb");
        if (f) { fwrite(result.data(), 4, result.size(), f); fclose(f);
                 fprintf(stderr, "xtts_dbg: dumped vocoder output to %s (%zd samples)\n", dp, result.size()); }
    }
    ggml_backend_buffer_free(vin_buf);
    ggml_backend_buffer_free(spk_buf);
    ggml_gallocr_free(ga); ggml_free(ctx0);
    voc_call++;
    return result;
}

std::vector<float> XttSv2Vocoder::speaker_embed(const float* ref_pcm, int n_samples) const {
    if (!impl_ || !ref_pcm || n_samples <= 0) return {};
    auto& c = *impl_;
    // Resample 22050->16000
    std::vector<float> pcm16;
    { double r = 22050.0 / 16000.0; int ol = (int)std::round(n_samples / r);
      pcm16.resize(ol);
      for (int i = 0; i < ol; i++) {
          double s = i * r; int idx = (int)s; double f = s - idx;
          if (idx + 1 < n_samples) pcm16[i] = (float)((1.0 - f) * ref_pcm[idx] + f * ref_pcm[idx + 1]);
          else if (idx < n_samples) pcm16[i] = ref_pcm[idx];
      }
    }
    // Reference torch_spec: PreEmphasis(0.97) -> MelSpectrogram(16k, n_fft=512, win=400,
    // hop=160, hamming, n_mels=64, power=2.0), then ResNetSpeakerEncoder log_input=True:
    // x = (x + 1e-6).log(), then InstanceNorm1d(64)
    std::vector<float> pre(pcm16.size());
    for (int i = (int)pcm16.size() - 1; i > 0; i--) pre[i] = pcm16[i] - 0.97f * pcm16[i - 1];
    if (!pcm16.empty()) pre[0] = pcm16[0];
    if (getenv("XTTS_DUMP_SPK_PRE")) {
        FILE* f = fopen(getenv("XTTS_DUMP_SPK_PRE"), "wb");
        if (f) { fwrite(pre.data(), 4, pre.size(), f); fclose(f); }
    }
    auto ts0 = std::chrono::steady_clock::now();
    auto mel = compute_mel_hamming(pre.data(), (int)pre.size(), 16000, 512, 160, 400, 64, 0, 8000);
    auto ts1 = std::chrono::steady_clock::now();
    if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing:   spk mel=%.2f s\n", std::chrono::duration<double>(ts1 - ts0).count());
    if (mel.empty()) return {};
    for (auto& v : mel) v = logf(v + 1e-6f); // LOGFIX
    if (getenv("XTTS_INJECT_SPK_MEL")) {
        // bypass: load reference log-mel (64 x T, channel-major) for numerical debugging
        FILE* f = fopen(getenv("XTTS_INJECT_SPK_MEL"), "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            std::vector<float> refmel(sz / 4);
            if (fread(refmel.data(), 4, refmel.size(), f) == refmel.size()) mel = std::move(refmel);
            fclose(f);
        }
    }
    if (getenv("XTTS_DUMP_SPK_MEL")) {
        FILE* f = fopen(getenv("XTTS_DUMP_SPK_MEL"), "wb");
        if (f) { fwrite(mel.data(), 4, mel.size(), f); fclose(f); }
    }
    int Tm = (int)mel.size() / 64;
    // InstanceNorm1d(64): per-channel zero-mean unit-variance over time
    for (int ch = 0; ch < 64; ch++) {
        float* row = mel.data() + (size_t)ch * Tm;
        double mean = 0; for (int t = 0; t < Tm; t++) mean += row[t];
        mean /= Tm;
        double var = 0; for (int t = 0; t < Tm; t++) var += (row[t] - mean) * (row[t] - mean);
        var /= Tm;
        float inv = (float)(1.0 / sqrt(var + 1e-5));
        for (int t = 0; t < Tm; t++) row[t] = (row[t] - (float)mean) * inv;
    }

    // ggml graph path for speaker encoder
    size_t need = 64 * 1024 * 1024;
    if (c.compute_meta.size() < need) c.compute_meta.resize(need);
    ggml_context* ctx0 = ggml_init({need, c.compute_meta.data(), true});
    ggml_cgraph* gf = build_spk_enc(ctx0, c.tensors, Tm);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c.backend_cpu));
    if (!ga || !ggml_gallocr_reserve(ga, gf)) { if (ga) ggml_gallocr_free(ga); ggml_free(ctx0); return {}; }
    if (!ggml_gallocr_alloc_graph(ga, gf)) { ggml_gallocr_free(ga); ggml_free(ctx0); return {}; }
    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "speaker_mel");
    ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));
    auto ts2 = std::chrono::steady_clock::now();
    if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing:   spk graph build+set=%.2f s\n", std::chrono::duration<double>(ts2 - ts1).count());
    auto ts3 = std::chrono::steady_clock::now();
    if (getenv("XTTS_GRAPH_PRINT")) ggml_graph_print(gf);
    if (ggml_backend_graph_compute(c.backend_cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(ga); ggml_free(ctx0); return {};
    }
    auto ts4 = std::chrono::steady_clock::now();
    if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing:   spk graph compute=%.2f s\n", std::chrono::duration<double>(ts4 - ts3).count());
    ggml_tensor* out = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    std::vector<float> result(512);
    ggml_backend_tensor_get(out, result.data(), 0, 512 * sizeof(float));
    ggml_gallocr_free(ga); ggml_free(ctx0);
    return result;
}

} // namespace engine::models::xtts_v2

