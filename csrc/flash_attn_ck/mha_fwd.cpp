/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

#include "fmha_fwd.hpp"
#include "mask.hpp"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <iostream>

namespace {

inline bool ck_tile_is_rdna_arch(const std::string& arch)
{
    return arch.rfind("gfx10", 0) == 0 || arch.rfind("gfx11", 0) == 0 ||
           arch.rfind("gfx12", 0) == 0;
}

inline std::string ck_tile_trim_gfx_arch(const char* arch_name)
{
    if(arch_name == nullptr)
        return {};
    std::string arch = arch_name;
    const auto pos   = arch.find(':');
    if(pos != std::string::npos)
        arch = arch.substr(0, pos);
    return arch;
}

inline size_t ck_tile_get_llc_cache_bytes(const std::string& arch)
{
    // Environment override (in MB). Prefer CK_TILE_*, fallback to Triton env.
    const char* env_llc_mb = std::getenv("CK_TILE_FMHA_LLC_CACHE_MB");
    if(env_llc_mb == nullptr)
        env_llc_mb = std::getenv("FLASH_ATTN_LLC_CACHE_MB");
    if(env_llc_mb == nullptr)
        env_llc_mb = std::getenv("FLASH_ATTN_L2_CACHE_MB"); // legacy alias in triton
    if(env_llc_mb != nullptr)
    {
        const int mb = std::atoi(env_llc_mb);
        if(mb > 0)
            return static_cast<size_t>(mb) * 1024ull * 1024ull;
    }

    // Known sizes (bytes)
    if(arch == "gfx1030")
        return 128ull * 1024ull * 1024ull;
    if(arch == "gfx1100")
        return 96ull * 1024ull * 1024ull;
    if(arch == "gfx1101")
        return 64ull * 1024ull * 1024ull;
    if(arch == "gfx1102")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1150")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1151")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1200")
        return 32ull * 1024ull * 1024ull;
    if(arch == "gfx1201")
        return 64ull * 1024ull * 1024ull;

    // Reasonable defaults by family
    if(arch.rfind("gfx10", 0) == 0)
        return 128ull * 1024ull * 1024ull;
    if(arch.rfind("gfx11", 0) == 0)
        return 96ull * 1024ull * 1024ull;
    if(arch.rfind("gfx12", 0) == 0)
        return 64ull * 1024ull * 1024ull;

    return 0; // unknown
}

inline bool ck_tile_head_group_log_enabled()
{
    const char* env = std::getenv("CK_TILE_FMHA_HEAD_GROUP_LOG");
    return env != nullptr && std::atoi(env) == 1;
}

inline bool ck_tile_head_grouping_disabled_by_env()
{
    const char* env_disable = std::getenv("CK_TILE_FMHA_DISABLE_HEAD_GROUPING");
    if(env_disable != nullptr && std::atoi(env_disable) == 1)
        return true;
    env_disable = std::getenv("FLASH_ATTN_DISABLE_HEAD_GROUPING");
    if(env_disable != nullptr && std::atoi(env_disable) == 1)
        return true;
    return false;
}

inline std::optional<ck_tile::index_t> ck_tile_get_head_group_size(
    ck_tile::index_t nhead_q,
    ck_tile::index_t nhead_k,
    ck_tile::index_t batch,
    ck_tile::index_t seqlen_k,
    ck_tile::index_t hdim_q,
    ck_tile::index_t hdim_v,
    size_t elem_bytes_k,
    size_t elem_bytes_v)
{
    if(ck_tile_head_grouping_disabled_by_env())
        return std::nullopt;

    // Manual override
    const char* env_group = std::getenv("CK_TILE_FMHA_HEAD_GROUP_SIZE");
    if(env_group != nullptr)
    {
        const int g = std::atoi(env_group);
        if(g > 0)
        {
            if(nhead_k <= 0 || nhead_q <= 0 || (nhead_q % nhead_k) != 0)
                return std::nullopt;
            const ck_tile::index_t gqa_ratio = nhead_q / nhead_k;
            ck_tile::index_t forced_group    = static_cast<ck_tile::index_t>(std::min<int>(g, nhead_q));
            if(gqa_ratio > 1)
            {
                const ck_tile::index_t min_group_aligned =
                    ((forced_group + gqa_ratio - 1) / gqa_ratio) * gqa_ratio;
                forced_group = min_group_aligned;
            }
            forced_group = std::min(forced_group, nhead_q);
            if(forced_group >= nhead_q)
                return std::nullopt;
            return forced_group;
        }
    }

    if(batch <= 0)
        return std::nullopt;

    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return std::nullopt;
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return std::nullopt;

    const std::string arch = ck_tile_trim_gfx_arch(props.gcnArchName);
    if(!ck_tile_is_rdna_arch(arch))
        return std::nullopt;

    const size_t llc_bytes = ck_tile_get_llc_cache_bytes(arch);
    if(llc_bytes == 0)
        return std::nullopt;

    if(nhead_k <= 0 || nhead_q <= 0 || (nhead_q % nhead_k) != 0)
        return std::nullopt;

    if(seqlen_k <= 0 || hdim_q <= 0 || hdim_v <= 0)
        return std::nullopt;

    const size_t kv_bytes_per_head =
        static_cast<size_t>(seqlen_k) *
        (static_cast<size_t>(hdim_q) * elem_bytes_k + static_cast<size_t>(hdim_v) * elem_bytes_v);
    if(kv_bytes_per_head == 0)
        return std::nullopt;

    const size_t bytes_per_group = kv_bytes_per_head * static_cast<size_t>(batch);
    if(bytes_per_group == 0)
        return std::nullopt;

    const size_t target_bytes = static_cast<size_t>(llc_bytes * 8 / 10); // 80% LLC budget
    ck_tile::index_t group     = static_cast<ck_tile::index_t>(target_bytes / bytes_per_group);
    if(group < 1)
        group = 1;

    const ck_tile::index_t gqa_ratio = nhead_q / nhead_k;
    if(gqa_ratio > 1)
    {
        const ck_tile::index_t min_group_aligned =
            ((group + gqa_ratio - 1) / gqa_ratio) * gqa_ratio;
        group = min_group_aligned;
    }

    group = std::min(group, nhead_q);
    if(group >= nhead_q)
        return std::nullopt;

    return group;
}

} // namespace

fmha_fwd_traits get_ck_fmha_fwd_traits(const mask_info &mask,
                                       std::string dtype,
                                       int head_size,
                                       bool has_dropout,
                                       bool has_lse,
                                       bool enable_alibi)
{
    return fmha_fwd_traits{head_size,
                           head_size,
                           dtype,
                           false, // is_group_mode
                           true,  // is_v_rowmajor
                           false, // has_logits_soft_cap
                           mask.type,
                           enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
                           has_lse,
                           has_dropout,
                           quant_scale_enum::no_scale}; // qscale_type
}

fmha_fwd_args get_ck_fmha_fwd_args(bool has_lse,
                                   bool has_dropout_randval,
                                   const mask_info &mask,
                                   // sizes
                                   const int b,
                                   const int seqlen_q,
                                   const int seqlen_k,
                                   const int h,
                                   const int h_k,
                                   const int d,
                                   // device pointers
                                   const at::Tensor q,
                                   const at::Tensor k,
                                   const at::Tensor v,
                                   std::optional<at::Tensor> &alibi_slopes_,
                                   at::Tensor out,
                                   at::Tensor softmax_lse,
                                   at::Tensor dropout_randval,
                                   float softmax_scale,
                                   float p_dropout,
                                   std::pair<uint64_t*, uint64_t*> drop_seed_offset)
{
    // q: (batch_size, seqlen_q, nheads, d)
    // k: (batch_size, seqlen_k, nheads_k, d)
    // v: (batch_size, seqlen_k, nheads_k, d)
    // o: (batch_size, seqlen_q, nheads, d)

    // alibi_slopes:(batch_size, nheads) or (nhead)
    // lse: (batch_size, nheads, seqlen_q)
    // randval: (batch_size, nheads, seqlen_q, seqlen_k)

    ck_tile::index_t stride_q = q.stride(1);
    ck_tile::index_t stride_k = k.stride(1);
    ck_tile::index_t stride_v = v.stride(1);
    ck_tile::index_t stride_o = out.stride(1);
    ck_tile::index_t stride_randval = has_dropout_randval ? dropout_randval.stride(2) : 0;

    ck_tile::index_t nhead_stride_q = q.stride(2);
    ck_tile::index_t nhead_stride_k = k.stride(2);
    ck_tile::index_t nhead_stride_v = v.stride(2);
    ck_tile::index_t nhead_stride_o = out.stride(2);
    ck_tile::index_t nhead_stride_lse = has_lse ? softmax_lse.stride(1) : 0;
    ck_tile::index_t nhead_stride_randval = has_dropout_randval ? dropout_randval.stride(1) : 0;

    ck_tile::index_t batch_stride_q = q.stride(0);
    ck_tile::index_t batch_stride_k = k.stride(0);
    ck_tile::index_t batch_stride_v = v.stride(0);
    ck_tile::index_t batch_stride_o = out.stride(0);

    ck_tile::index_t batch_stride_lse = has_lse ? softmax_lse.stride(0) : 0;
    ck_tile::index_t batch_stride_randval = has_dropout_randval ? dropout_randval.stride(0) : 0;

    void *alibi_slopes_ptr = nullptr;
    ck_tile::index_t stride_alibi_slopes = 0;

    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) || alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        alibi_slopes_ptr = alibi_slopes.data_ptr();
        stride_alibi_slopes = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    return fmha_fwd_args{
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        alibi_slopes_ptr, // bias
        nullptr,          // q_descale_ptr
        nullptr,          // k_descale_ptr
        nullptr,          // v_descale_ptr
        has_dropout_randval ? dropout_randval.data_ptr() : nullptr,
        has_lse ? softmax_lse.data_ptr() : nullptr,
        out.data_ptr(),
        nullptr, // seqstart_q_ptr
        nullptr, // seqstart_k_ptr
        nullptr, // seqlen_q_ptr
        nullptr, // seqlen_k_ptr
        nullptr, // cu_seqlen_q_ptr
        nullptr, // cu_seqlen_k_ptr
        nullptr, // block_scale_seqstart_q_ptr
        nullptr, // block_scale_seqstart_k_ptr
        nullptr, // sink_ptr
        seqlen_q,
        seqlen_k,
        b,
        seqlen_q, // max_seqlen_q
        d,        // hdim_q
        d,        // hdim_v
        h,        // nhead
        h_k,      // nhead_k
        0,        // num_head_q_total
        0,        // head_start
        softmax_scale, // scale_s
        0.0f,          // logits_soft_cap
        stride_q,
        stride_k,
        stride_v,
        stride_alibi_slopes,
        stride_randval,
        stride_o,
        nhead_stride_q,
        nhead_stride_k,
        nhead_stride_v,
        0, // nhead_stride_bias, FA without bias
        nhead_stride_randval,
        nhead_stride_lse,
        nhead_stride_o,
        0, // nhead_stride_q_descale
        0, // nhead_stride_k_descale
        0, // nhead_stride_v_descale
        batch_stride_q,
        batch_stride_k,
        batch_stride_v,
        0, // batch_stride_bias, FA without bias
        batch_stride_randval,
        batch_stride_lse,
        batch_stride_o,
        0, // batch_stride_q_descale
        0, // batch_stride_k_descale
        0, // batch_stride_v_descale
        mask.left,
        mask.right,
        0, // sink_size
        static_cast<ck_tile::index_t>(mask.type),
        0, // min_seqlen_q
        p_dropout,
        has_dropout_randval,
        std::make_pair(static_cast<const void*>(drop_seed_offset.first),
                       static_cast<const void*>(drop_seed_offset.second)),
        0, // block_scale_size_q
        0  // block_scale_size_kv
    };
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,                            // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,          // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float /*softcap*/,
        const bool return_dropout_randval,
        std::optional<at::Generator> gen_)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }

    mask_info mask;
    if (is_causal) {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right = 0;
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
    }
    else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
    }
    else {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // local
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
    const int ngroups = num_heads / num_heads_k;
    if (seqlenq_ngroups_swapped) {
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
        if (seqlenq_ngroups_swapped) {
            out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
        }
    }
    else {
        out = torch::empty_like(q);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();
    const bool needs_grad = q.requires_grad() || k.requires_grad() || v.requires_grad();
#ifdef FLASH_ATTENTION_CK_FWD_ONLY
    TORCH_CHECK(p_dropout == 0.0f, "CK fwd-only build supports p_dropout=0");
    // FWD-only build: always prefer NLSE kernels.
    bool has_lse = false;
    bool has_dropout = false;
#else
    // In non fwd-only builds, use LSE when gradients or softmax/dropout outputs are needed.
    bool has_lse = needs_grad || return_dropout_randval || return_softmax;
    bool has_dropout = p_dropout > 0.0f;
#endif

    // Still allocate softmax_lse to satisfy the Python interface shape.
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(torch::kFloat32));

    at::Tensor p;
    if (return_dropout_randval) {
        TORCH_CHECK(has_dropout, "return_dropout_randval require p_dropout > 0");
        p = torch::empty({batch_size, num_heads, seqlen_q, seqlen_k}, opts.dtype(torch::kUInt8));
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        auto philox_args = gen->philox_cuda_state(counter_offset);
        hipLaunchKernelGGL(
            flash::ParsePhiloxCudaState, dim3(1), dim3(64), 0, 0, philox_args, rng_state_ptr);
    }

    if (seqlen_k > 0) {
        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
#ifdef HIPIFY_V2
        auto stream = at::cuda::getCurrentCUDAStream().stream();
#else
        auto stream = at::cuda::getCurrentHIPStream().stream();
#endif
        ck_tile::stream_config stream_config{stream};

        auto traits =
            get_ck_fmha_fwd_traits(
                mask,
                q_dtype_str,
                head_size,
                has_dropout,
                has_lse,
                alibi_slopes_.has_value());

        auto args =
            get_ck_fmha_fwd_args(
                has_lse,
                return_dropout_randval,
                mask,
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                q,
                k,
                v,
                alibi_slopes_,
                out,
                softmax_lse,
                p,
                softmax_scale,
                p_dropout,
                drop_seed_offset);
        args.num_head_q_total = num_heads;
        args.head_start       = 0;

        const auto offset_const_ptr = [](const void* p, ck_tile::index_t elem_offset, size_t elem_bytes) {
            if(p == nullptr || elem_offset == 0)
                return p;
            return static_cast<const void*>(static_cast<const char*>(p) + elem_offset * elem_bytes);
        };
        const auto offset_mut_ptr = [](void* p, ck_tile::index_t elem_offset, size_t elem_bytes) {
            if(p == nullptr || elem_offset == 0)
                return p;
            return static_cast<void*>(static_cast<char*>(p) + elem_offset * elem_bytes);
        };

        const size_t q_elem_bytes    = q.element_size();
        const size_t k_elem_bytes    = k.element_size();
        const size_t v_elem_bytes    = v.element_size();
        const size_t o_elem_bytes    = out.element_size();
        const size_t lse_elem_bytes  = has_lse ? softmax_lse.element_size() : 0;
        const size_t rand_elem_bytes = return_dropout_randval ? p.element_size() : 0;
        const size_t bias_elem_bytes = alibi_slopes_.has_value() ? alibi_slopes_.value().element_size() : 0;

        const auto run_fwd_head_grouped = [&](ck_tile::index_t group_size) {
            fmha_fwd_traits base_traits = traits;
            fmha_fwd_args base_args     = args;

            if(num_heads_k <= 0 || (num_heads % num_heads_k) != 0)
                return -1.0f;
            const ck_tile::index_t gqa_ratio = num_heads / num_heads_k;

            float total_time = 0.0f;
            bool first_group = true;
            for(ck_tile::index_t start_h = 0; start_h < num_heads; start_h += group_size)
            {
                const ck_tile::index_t end_h     = std::min(start_h + group_size, (ck_tile::index_t)num_heads);
                const ck_tile::index_t heads_q   = end_h - start_h;
                const ck_tile::index_t start_h_k = start_h / gqa_ratio;
                const ck_tile::index_t end_h_k   = ck_tile::integer_divide_ceil(end_h, gqa_ratio);
                const ck_tile::index_t heads_k   = end_h_k - start_h_k;

                fmha_fwd_args fmha_args   = base_args;
                fmha_args.nhead_q         = heads_q;
                fmha_args.nhead_k         = heads_k;
                fmha_args.num_head_q_total = num_heads;
                fmha_args.head_start       = start_h;

                fmha_args.q_ptr = offset_const_ptr(
                    fmha_args.q_ptr, start_h * fmha_args.nhead_stride_q, q_elem_bytes);
                fmha_args.k_ptr = offset_const_ptr(
                    fmha_args.k_ptr, start_h_k * fmha_args.nhead_stride_k, k_elem_bytes);
                fmha_args.v_ptr = offset_const_ptr(
                    fmha_args.v_ptr, start_h_k * fmha_args.nhead_stride_v, v_elem_bytes);
                fmha_args.o_ptr = offset_mut_ptr(
                    fmha_args.o_ptr, start_h * fmha_args.nhead_stride_o, o_elem_bytes);

                if(fmha_args.bias_ptr != nullptr)
                {
                    fmha_args.bias_ptr = offset_const_ptr(
                        fmha_args.bias_ptr, start_h * fmha_args.nhead_stride_bias, bias_elem_bytes);
                }
                if(fmha_args.lse_ptr != nullptr)
                {
                    fmha_args.lse_ptr = offset_mut_ptr(
                        fmha_args.lse_ptr, start_h * fmha_args.nhead_stride_lse, lse_elem_bytes);
                }
                if(fmha_args.rand_val_ptr != nullptr)
                {
                    fmha_args.rand_val_ptr = offset_mut_ptr(
                        fmha_args.rand_val_ptr,
                        start_h * fmha_args.nhead_stride_randval,
                        rand_elem_bytes);
                }
                if(fmha_args.sink_ptr != nullptr)
                {
                    fmha_args.sink_ptr = offset_const_ptr(fmha_args.sink_ptr, start_h, sizeof(float));
                }
                if(fmha_args.q_descale_ptr != nullptr)
                {
                    fmha_args.q_descale_ptr = offset_const_ptr(
                        fmha_args.q_descale_ptr,
                        start_h * fmha_args.nhead_stride_q_descale,
                        sizeof(float));
                }
                if(fmha_args.k_descale_ptr != nullptr)
                {
                    fmha_args.k_descale_ptr = offset_const_ptr(
                        fmha_args.k_descale_ptr,
                        start_h_k * fmha_args.nhead_stride_k_descale,
                        sizeof(float));
                }
                if(fmha_args.v_descale_ptr != nullptr)
                {
                    fmha_args.v_descale_ptr = offset_const_ptr(
                        fmha_args.v_descale_ptr,
                        start_h_k * fmha_args.nhead_stride_v_descale,
                        sizeof(float));
                }

                if(ck_tile_head_group_log_enabled())
                {
                    std::cout << "[LLC Head Grouping] group heads_q=[" << start_h << ", " << end_h
                              << ") heads_k=[" << start_h_k << ", " << end_h_k << ")"
                              << std::endl;
                }

                ck_tile::stream_config sc_group = stream_config;
                if(!first_group)
                    sc_group.log_level_ = 0;
                const float t = fmha_fwd(base_traits, fmha_args, sc_group);
                if(t < 0.0f)
                    return t;
                total_time += t;
                first_group = false;
            }
            return total_time;
        };

        float t = -1.0f;
        if(ck_tile_head_grouping_disabled_by_env())
        {
            if(ck_tile_head_group_log_enabled())
                std::cout << "[LLC Head Grouping] disabled by env" << std::endl;
        }
        else
        {
            const auto group_size_opt = ck_tile_get_head_group_size(
                num_heads,
                num_heads_k,
                batch_size,
                seqlen_k,
                head_size,
                head_size,
                k_elem_bytes,
                v_elem_bytes);
            if(group_size_opt.has_value() && group_size_opt.value() < num_heads)
            {
                if(ck_tile_head_group_log_enabled())
                {
                    int device = 0;
                    hipDeviceProp_t props{};
                    std::string arch = {};
                    size_t llc_bytes = 0;
                    if(hipGetDevice(&device) == hipSuccess &&
                       hipGetDeviceProperties(&props, device) == hipSuccess)
                    {
                        arch      = ck_tile_trim_gfx_arch(props.gcnArchName);
                        llc_bytes = ck_tile_get_llc_cache_bytes(arch);
                    }
                    const ck_tile::index_t gqa_ratio = (num_heads_k > 0 ? (num_heads / num_heads_k) : 1);
                    const ck_tile::index_t group_sz  = group_size_opt.value();
                    const ck_tile::index_t n_groups  = ck_tile::integer_divide_ceil(num_heads, group_sz);
                    std::cout << "[LLC Head Grouping] enabled" << std::endl;
                    std::cout << "[LLC Head Grouping] arch=" << (arch.empty() ? "unknown" : arch)
                              << " llc_mb=" << (llc_bytes / (1024ull * 1024ull))
                              << " nhead_q=" << num_heads << " nhead_k=" << num_heads_k
                              << " gqa_ratio=" << gqa_ratio << " group_size=" << group_sz
                              << " groups=" << n_groups << std::endl;
                }
                t = run_fwd_head_grouped(group_size_opt.value());
            }
            else if(ck_tile_head_group_log_enabled())
            {
                std::cout << "[LLC Head Grouping] skipped (group_size not set or >= nhead)"
                          << std::endl;
            }
        }
        if(t < 0.0f)
        {
            t = fmha_fwd(traits, args, stream_config);
        }
        TORCH_CHECK(t >= 0, "invalid argument for fmha_fwd");
    }
    else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }
    return {out, softmax_lse, p, rng_state};
}
