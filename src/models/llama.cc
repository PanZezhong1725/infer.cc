#include "../tensor.h"
#include "../utils.h"
#include "infini_infer.h"
#include "infiniccl.h"
#include "infinirt.h"
#include "llama_weights.h"
#include <random>
#include <thread>
#include <vector>

struct DeviceResource
{
    // Device
    DeviceType device;
    unsigned int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table,
        cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    // Streams
    infinirtStream_t stream_compute, stream_data, stream_cache;
    infinicclComm_t comm;
};

void create_device_resource(DeviceResource *rsrc, LlamaMeta const *meta,
                                   LlamaWeights const *weights,
                                   DeviceType device, unsigned int idev,
                                   unsigned int ndev, unsigned int dev_id,
                                   infinicclComm_t comm) {
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle, (Device)device, dev_id);
    infinirtStream_t stream_compute, stream_data, stream_cache;
    infinirtStreamCreate(&stream_compute, device, dev_id);
    infinirtStreamCreate(&stream_data, device, dev_id);
    infinirtStreamCreate(&stream_cache, device, dev_id);
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            get_attn_norm(meta, weights, layer, device, dev_id));
        w_attn_qkv.push_back(
            get_attn_qkv(meta, weights, layer, idev, ndev, device, dev_id));
        w_attn_out.push_back(
            get_attn_o(meta, weights, layer, idev, ndev, device, dev_id));
        w_ffn_norm.push_back(
            get_ffn_norm(meta, weights, layer, device, dev_id));
        w_ffn_gate_up.push_back(
            get_ffn_gate_up(meta, weights, layer, idev, ndev, device, dev_id));
        w_ffn_down.push_back(
            get_ffn_down(meta, weights, layer, idev, ndev, device, dev_id));
    }

    *rsrc = DeviceResource{device,
                              dev_id,
                              handle,
                              get_in_embd(meta, weights, device, dev_id),
                              get_out_norm(meta, weights, device, dev_id),
                              get_out_embd(meta, weights, device, dev_id),
                              get_sin_table(meta, device, dev_id),
                              get_cos_table(meta, device, dev_id),
                              w_attn_norm,
                              w_attn_qkv,
                              w_attn_out,
                              w_ffn_norm,
                              w_ffn_gate_up,
                              w_ffn_down,
                              stream_compute,
                              stream_data,
                              stream_cache,
                              comm};
}

struct Model
{
    LlamaMeta meta;
    std::vector<DeviceResource> const dev;
    Model(LlamaMeta const &_meta, std::vector<DeviceResource> const &&_dev)
        : meta(_meta), dev(std::move(_dev)) {}
};

__C struct Model *create_model(LlamaMeta const *meta,
                               LlamaWeights const *weights, DeviceType device,
                               unsigned int ndev, unsigned int const *dev_ids) {
    ASSERT_EQ(meta->nh % ndev, 0);
    ASSERT_EQ(meta->nkvh % ndev, 0);
    ASSERT_EQ(meta->di % ndev, 0);
    RUN_INFINI(infinirtInit(device));
    auto dev = std::vector<DeviceResource>(ndev);
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids));
    }
    auto threads = std::vector<std::thread>(ndev);
    for (unsigned int idev = 0; idev < ndev; idev++) {
        threads[idev] =
            std::thread(create_device_resource, &(dev[idev]), meta, weights,
                        device, idev, ndev, dev_ids[idev], comms[idev]);
    }
    for (unsigned int idev = 0; idev < ndev; idev++) {
        threads[idev].join();
    }
    
    auto model = new Model(*meta, std::move(dev));
    return model;
}

struct KVCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k, v;
};

__C struct KVCache *create_kv_cache(struct Model const *model) {
    KVCache *cache = new KVCache();
    auto ndev = model->dev.size();
    auto nkvh = model->meta.nkvh / ndev;
    auto max_len = model->meta.dctx;
    auto dh = model->meta.dh;
    auto shape = std::vector<index_t>{nkvh, max_len, dh};
    for (unsigned int idev = 0; idev < ndev; idev++) {
        auto kcache = std::vector<std::shared_ptr<Tensor>>();
        auto vcache = std::vector<std::shared_ptr<Tensor>>();
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            kcache.push_back(std::move(Tensor::buffer(model->meta.dt_mat, shape,
                                                model->dev[idev].device,
                                                model->dev[idev].device_id,
                                                model->dev[idev].stream_cache)));
            vcache.push_back(std::move(Tensor::buffer(model->meta.dt_mat, shape,
                                                model->dev[idev].device,
                                                model->dev[idev].device_id,
                                                model->dev[idev].stream_cache)));
        }
        cache->k.push_back(kcache);
        cache->v.push_back(vcache);
    }

    return cache;
}

__C struct KVCache *duplicate_kv_cache(struct Model const *model,
                                       struct KVCache const *kv_cache,
                                       unsigned int seq_len) {
    auto new_kv_cache = create_kv_cache(model);
    auto ndev = model->dev.size();
    for (unsigned int idev = 0; idev < ndev; idev++) {
        for (unsigned int layer = 0; layer < model->meta.nlayer; layer++) {
            new_kv_cache->k[idev][layer]
                ->slice(1, 0, seq_len)
                ->copy_from(kv_cache->k[idev][layer]->slice(1, 0, seq_len),
                            model->dev[idev].handle,
                            model->dev[idev].stream_cache);

            new_kv_cache->v[idev][layer]
                ->slice(1, 0, seq_len)
                ->copy_from(kv_cache->v[idev][layer]->slice(1, 0, seq_len),
                            model->dev[idev].handle,
                            model->dev[idev].stream_cache);
        }
    }
    return new_kv_cache;
}

__C void drop_kv_cache(struct Model const *, struct KVCache *kv_cache) {
    delete kv_cache;
}

void infer_device(LlamaMeta const &meta, DeviceResource const &rsrc,
                  unsigned int idev, unsigned int ndev, unsigned int ntok,
                  unsigned int const *tokens, unsigned int nreq,
                  unsigned int const *req_lens, unsigned int const *req_pos,
                  struct KVCache **kv_caches, unsigned int *ans,
                  float temperature, unsigned int topk, float topp) {
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    auto dctx = meta.dctx;
    auto dh = meta.dh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto di = meta.di / ndev;
    auto dvoc = meta.dvoc;
    auto device = rsrc.device;
    auto device_id = rsrc.device_id;
    auto stream_compute = rsrc.stream_compute;
    auto stream_data = rsrc.stream_data;
    auto stream_cache = rsrc.stream_cache;
    void *stream_compute_raw;
    infinirtGetRawStream(&stream_compute_raw, stream_compute);

    // Allocate buffers
    auto logits_in =
        Tensor::buffer(dt_logits, {ntok, d}, device, device_id, stream_data);
    auto logits_out =
        Tensor::buffer(dt_logits, {ntok, d}, device, device_id, stream_data);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh},
                                  device, device_id, stream_data);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, device, device_id,
                                stream_data);
    auto prob_buf =
        Tensor::buffer(dt_logits, {nreq, dvoc}, device, device_id, stream_data);
    auto result_buf =
        Tensor::buffer(INFINI_U64, {nreq}, device, device_id, stream_data);
    auto result_cpu = std::vector<uint64_t>(nreq);
    // Prepare inputs
    auto batch_pos_ids = std::vector<index_t>(ntok);
    index_t req_start = 0;
    for (unsigned int req = 0; req < nreq; req++) {
        for (unsigned int i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_U64, {ntok},
                                     rsrc.device, rsrc.device_id);
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_U64, {ntok}, rsrc.device,
                                     rsrc.device_id, rsrc.stream_compute);
        RUN_INFINI(infinirtMemcpyH2DAsync(pos_ids_buf->data(rsrc.stream_compute), device,
                               device_id, batch_pos_ids.data(), sizeof(uint64_t) * ntok,
                               rsrc.stream_compute));
    }
    for (unsigned int i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d, stream_compute),
                            rsrc.w_in_embd->data(tokens[i] * d, stream_compute),
                            device, device_id,
                            dt_size(dt_logits) * d, stream_compute));
    }

    // Prepare operators and workspace
    void *workspace;
    size_t workspace_size = 0, temp_size = 0;
    infiniopRMSNormDescriptor_t desc_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm, logits_in->desc()->get(),
        logits_out->desc()->get(), rsrc.w_attn_norm[0]->desc()->get(),
        meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &workspace_size));
    infiniopMatmulDescriptor_t desc_attn_qkv, desc_attn_o;
    RUN_INFINI(infiniopCreateMatmulDescriptor(
        rsrc.handle, &desc_attn_qkv, qkv_buf->desc()->get(), 1.0,
        logits_in->desc()->get(), rsrc.w_attn_qkv[0]->desc()->get(), 0.0));
    RUN_INFINI(infiniopCreateMatmulDescriptor(
        rsrc.handle, &desc_attn_o, logits_in->desc()->get(), 1.0,
        o_buf->desc()->get(), rsrc.w_attn_out[0]->desc()->get(),
        idev == 0 ? 1.0 : 0.0)); // only rank 0 adds residual
    RUN_INFINI(infiniopGetMatmulWorkspaceSize(desc_attn_qkv, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetMatmulWorkspaceSize(desc_attn_o, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopRoPEDescriptor_t desc_rope_q, desc_rope_k;
    qkv_buf->dim_split(1, {nh + nkvh * 2, dh}); // (ntok, nh + 2 * nkvh, dh)
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_q, qkv_buf->slice(1, 0, nh)->desc()->get(),
        pos_ids_buf->desc()->get(), rsrc.sin_table->desc()->get(),
        rsrc.cos_table->desc()->get()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_q, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_k, qkv_buf->slice(1, nh, nkvh)->desc()->get(),
        pos_ids_buf->desc()->get(), rsrc.sin_table->desc()->get(),
        rsrc.cos_table->desc()->get()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_k, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopMLPDescriptor_t desc_mlp;
    RUN_INFINI(infiniopCreateMLPDescriptor(
        rsrc.handle, &desc_mlp, logits_in->desc()->get(),
        logits_out->desc()->get(), rsrc.w_ffn_gate_up[0]->desc()->get(),
        rsrc.w_ffn_down[0]->desc()->get(), 1.0, idev == 0));
    RUN_INFINI(infiniopGetMLPWorkspaceSize(desc_mlp, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    auto desc_attns = std::vector<infiniopAttentionDescriptor_t>(nreq);
    size_t token_offset = 0;
    o_buf->dim_split(1, {nh, dh});
    for (unsigned int req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto o = o_buf->slice({{0, token_offset, seq_len}});
        auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}})
                     ->permute({1, 0, 2});
        auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}})
                     ->permute({1, 0, 2});
        auto v =
            qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}})
                ->permute({1, 0, 2});
        auto k_cache = kv_caches[req]->k[idev][0];
        auto v_cache = kv_caches[req]->v[idev][0];
        RUN_INFINI(infiniopCreateAttentionDescriptor(
            rsrc.handle, &desc_attns[req], o->desc()->get(), q->desc()->get(),
            k->desc()->get(), v->desc()->get(), k_cache->desc()->get(),
            v_cache->desc()->get(), past_len));
        RUN_INFINI(
            infiniopGetAttentionWorkspaceSize(desc_attns[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);
        token_offset += seq_len;
    }
    infiniopRMSNormDescriptor_t desc_norm_out;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_out, logits_out->slice(0, 0, 1)->desc()->get(),
        logits_out->slice(0, 0, 1)->desc()->get(),
        rsrc.w_out_norm->desc()->get(), meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopMatmulDescriptor_t desc_out_embd;
    RUN_INFINI(infiniopCreateMatmulDescriptor(
        rsrc.handle, &desc_out_embd, prob_buf->desc()->get(), 1.0,
        logits_out->slice(0, 0, nreq)->desc()->get(),
        rsrc.w_out_embd->desc()->get(), 0.0));
    RUN_INFINI(infiniopGetMatmulWorkspaceSize(desc_out_embd, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopRandomSampleDescriptor_t desc_sample;
    RUN_INFINI(infiniopCreateRandomSampleDescriptor(
        rsrc.handle, &desc_sample,
        TensorDesc::create(INFINI_U64, {1}, {1})->get(),
        TensorDesc::create(dt_logits, {dvoc}, {1})->get()));
    RUN_INFINI(infiniopGetRandomSampleWorkspaceSize(desc_sample, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    // Allocate workspace
    RUN_INFINI(infinirtMallocAsync(&workspace, device, device_id,
                                   workspace_size, stream_compute));
    for (unsigned int layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(stream_compute), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(stream_compute), stream_compute_raw));
        // qkv_proj
        RUN_INFINI(infiniopMatmul(
            desc_attn_qkv, workspace, workspace_size,
            qkv_buf->data(stream_compute), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(stream_compute), stream_compute_raw));
        // rope
        RUN_INFINI(infiniopRoPE(
            desc_rope_q, workspace, workspace_size,
            qkv_buf->data(stream_compute), pos_ids_buf->data(stream_compute),
            rsrc.sin_table->data(stream_compute),
            rsrc.cos_table->data(stream_compute), stream_compute_raw));
        RUN_INFINI(infiniopRoPE(desc_rope_k, workspace, workspace_size,
                                qkv_buf->data(nh * dh, stream_compute),
                                pos_ids_buf->data(stream_compute),
                                rsrc.sin_table->data(stream_compute),
                                rsrc.cos_table->data(stream_compute),
                                stream_compute_raw));

        size_t token_offset = 0;
        for (unsigned int req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            // self attention
            RUN_INFINI(infiniopAttention(
                desc_attns[req], workspace, workspace_size,
                o_buf->data(token_offset * nh * dh, stream_compute),
                qkv_buf->data(token_offset * (nh + nkvh * 2) * dh,
                              stream_compute),
                qkv_buf->data(token_offset * (nh + nkvh * 2) * dh + nh * dh,
                              stream_compute),
                qkv_buf->data(token_offset * (nh + nkvh * 2) * dh +
                                  (nh + nkvh) * dh,
                              stream_compute),
                kv_caches[req]->k[idev][layer]->data(stream_compute),
                kv_caches[req]->v[idev][layer]->data(stream_compute),
                stream_compute_raw));

            token_offset += seq_len;
        }
        // o_proj
        RUN_INFINI(infiniopMatmul(
            desc_attn_o, workspace, workspace_size,
            logits_in->data(stream_compute), o_buf->data(),
            rsrc.w_attn_out[layer]->data(stream_compute), stream_compute_raw));

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduceSum(
                rsrc.comm, logits_in->data(stream_compute),
                logits_in->data(stream_compute), ntok * d, dt_logits,
                stream_compute));
        }

        // 2. FFN
        // rms_norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(stream_compute), logits_in->data(stream_compute),
            rsrc.w_ffn_norm[layer]->data(stream_compute), stream_compute_raw));
        // mlp
        RUN_INFINI(infiniopMLP(
            desc_mlp, workspace, workspace_size,
            logits_in->data(stream_compute), logits_out->data(stream_compute),
            rsrc.w_ffn_gate_up[layer]->data(stream_compute),
            rsrc.w_ffn_down[layer]->data(stream_compute), stream_compute_raw));

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduceSum(
                rsrc.comm, logits_in->data(stream_compute),
                logits_in->data(stream_compute), ntok * d, dt_logits,
                stream_compute));
        }
    }
    // Sample and Output
    uint64_t tmp;
    if (idev == 0) {
        size_t token_offset = 0;
        for (unsigned int req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            token_offset += seq_len;   
            RUN_INFINI(infiniopRMSNorm(
                desc_norm_out, workspace, workspace_size,
                logits_out->data(req * d, stream_compute),
                logits_in->data((token_offset - 1) * d, stream_compute),
                rsrc.w_out_norm->data(stream_compute), stream_compute_raw));
        }
        RUN_INFINI(infiniopMatmul(
            desc_out_embd, workspace, workspace_size,
            prob_buf->data(stream_compute), logits_out->data(stream_compute),
            rsrc.w_out_embd->data(stream_compute), stream_compute_raw));
        std::random_device _rd;
        std::mt19937 gen(_rd());
        token_offset = 0;
        for (unsigned int req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
            RUN_INFINI(infiniopRandomSample(
                desc_sample, workspace, workspace_size,
                result_buf->data(req, stream_compute),
                prob_buf->data(req * dvoc, stream_compute), random_val, topp,
                topk, temperature, stream_compute_raw));
            token_offset += seq_len;
        }
        RUN_INFINI(infinirtStreamSynchronize(stream_compute));
        RUN_INFINI(infinirtMemcpyD2H(&tmp, result_buf->data(stream_data),
                          device, device_id, sizeof(uint64_t) * nreq));
        for (unsigned int req = 0; req < nreq; req++) {
            // ans[req] = (unsigned int)result_cpu[req];
            ans[req] = (unsigned int)tmp;

        }
    }

    // Clean up
    infiniopDestroyRMSNormDescriptor(desc_norm);
    infiniopDestroyMatmulDescriptor(desc_attn_qkv);
    infiniopDestroyMatmulDescriptor(desc_attn_o);
    infiniopDestroyRoPEDescriptor(desc_rope_q);
    infiniopDestroyRoPEDescriptor(desc_rope_k);
    infiniopDestroyMLPDescriptor(desc_mlp);
    for (unsigned int req = 0; req < nreq; req++) {
        infiniopDestroyAttentionDescriptor(desc_attns[req]);
    }
    infiniopDestroyRMSNormDescriptor(desc_norm_out);
    infiniopDestroyMatmulDescriptor(desc_out_embd);
    infiniopDestroyRandomSampleDescriptor(desc_sample);
    infinirtFree(workspace, device, device_id);
}

__C void infer(struct Model const *model, unsigned int ntok,
               unsigned int const *tokens, unsigned int nreq,
               unsigned int const *req_lens, unsigned int const *req_pos,
               struct KVCache **kv_caches, unsigned int *ans, float temperature,
               unsigned int topk, float topp) {
    auto ndev = model->dev.size();
    auto threads = std::vector<std::thread>(ndev);
    for (unsigned int idev = 0; idev < ndev; idev++) {
        threads[idev] =
            std::thread(infer_device, model->meta, model->dev[idev], idev, ndev,
                        ntok, tokens, nreq, req_lens, req_pos, kv_caches, ans,
                        temperature, topk, topp);
    }
    for (unsigned int idev = 0; idev < ndev; idev++) {
        threads[idev].join();
    }
}

__C void destroy_model(struct Model *model) {
    auto ndev = model->dev.size();
    for (unsigned int i = 0; i < ndev; i++) {
        infiniopDestroyHandle(model->dev[i].handle);
        infinirtStreamDestroy(model->dev[i].stream_compute);
        infinirtStreamDestroy(model->dev[i].stream_data);
        infinirtStreamDestroy(model->dev[i].stream_cache);
        infinicclCommDestroy(model->dev[i].comm);
    }
    delete model;
}
