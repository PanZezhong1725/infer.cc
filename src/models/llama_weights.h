#include "infini_infer.h"
#include "../tensor.h"

inline Tensor get_in_embd(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->dvoc, meta->nkvh * meta->dh});
    return Tensor::weight((char *)w->input_embd, meta->dt_mat, shape, device, device_id);
}

inline Tensor get_out_norm(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->nkvh * meta->dh});
    return Tensor::weight((char *)w->output_norm, meta->dt_mat, shape, device, device_id);
}

inline Tensor get_out_embd(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->dvoc, meta->nkvh * meta->dh});
    return Tensor::weight((char *)w->output_embd, meta->dt_mat, shape, device, device_id);
}

inline Tensor get_attn_norm(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->nkvh * meta->dh});
    return Tensor::weight((char *)(w->attn_norm[layer]), meta->dt_norm, shape, device, device_id);
}

inline Tensor get_attn_qkv(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer, size_t idev, size_t ndev,
    DeviceType device, unsigned int device_id)
{
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = nkvh * dh;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d;
    auto shape = std::vector<index_t>({(nh + 2 * nkvh) / ndev * dh, d});
    return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, meta->dt_mat, shape, device, device_id);
}

inline Tensor get_attn_o(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer, size_t idev, size_t ndev,
    DeviceType device, unsigned int device_id)
{
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = nkvh * dh;
    size_t offset = idev * d * (nkvh / ndev * dh);
    auto shape = std::vector<index_t>({d, nkvh / ndev * dh});
    return Tensor::weight((char *)(w->attn_o[layer]) + offset, meta->dt_mat, shape, device, device_id);
}

inline Tensor get_ffn_norm(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer,
    DeviceType device, unsigned int device_id)
{
    auto shape = std::vector<index_t>({meta->nkvh * meta->dh});
    return Tensor::weight((char *)(w->ffn_norm[layer]), meta->dt_norm, shape, device, device_id);
}

inline Tensor get_ffn_gate_up(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer, size_t idev, size_t ndev,
    DeviceType device, unsigned int device_id)
{
    auto di = meta->di;
    auto d = meta->nkvh * meta->dh;
    size_t offset = idev * (2 * di / ndev) * d;
    auto shape = std::vector<index_t>({2 * di / ndev, d});
    return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset, meta->dt_mat, shape, device, device_id);
}

inline Tensor get_ffn_down(
    LlamaMeta const *meta,
    LlamaWeights const *w,
    size_t layer, size_t idev, size_t ndev,
    DeviceType device, unsigned int device_id)
{
    auto di = meta->di;
    auto d = meta->nkvh * meta->dh;
    size_t offset = idev * d * (di / ndev);
    auto shape = std::vector<index_t>({d, di / ndev});
    return Tensor::weight((char *)(w->ffn_down[layer]) + offset, meta->dt_mat, shape, device, device_id);
}
