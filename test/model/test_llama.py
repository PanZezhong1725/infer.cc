import ctypes
from ctypes import c_void_p, c_uint, c_float, POINTER
import sys
from libinfer import open_library, DataType, DeviceType, LlamaWeights, LlamaMeta, KVCache
import torch
import transformers
import time
    
lib = open_library()

# Define the LlamaWeights struct
class LlamaWeightsHF(LlamaWeights):
    def __init__(self, llama, ndev=1):
        self.nlayer = llama.config.num_hidden_layers
        state_dict = llama.state_dict()
        self.input_embd = state_dict["model.embed_tokens.weight"].data_ptr()
        self.output_norm = state_dict["model.norm.weight"].data_ptr()
        self.output_embd = state_dict["lm_head.weight"].data_ptr()
        self.attn_norm = (c_void_p * self.nlayer)(
            *[
                state_dict[f"model.layers.{i}.input_layernorm.weight"].data_ptr()
                for i in range(self.nlayer)
            ]
        )
        nh = llama.config.num_attention_heads
        nkvh = llama.config.num_key_value_heads
        dh = llama.config.hidden_size // llama.config.num_attention_heads
        d = llama.config.hidden_size
        di = llama.config.intermediate_size
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        def qkv_slices(_i):
            _Q = state_dict[f"model.layers.{_i}.self_attn.q_proj.weight"].reshape([nh, 2, dh // 2, d]).transpose(1, 2)
            _K = state_dict[f"model.layers.{_i}.self_attn.k_proj.weight"].reshape([nkvh, 2, dh // 2, d]).transpose(1, 2)
            _V = state_dict[f"model.layers.{_i}.self_attn.v_proj.weight"].reshape([nkvh, dh // 2, 2, d])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(
                    _Q[_idev * _nh:(_idev + 1) * _nh, :, :, :]
                )
                _result.append(
                    _K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :, :]
                )
                _result.append(
                    _V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :]
                )
            return _result
        self.qkv_tensor = [
            torch.concat(
                qkv_slices(i)
            )
            for i in range(self.nlayer)
        ]
        self.attn_qkv = (c_void_p * self.nlayer)(
            *[self.qkv_tensor[i].data_ptr() for i in range(self.nlayer)]
        )
        self.attn_o_tensor = [
            state_dict[f"model.layers.{i}.self_attn.o_proj.weight"]
            .reshape([d, ndev, nh//ndev * dh])
            .transpose(0, 1)
            .contiguous()
            for i in range(self.nlayer)
        ]
        self.attn_o = (c_void_p * self.nlayer)(
            *[
                self.attn_o_tensor[i].data_ptr()
                for i in range(self.nlayer)
            ]
        )
        self.ffn_norm = (c_void_p * self.nlayer)(
            *[
                state_dict[
                    f"model.layers.{i}.post_attention_layernorm.weight"
                ].data_ptr()
                for i in range(self.nlayer)
            ]
        )
        def gate_up_slices(_i):
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(
                    state_dict[f"model.layers.{_i}.mlp.gate_proj.weight"][_start:_end, :]
                )
                _result.append(
                    state_dict[f"model.layers.{_i}.mlp.up_proj.weight"][_start:_end, :]
                )
            return _result
        
        self.gate_up_tensor = [
            torch.concat(
                gate_up_slices(i)
            )
            for i in range(self.nlayer)
        ]

        self.ffn_gate_up = (c_void_p * self.nlayer)(
            *[self.gate_up_tensor[i].data_ptr() for i in range(self.nlayer)]
        )
        
        self.ffn_down_tensor = [
            state_dict[f"model.layers.{i}.mlp.down_proj.weight"]
            .reshape([d, ndev, di//ndev])
            .transpose(0, 1)
            .contiguous()
            for i in range(self.nlayer)
        ]
        self.ffn_down = (c_void_p * self.nlayer)(
            *[
                self.ffn_down_tensor[i].data_ptr()
                for i in range(self.nlayer)
            ]
        )

class LlamaModel():
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, n_device = 1):
        llama = transformers.LlamaForCausalLM.from_pretrained(
            model_dir_path, torch_dtype=torch.float16
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path
        )

        self.meta = LlamaMeta(
            dt_logits=DataType.INFINI_F16,
            dt_norm=DataType.INFINI_F16,
            dt_mat=DataType.INFINI_F16,
            nlayer=llama.config.num_hidden_layers,
            d=llama.config.hidden_size,
            nh=llama.config.num_attention_heads,
            nkvh=(
                llama.config.num_key_value_heads
                if llama.config.num_key_value_heads
                else llama.config.num_attention_heads
            ),
            dh=llama.config.hidden_size // llama.config.num_attention_heads,
            di=llama.config.intermediate_size,
            dctx=llama.config.max_position_embeddings,
            dvoc=llama.config.vocab_size,
            epsilon=llama.config.rms_norm_eps,
            theta=llama.config.rope_theta,
        )

        self.weights = LlamaWeightsHF(llama, n_device)
        dev_ids = (c_uint * n_device)(*[i for i in range(n_device)])
        self.model_instance = lib.create_model(
            ctypes.byref(self.meta),
            ctypes.byref(self.weights),
            device,
            n_device,
            dev_ids,
        )
    
    def infer(self, input_content, max_steps, topp=1.0, topk=1, temperature=1.0):
        print(input_content, end="", flush=True)
        kv_cache = lib.create_kv_cache(self.model_instance)
        tokens = self.tokenizer.encode(input_content)
        ntok = len(tokens)
        nreq = 1
        output_content = ""
        tokens = (c_uint * ntok)(*tokens)
        req_lens = (c_uint * nreq)(*[ntok])
        req_pos = (c_uint * nreq)(*[0])
        kv_caches = (POINTER(KVCache) * nreq)(*[kv_cache])
        ans = (c_uint * nreq)()

        steps = 0
        start_time = time.time()
        for _ in range(max_steps):
            lib.infer(
                self.model_instance,
                ntok,
                tokens,
                nreq,
                req_lens,
                req_pos,
                kv_caches,
                ans,
                temperature,
                topk,
                topp,
            )
            steps += 1
            output_tokens = list(ans)
            output_str = self.tokenizer._tokenizer.id_to_token(output_tokens[0]).replace("▁", " ").replace("<0x0A>", "\n")
            if output_str.endswith("</s>"):
                break
            output_content += output_str
            print(output_str, end="", flush=True)
            req_pos[0] = req_pos[0] + ntok
            ntok = 1
            tokens = (c_uint * ntok)(*output_tokens)
            req_lens = (c_uint * nreq)(*[ntok])

        print("\n")
        end_time = time.time()
        avg_time = (end_time - start_time) *  1000 / steps
        print(f"Time per step: {avg_time:.3f}ms")
        for kv_cache in kv_caches:
            lib.drop_kv_cache(self.model_instance, kv_cache)
        return output_content, avg_time

def test():
    if len(sys.argv) < 3:
        print("Usage: python test_llama.py [--cpu | --cuda | --cambricon | --ascend | --maca] <path/to/model_dir> [n_device]")
        sys.exit(1)
    model_path =  sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU
    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--cuda":
        device_type = DeviceType.DEVICE_TYPE_CUDA
    elif sys.argv[1] == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif sys.argv[1] == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif sys.argv[1] == "--maca":
        device_type = DeviceType.DEVICE_TYPE_METAX
    else:
        print("Usage: python test_llama.py [--cpu | --cuda | --cambricon | --ascend | --maca] <path/to/model_dir> [n_device]")
        sys.exit(1)
    
    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = LlamaModel(model_path, device_type, ndev)
    model.infer("讲个长故事", 500)

if __name__ == "__main__":
    test()