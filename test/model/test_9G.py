import ctypes
from ctypes import c_void_p, c_uint, POINTER
import io
import sys
import os
import re
from typing import IO, Dict, List
from pytrie import StringTrie
import json
import torch
import time
from libinfer import open_library, DataType, DeviceType, LlamaWeights, LlamaMeta, KVCache

lib = open_library()

class LlamaWeights9G(LlamaWeights):
    def __init__(self, state_dict, meta, ndev=1):
        self.nlayer = meta.nlayer
        self.input_embd = state_dict["input_embedding.weight"].data_ptr()
        self.output_norm = state_dict["encoder.output_layernorm.weight"].data_ptr()
        self.output_embd = state_dict["lm_head.weight"].data_ptr()
        self.attn_norm = (c_void_p * self.nlayer)(
            *[
                state_dict[f"encoder.layers.{i}.self_att.layernorm_before_attention.weight"].data_ptr()
                for i in range(self.nlayer)
            ]
        )
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        def qkv_slices(_i):
            _Q = state_dict[f"encoder.layers.{_i}.self_att.self_attention.project_q.weight"].reshape([nh, 2, dh // 2, d]).transpose(1, 2)
            _K = state_dict[f"encoder.layers.{_i}.self_att.self_attention.project_k.weight"].reshape([nkvh, 2, dh // 2, d]).transpose(1, 2)
            _V = state_dict[f"encoder.layers.{_i}.self_att.self_attention.project_v.weight"].reshape([nkvh, dh // 2, 2, d])
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
            state_dict[f"encoder.layers.{i}.self_att.self_attention.attention_out.weight"]
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
                    f"encoder.layers.{i}.ffn.layernorm_before_ffn.weight"
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
                    state_dict[f"encoder.layers.{_i}.ffn.ffn.w_in.w_0.weight"][_start:_end, :]
                )
                _result.append(
                    state_dict[f"encoder.layers.{_i}.ffn.ffn.w_in.w_1.weight"][_start:_end, :]
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
            state_dict[f"encoder.layers.{i}.ffn.ffn.w_out.weight"]
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


def load_ckpt(ckpt_file_path):
    import struct
    def _load_dtype(fp):
        dtype = struct.unpack("B", fp.read(1))[0]
        return dtype

    def _load_string(fp):
        size = struct.unpack("I", fp.read(4))[0]
        return fp.read(size).decode("utf-8")

    def _load_tuple(fp):
        ndim = struct.unpack("B", fp.read(1))[0]
        ret = []
        for _ in range(ndim):
            ret.append(struct.unpack("I", fp.read(4))[0])
        return tuple(ret)
    
    ckpt = {}
    _nlayer = 0
    with open(ckpt_file_path, "rb") as fp:
        num_parameters = struct.unpack("I", fp.read(4))[0]
        _nlayer = (num_parameters - 3) // 9
        for _ in range(num_parameters):
            param_name = _load_string(fp)
            _shape = _load_tuple(fp)
            param_size = struct.unpack("I", fp.read(4))[0]
            _ = _load_dtype(fp)
            param = bytearray(fp.read(param_size))
            
            ckpt[param_name] = torch.frombuffer(param, dtype=torch.float16, count=int(param_size)//2).reshape(_shape)
    state_dict = {}
    state_dict["input_embedding.weight"] = ckpt["input_embedding.weight"]
    state_dict["lm_head.weight"] = ckpt["lm_head.weight"]
    state_dict["encoder.output_layernorm.weight"] = ckpt["output_layernorm.weight"]
    for i in range(_nlayer):
        state_dict[f"encoder.layers.{i}.self_att.layernorm_before_attention.weight"] = ckpt[f"layers.{i}.ln_attn.weight"]
        state_dict[f"encoder.layers.{i}.self_att.self_attention.project_q.weight"] = ckpt[f"layers.{i}.attn.project_q.weight"]
        state_dict[f"encoder.layers.{i}.self_att.self_attention.project_k.weight"] = ckpt[f"layers.{i}.attn.project_k.weight"]
        state_dict[f"encoder.layers.{i}.self_att.self_attention.project_v.weight"] = ckpt[f"layers.{i}.attn.project_v.weight"]
        state_dict[f"encoder.layers.{i}.self_att.self_attention.attention_out.weight"] = ckpt[f"layers.{i}.attn.attn_out.weight"]
        state_dict[f"encoder.layers.{i}.ffn.layernorm_before_ffn.weight"] = ckpt[f"layers.{i}.ln_ff.weight"]
        state_dict[f"encoder.layers.{i}.ffn.ffn.w_in.w_0.weight"] = ckpt[f"layers.{i}.ff.w_in.weight"]
        state_dict[f"encoder.layers.{i}.ffn.ffn.w_in.w_1.weight"] = ckpt[f"layers.{i}.ff.w_gated.weight"]
        state_dict[f"encoder.layers.{i}.ffn.ffn.w_out.weight"] = ckpt[f"layers.{i}.ff.w_out.weight"]
    return state_dict
 
def load_vocab(fp: IO[bytes]) -> Dict[str, int]:
    """Loads a vocabulary file into a dictionary."""
    vocab: Dict[str, int] = {}

    reader = io.TextIOWrapper(fp, encoding="utf-8")
    for token in reader.readlines():
        token = token.strip()
        if len(token) == 0:
            continue
        token = json.loads(token)
        vocab[token] = len(vocab)
    return vocab

class CPM9GTokenizer(object):
    def __init__(self, path):
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.byte_list = ["<0x0{}>".format(hex(i).upper()[2:]) for i in range(0x10)] + [
            "<0x{}>".format(hex(i).upper()[2:]) for i in range(0x10, 0x100)
        ]

        self._special_token_set = set([self.unk_token, self.bos_token, self.eos_token] + self.byte_list)

        all_tokens = load_vocab(io.FileIO(path, "rb"))

        self.encoder: Dict[str, int] = {}
        self._special_encoder: Dict[str, int] = {}
        for token, token_id in all_tokens.items():
            if token in self._special_token_set:
                self._special_encoder[token] = token_id
            else:
                self.encoder[token] = token_id

        self.decoder = {v: k for k, v in self.encoder.items()}
        self._byte_decoder = {self._special_encoder[token]: i for i, token in enumerate(self.byte_list)}

        self._max_word_len = max([len(x) for x in self.encoder.keys()])

        self._len_word_first = {}
        for x in self.encoder.keys():
            if not x[0] in self._len_word_first:
                self._len_word_first[x[0]] = 1
            if len(x) > self._len_word_first[x[0]]:
                self._len_word_first[x[0]] = len(x)
        self.tencoder = StringTrie(self.encoder)

    def get_piece(self, text: str) -> str:
        if text[0] in self._len_word_first:
            text = text[: self._len_word_first[text[0]]]
            len_text = len(text)
            for i in range(len(text)):
                sub = text[: len_text - i]
                if sub in self.encoder:
                    return sub
        return text[0]

    @property
    def vocab_size(self):
        return len(self)

    @property
    def eos_id(self):
        return self._special_encoder[self.eos_token]

    @property
    def bos_id(self):
        return self._special_encoder[self.bos_token]

    @property
    def unk_id(self):
        return self._special_encoder[self.unk_token]

    def __len__(self):
        return len(self.encoder) + len(self._special_encoder)

    def tokenize(self, text: str) -> List[str]:
        output_tokens: List[str] = []
        st = 0
        while st < len(text):
            piece = self.get_piece(text[st:])
            output_tokens.append(piece)
            st += len(piece)
        return output_tokens

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def unescape(text: str) -> str:
        return text

    def encode(self, text: str, with_bos = True) -> List[int]:
        ret = []
        if with_bos:
            ret.append(self.bos_id)
        for x in self.tokenize(text):
            if x in self.encoder:
                ret.append(self.encoder[x])
            else:
                ret.extend(self._encode_unicode(x))
        return ret

    def decode(self, tokens: List[int]):
        """Decode ids into a string."""
        ret = []
        st = 0

        while st < len(tokens):
            if tokens[st] in self.decoder:
                ret.append(self.decoder[tokens[st]])
                st += 1
            elif tokens[st] in self._byte_decoder:
                first = self._byte_decoder[tokens[st]]
                length = 1 if first < 128 else len(re.search('^1+0', bin(first)[2:])[0])-1
                code = 0
                try:
                    for j in range(length):
                        code = code << 8 | self._byte_decoder[tokens[st + j]]
                    code = int.to_bytes(code, length, "big").decode("utf-8")
                    ret.append(code)
                except:
                    pass
                st = st + length
            elif tokens[st] == self.eos_id:
                ret.append(self.eos_token)
                st += 1
            elif tokens[st] == self.bos_id:
                ret.append(self.bos_token)
                st += 1
            else:
                ret.append(self.unk_token)
                st += 1
        return "".join(ret)

    def _encode_unicode(self, token):
        # wrap unicode encoding into a helper function
        ids = []
        utf8_id = token.encode("utf-8")
        for _id in utf8_id:
            ids.append(self._special_encoder[self.byte_list[_id]])
        return ids

    def next_token(self, text):
        # fast next token matching
        token, token_id = self.tencoder.longest_prefix_item(text, (None, None))
        if token is None:
            token = text[0]
            token_ids = self._encode_unicode(token)
        else:
            token_ids = [token_id]
        return token, token_ids

class CPM9GModel():
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, n_device = 1):
        if not os.path.isdir(model_dir_path):
            print(f"Model directory {model_dir_path} does not exist")
            sys.exit(1)
        model_file = ""
        config_file = os.path.join(model_dir_path, "config.json")
        vocab_file = os.path.join(model_dir_path, "vocab.txt")
        for file_name in os.listdir(model_dir_path):
            if file_name.endswith(".ckpt") or file_name.endswith(".pt"):
                model_file = os.path.join(model_dir_path, file_name)
        
        self.ndev = n_device
        self.dev_ids = (c_uint * self.ndev)(*[i for i in range(self.ndev)])
        self.device_type = device
        
        _t0 = time.time()
        state_dict = {}
        if model_file.endswith(".pt"):
            state_dict = torch.load(model_file, weights_only=True)
        elif model_file.endswith(".ckpt"):
            state_dict = load_ckpt(model_file)
        else:
            raise ValueError("Unsupported model file format")
        self.tokenizer = CPM9GTokenizer(vocab_file)
        config = json.load(open(config_file))
        
        self.meta = LlamaMeta(
            dt_logits=DataType.INFINI_F16,
            dt_norm=DataType.INFINI_F16,
            dt_mat=DataType.INFINI_F16,
            nlayer=config.get("num_layers") or config.get("num_hidden_layers"),
            d=config.get("dim_model") or config.get("hidden_size"),
            nh=config.get("num_heads") or config.get("num_attention_heads"),
            nkvh=config.get("num_kv_heads") or config.get("num_key_value_heads"),
            dh=config.get("dim_head") or config.get("hidden_size") // config.get("num_attention_heads"),
            di=config.get("dim_ff") or config.get("intermediate_size"),
            dctx=4096,
            dvoc=config["vocab_size"],
            epsilon=config.get("eps") or config.get("rms_norm_eps"),
            theta=10000.0,
        )
        weights = LlamaWeights9G(state_dict, self.meta, self.ndev)
        
        _t1 = time.time()
        print(f"Load: {_t1 - _t0}")
    
        self.model_instance = lib.create_model(
            ctypes.byref(self.meta),
            ctypes.byref(weights),
            self.device_type,
            self.ndev,
            self.dev_ids,
        )
        _t2 = time.time()
        print(f"Create model: {_t2 - _t1}")
    
    def infer(self, input_content, max_steps, topp=1.0, topk=1, temperature=1.0):
        output_content = ""
        print(input_content, end="", flush=True)
        kv_cache = lib.create_kv_cache(self.model_instance)
        tokens = self.tokenizer.encode(input_content)
        ntok = len(tokens)
        nreq = 1
  
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
            output_str = self.tokenizer.decode(output_tokens)
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
    
    def apply_template(self, messages):
        complete_msg = ""
        for message in messages:
            if message["role"] == "user":
                complete_msg += "<用户>" +message["content"]
            elif message["role"] == "assistant":
                complete_msg += "<AI>" + message["content"]
            elif message["role"] == "system":
                complete_msg += "<用户>" + message["content"]
            else:
                print(f"Unknown role: {message['role']}")
                complete_msg += message["content"]
        return complete_msg + "<AI>"

def test():
    if len(sys.argv) < 3:
        print("Usage: python test_9G.py [--cpu | --cuda | --cambricon | --ascend | --maca] <path/to/model_dir> [n_device]")
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
    else:
        print("Usage: python test_9G.py [--cpu | --cuda | --cambricon | --ascend | --maca] <path/to/model_dir> [n_device]")
        sys.exit(1)
    
    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = CPM9GModel(model_path, device_type, ndev)
    model.infer("<用户>讲个长故事<AI>", 500)


if __name__ == '__main__':
    test()
