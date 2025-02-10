
import ctypes
from ctypes import c_uint, c_float, c_void_p, POINTER
import os

class DataType(ctypes.c_int):
    INFINI_BYTE = 0
    INFINI_I8 = 1
    INFINI_I16 = 2
    INFINI_I32 = 3
    INFINI_I64 = 4
    INFINI_U8 = 5
    INFINI_U16 = 6
    INFINI_U32 = 7
    INFINI_U64 = 8
    INFINI_F8 = 9
    INFINI_F16 = 10
    INFINI_F32 = 11
    INFINI_F64 = 12
    INFINI_BF16 = 13
    INFINI_BOOL = 14

class DeviceType(ctypes.c_int):
    DEVICE_TYPE_CPU = 0
    DEVICE_TYPE_CUDA = 1
    DEVICE_TYPE_CAMBRICON = 2
    DEVICE_TYPE_ASCEND = 3
    DEVICE_TYPE_METAX = 4
    DEVICE_TYPE_MTHREADS = 5

class LlamaMeta(ctypes.Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("dt_norm", DataType),
        ("dt_mat", DataType),
        ("nlayer", c_uint),
        ("d", c_uint),
        ("nh", c_uint),
        ("nkvh", c_uint),
        ("dh", c_uint),
        ("di", c_uint),
        ("dctx", c_uint),
        ("dvoc", c_uint),
        ("epsilon", c_float),
        ("theta", c_float),
    ]

# Define the LlamaWeights struct
class LlamaWeights(ctypes.Structure):
    _fields_ = [
        ("nlayer", c_uint),
        ("input_embd", c_void_p),
        ("output_norm", c_void_p),
        ("output_embd", c_void_p),
        ("attn_norm", POINTER(c_void_p)),
        ("attn_qkv", POINTER(c_void_p)),
        ("attn_o", POINTER(c_void_p)),
        ("ffn_norm", POINTER(c_void_p)),
        ("ffn_gate_up", POINTER(c_void_p)),
        ("ffn_down", POINTER(c_void_p)),
    ]

class Model(ctypes.Structure):
    pass


class KVCache(ctypes.Structure):
    pass


def open_library():    
    lib_path = os.path.join(os.environ.get("INFINI_ROOT"), "lib", "libinfiniinfer.so")
    lib = ctypes.CDLL(lib_path)
    lib.create_model.restype = POINTER(Model)
    lib.create_model.argtypes = [
        POINTER(LlamaMeta),  # LlamaMeta const *
        POINTER(LlamaWeights),  # LlamaWeights const *
        DeviceType,  # DeviceType
        c_uint,  # unsigned int ndev
        POINTER(c_uint),  # unsigned int const *dev_ids
    ]

    lib.create_kv_cache.restype = POINTER(KVCache)
    lib.drop_kv_cache.argtypes= [ctypes.POINTER(Model), POINTER(KVCache)]
    lib.infer.restype = None
    lib.infer.argtypes = [
        ctypes.POINTER(Model),  # struct Model const *
        c_uint,  # unsigned int ntok
        POINTER(c_uint),  # unsigned int const *tokens
        c_uint,  # unsigned int nreq
        POINTER(c_uint),  # unsigned int const *req_lens
        POINTER(c_uint),  # unsigned int const *req_pos
        POINTER(POINTER(KVCache)),  # struct KVCache **kv_caches
        POINTER(c_uint),  # unsigned int *ans
        c_float,  # float temperature
        c_uint,  # unsigned int topk
        c_float,  # float topp
    ]
    
    return lib
