"""Bounded one-shot Zstandard through its stable system-library API.

API: https://github.com/facebook/zstd/blob/dev/lib/zstd.h
No numerical arrays are converted, quantized or interpreted by this codec.
"""
import ctypes
import ctypes.util
from functools import lru_cache

MAX_FRAME_BYTES = 16 * 1024 ** 2


@lru_cache(maxsize=1)
def library():
    path = ctypes.util.find_library('zstd')
    if not path:
        raise RuntimeError('Indexed observations require the system libzstd library')
    lib = ctypes.CDLL(path)
    signatures = {
        'ZSTD_compressBound': ([ctypes.c_size_t], ctypes.c_size_t),
        'ZSTD_compress': ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p,
                           ctypes.c_size_t, ctypes.c_int], ctypes.c_size_t),
        'ZSTD_decompress': ([ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p,
                             ctypes.c_size_t], ctypes.c_size_t),
        'ZSTD_getFrameContentSize': ([ctypes.c_char_p, ctypes.c_size_t], ctypes.c_ulonglong),
        'ZSTD_isError': ([ctypes.c_size_t], ctypes.c_uint),
    }
    for name, (arguments, result) in signatures.items():
        operation = getattr(lib, name)
        operation.argtypes, operation.restype = arguments, result
    return lib


def compress(body, *, level=1):
    if not 0 < len(body) <= MAX_FRAME_BYTES or not 1 <= level <= 9:
        raise ValueError('Frame size or compression level is outside the supported bounds')
    lib = library()
    capacity = lib.ZSTD_compressBound(len(body))
    if lib.ZSTD_isError(capacity):
        raise ValueError('Zstandard compression capacity failed')
    target = ctypes.create_string_buffer(capacity)
    size = lib.ZSTD_compress(target, capacity, body, len(body), level)
    if lib.ZSTD_isError(size):
        raise ValueError('Zstandard frame compression failed')
    return target.raw[:size]


def decompress(body):
    lib = library()
    size = lib.ZSTD_getFrameContentSize(body, len(body))
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ValueError('Invalid or oversized Zstandard frame')
    target = ctypes.create_string_buffer(size)
    result = lib.ZSTD_decompress(target, size, body, len(body))
    if lib.ZSTD_isError(result) or result != size:
        raise ValueError('Zstandard frame decompression failed')
    return target.raw[:size]
