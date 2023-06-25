# load("@rules_cuda//cuda:defs.bzl", "cuda_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
package(default_visibility = ["//visibility:public"])

cuda_library(
    name="kernel",
    srcs=[
        'wrappers.h',
        'kernel.cu.cc',
       
    ],
    hdrs=[
        'private.h',
        'minhashcuda.h',
    ],
    deps=[
        "@local_config_cuda//cuda:cuda",
    ],
)

cc_library(
    name="minhashcuda",
    srcs=[
        'minhashcuda.cc',
    ],
    defines=[
        "CUDA_ARCH=86",
    ],
    hdrs=[
    ],
    deps=[
        ":kernel",
    ]
)
