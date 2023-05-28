load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name="kernel",
    srcs=[
        'wrappers.h',
        'kernel.cu',
       
    ],
    hdrs=[
        'private.h',
        'minhashcuda.h',
    ]
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
        "@local_cuda//:curand",
    ]
)
