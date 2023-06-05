cc_library(
    name="page_featurizer",
    srcs=[
        "page_featurizer.cc",
    ],
    hdrs=[
        "page_featurizer.h",
    ],
    deps=[
        "//common:page",
        "@tokme//:tokme",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name="hash_dumper",
    srcs=[
        "hash_dumper.cc",
    ],
    hdrs=[
        "hash_dumper.h",
    ],
    deps=[
        ":page_featurizer",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@minhashcuda//:minhashcuda"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name="lsh_index",
    srcs=[
        "lsh_index.cc",
    ],
    hdrs=[
        "lsh_index.h",
    ],
    deps=[
        "@abseil-cpp//absl/log",
    ],
    visibility = ["//visibility:public"],
)


cc_binary(
    name ="dedup_main",
    srcs=[
        "dedup_main.cc",
    ],
    deps=[
        "//common:page_producer",
        ":page_featurizer",
        ":hash_dumper",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/flags:parse",
        "@minhashcuda//:minhashcuda",
    ]
)


cc_binary(
    name ="lsh_main",
    srcs=[
        "lsh_main.cc",
    ],
    deps=[
        ":lsh_index",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/strings",
    ]
)