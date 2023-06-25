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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
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
        "@com_google_absl//absl/log",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/flags:parse",
        "@minhashcuda//:minhashcuda",
    ],
    linkopts = [
        '-lm',
    ],
)


cc_binary(
    name ="lsh_main",
    srcs=[
        "lsh_main.cc",
    ],
    deps=[
        ":lsh_index",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
    ],
    linkopts = [
        '-lm',
    ],
)