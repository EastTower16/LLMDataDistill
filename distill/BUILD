package(default_visibility = ["//visibility:public"])

load("@australis//:australis.bzl", "australis")

licenses(["notice"])

australis(
    name = "marketing_detection",
    cc_namespace = "pd",
    py_deps = [],  # Internal flax deps
)

cc_binary(
    name ="gen_ex_main",
    srcs=[
        "gen_ex_main.cc",
    ],
    deps=[
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@tokme//:tokme",
    ],
    linkopts = [
        '-lm',
    ],
)



cc_binary(
    name ="train_marketing_detection",
    srcs=[
        "train_marketing_detection.cc",
    ],
    linkstatic = 1,
    deps=[
        ":marketing_detection_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@australis//australis:cpu_support",
        "@australis//australis:gpu_support",
        "@australis//australis:petri",
        "@australis//australis",
        "@sentencepiece//src:sentencepiece_processor",

    ],
    linkopts = [
        '-lm',
    ],
)


cc_binary(
    name ="distill_page_main",
    srcs=[
        "distill_page_main.cc",
    ],
    linkstatic = 1,
    deps=[
        ":marketing_detection_cc",
        "//common:page_producer",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/strings",
        "@australis//australis:cpu_support",
        "@australis//australis:gpu_support",
        "@australis//australis:petri",
        "@australis//australis",
        "@sentencepiece//src:sentencepiece_processor",

    ],
    linkopts = [
        '-lm',
    ],
)
