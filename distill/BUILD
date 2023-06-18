package(default_visibility = ["//visibility:public"])

cc_binary(
    name ="gen_ex_main",
    srcs=[
        "gen_ex_main.cc",
    ],
    deps=[
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/strings",
        "@tokme//:tokme",
    ]
)
