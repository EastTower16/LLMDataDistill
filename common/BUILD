package(default_visibility = ["//visibility:public"])

cc_library(
    name="page",
    hdrs=[
        "page.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "page_producer",
    hdrs =[
        "page_producer.h"
    ],
    srcs=[
        "page_producer.cc",
    ],
    deps=[
        ":page",
        "@concurrentqueue//:concurrentqueue",
        "@nlohmann_json//:json",
        "@abseil-cpp//absl/log",
    ]
)
