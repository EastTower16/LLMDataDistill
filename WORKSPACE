load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cuda",
    #sha256 = "{sha256_to_replace}",
    strip_prefix = "rules_cuda-1e9954093c7d789c628ddf052035f725249c122f",
    urls = ["https://github.com/bazel-contrib/rules_cuda/archive/1e9954093c7d789c628ddf052035f725249c122f.tar.gz"],
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

register_detected_cuda_toolchains()
