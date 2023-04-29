load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cuda",
    sha256 = "404efdc116e94a28c4191cb294c4c818c2b2c132f73f7970f9c9326db9e7117e",
    strip_prefix = "rules_cuda-1e9954093c7d789c628ddf052035f725249c122f",
    urls = ["https://github.com/bazel-contrib/rules_cuda/archive/1e9954093c7d789c628ddf052035f725249c122f.tar.gz"],
)

load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")

rules_cuda_dependencies()

register_detected_cuda_toolchains()


http_archive(
    name = "minhashcuda",
    #sha256 = "",
    strip_prefix = "minhashcuda-d057b0769ef983aa1315ca8d78be6b6f67b380ae",
    urls = ["https://github.com/src-d/minhashcuda/archive/d057b0769ef983aa1315ca8d78be6b6f67b380ae.tar.gz"],
    build_file ="@//thirdparty:minhashcuda.BUILD"
)
