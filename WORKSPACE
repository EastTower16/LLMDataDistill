load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")



# Bazel Skylib.
http_archive(
    name = "bazel_skylib",  # 2022-11-16T18:29:32Z
    sha256 = "a22290c26d29d3ecca286466f7f295ac6cbe32c0a9da3a91176a90e0725e3649",
    strip_prefix = "bazel-skylib-5bfcb1a684550626ce138fe0fe8f5f702b3764c3",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/5bfcb1a684550626ce138fe0fe8f5f702b3764c3.zip"],
)



http_archive(
    name = "minhashcuda",
    # sha256 = "7c0101c68422aa038314e07842b2f83aa4c3d2b5520ebba350319277c4ad9c03",
    strip_prefix = "minhashcuda-a0d014aa31b6cdb26bb5cc2b11ccefe137b0193c",
    urls = ["https://github.com/koth/minhashcuda/archive/a0d014aa31b6cdb26bb5cc2b11ccefe137b0193c.tar.gz"],
    build_file ="@//thirdparty:minhashcuda.BUILD"
)

http_archive(
    name="nlohmann_json",
    #sha256 = "",
    strip_prefix = "json-6af826d0bdb55e4b69e3ad817576745335f243ca",
    urls = ["https://github.com/nlohmann/json/archive/6af826d0bdb55e4b69e3ad817576745335f243ca.zip"],
)


http_archive(
    name="tokme",
    sha256 = "b055f3a5b3db636277ea56ae2ed8c8313e95bd75897e95e7f0705588460543e1",
    strip_prefix = "tokme-41d6f49e2ac3bddd8116a0ff0c56b35f370a9fae",
    urls = ["https://github.com/koth/tokme/archive/41d6f49e2ac3bddd8116a0ff0c56b35f370a9fae.zip"],
)

http_archive(
    name="concurrentqueue",
    #sha256 = "",
    strip_prefix = "concurrentqueue-810f6213a2ee3bbd0c2ff647c28996cfff84df06",
    urls = ["https://github.com/koth/concurrentqueue/archive/810f6213a2ee3bbd0c2ff647c28996cfff84df06.zip"],
    build_file ="@//thirdparty:concurrentqueue.BUILD"
)


# http_archive(
#     name="abseil-cpp",
#     sha256 = "ea1d31db00eb37e607bfda17ffac09064670ddf05da067944c4766f517876390",
#     strip_prefix = "abseil-cpp-c2435f8342c2d0ed8101cb43adfd605fdc52dca2",
#     urls = ["https://github.com/abseil/abseil-cpp/archive/c2435f8342c2d0ed8101cb43adfd605fdc52dca2.zip"],
# )

http_archive(
    name="australis",
    sha256 = "4627ebc92d8135de11172cd91cb45e408fccb8b27e78d4debd89d75921bf845f",
    strip_prefix = "australis-bea893c8cf08fde4e09b7b2dd893f86b935bfd9d",
    urls = ["https://github.com/EastTower16/australis/archive/bea893c8cf08fde4e09b7b2dd893f86b935bfd9d.zip"],
)


git_repository(
     name = "jax",
     commit = "c3e242700872c2f7e098a07f3911ee6d2de8132c",
     remote = "https://github.com/google/jax.git",
)

http_archive(
    name = "sentencepiece",
    sha256 = "0c28dfd2fad9f215ea276f60e62c41aca7f0ad48fd4bc072dd79180f59b44ec2",
    strip_prefix = "sentencepiece-cf093775361a08dbe8d2a5ec98f548b25d7d6e37",
    urls = [
        "https://github.com/EastTower16/sentencepiece/archive/cf093775361a08dbe8d2a5ec98f548b25d7d6e37.tar.gz",
    ],
)
# local_repository(
#    name = "sentencepiece",
#    path = "/f/workspace/sentencepiece",
# )

http_archive(
    name = "xla",
    sha256 = "4ec16aff3862c5a243db956ce558d7a62eb79f5e20747b0e80802a3b0d12e419",
    strip_prefix = "xla-12de6ec958419b57be248d0acd2d9f757e71748c",
    urls = [
        "https://github.com/openxla/xla/archive/12de6ec958419b57be248d0acd2d9f757e71748c.tar.gz",
    ],
)
load("@xla//third_party/gpus:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "local_config_cuda")
# For development, one can use a local TF repository instead.
# local_repository(
#    name = "org_tensorflow",
#    path = "tensorflow",
# )


load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()

load("@jax//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()
