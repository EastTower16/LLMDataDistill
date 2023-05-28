#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common/page_producer.h"
#include "dedup/hash_dumper.h"

ABSL_FLAG(std::string, wudao_dir, "/g/wudao", "wudao dataset dir");
ABSL_FLAG(std::string, output_path, "/g/wudao_features.txt",
          "feature output path");

namespace fs = std::filesystem;

void traverseDirectory(const std::string& path,
                       std::vector<std::string>& pathList) {
  for (const auto& entry : fs::directory_iterator(path)) {
    if (entry.is_directory()) {
      fs::path newdir = fs::path(path) / entry.path();
      // 如果是子目录，则递归遍历
      traverseDirectory(newdir.string(), pathList);
    } else if (entry.is_regular_file() && entry.path().string().size() > 5 &&
               entry.path().string().substr(entry.path().string().size() - 5) ==
                   std::string(".json")) {
      fs::path newpath = fs::path(path) / entry.path();
      pathList.emplace_back(newpath.string());
    }
  }
}

void doWork(pd::PageProducer* producer, pd::HashDumper* hasher){
  
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  pd::PageProducer pageProducer;
  pd::HashDumper hashDumper(absl::GetFlag(FLAGS_output_path));
  std::vector<std::string> pathList;
  traverseDirectory(absl::GetFlag(FLAGS_wudao_dir), pathList);
  LOG(INFO) << "got " << pathList.size() << " paths!";
  pageProducer.initByFileList(pathList);
  std::this_thread::sleep_for(std::chrono::seconds(20));
  pageProducer.shutdown();
  return 0;
}
