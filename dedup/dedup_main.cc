#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "common/page_producer.h"
#include "dedup/hash_dumper.h"

ABSL_FLAG(std::string, wudao_dir, "/g/wudao", "wudao dataset dir");
ABSL_FLAG(std::string, output_path, "/g/duplicate_keys.txt",
          "debup output path");
ABSL_FLAG(std::string, tokenizer_path, "vocab/tokme.model", "the tokenizer model path");

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
  pd::Page p;
  uint64_t count = 0;
  producer->takeOnePage(p, true);
  while(p.idkey != pd::PageProducer::EOFPageHashKey){
    hasher->process(p);
    count +=1;
    if(count % 20000 == 0){
      LOG(INFO) << count << " pages processed!!";
    }
    producer->takeOnePage(p, true); 
  }
  hasher->doBatch();
  LOG(INFO) << count << " total pages processed!!";
}

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage("Dedup Main");
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  pd::PageProducer pageProducer;
  pd::HashDumper hashDumper(absl::GetFlag(FLAGS_tokenizer_path),absl::GetFlag(FLAGS_output_path));
  std::vector<std::string> pathList;
  traverseDirectory(absl::GetFlag(FLAGS_wudao_dir), pathList);
  LOG(INFO) << "got " << pathList.size() << " paths!";
  pageProducer.initByFileList(pathList);
  doWork(&pageProducer, &hashDumper);
  pageProducer.shutdown();
  return 0;
}

