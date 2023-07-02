#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "common/page_producer.h"
#include "xla/pjrt/pjrt_client.h"
#include "australis/australis.h"
#include "australis/petri.h"

#include "distill/marketing_detection.h"
#include "youtokentome/cpp/bpe.h"

ABSL_FLAG(std::string, wudao_dir, "/g/wudao", "wudao dataset dir");
ABSL_FLAG(std::string, distilled_output_path, "/g/distilled_pages.txt",
          "distilled output path");
ABSL_FLAG(std::string, tokenizer_path, "vocab/tokme.model",
          "the tokenizer model path");
ABSL_FLAG(std::string, model_path, "md_model.jax",
          "the marketing detection model path");

namespace fs = std::filesystem;
const static int kMaxLen = 2048;
void traverseDirectory(const std::string& path,
                       std::vector<std::string>& pathList) {
  for (const auto& entry : fs::directory_iterator(path)) {
    if (entry.is_directory()) {
      fs::path newdir = fs::path(path) / entry.path();
      traverseDirectory(newdir.string(), pathList);
    } else if (entry.is_regular_file() && entry.path().string().size() > 5 &&
               entry.path().string().substr(entry.path().string().size() - 5) ==
                   std::string(".json")) {
      fs::path newpath = fs::path(path) / entry.path();
      pathList.emplace_back(newpath.string());
    }
  }
}

struct DistillClient {
  pd::FlaxServing& serving;
  aux::PTree params;
  aux::Device device;
  FILE* output=nullptr;
  explicit DistillClient(aux::Device dev, pd::FlaxServing& serving_, const std::string& outputPath):serving(serving_),device(dev) {
    output = fopen(outputPath.c_str(), "w");
  }
  ~DistillClient(){
    if(output != nullptr){
      fclose(output);
      output = nullptr;
    }
    
  }
  bool LoadModel(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
      return false;
    }

    int numBuf = 0;
    file.read(reinterpret_cast<char*>(&numBuf), sizeof(int));
    std::vector<aux::PTree> modelBuffers;
    for (int i = 0; i < numBuf; i++) {
      int ndim = 0;
      std::vector<long int> dims;
      int numData = 0;
      file.read(reinterpret_cast<char*>(&ndim), sizeof(int));
      for (int k = 0; k < ndim; k++) {
        int dim = 0;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        dims.push_back(dim);
      }
      file.read(reinterpret_cast<char*>(&numData), sizeof(int));
      std::vector<float> buffer(numData, 0);
      file.read(reinterpret_cast<char*>(buffer.data()),
                numData * sizeof(float));

      modelBuffers.push_back(
          *aux::PTree::BufferRN<float>(buffer,dims, device));
    }
    params = aux::PTree::Tuple(std::move(modelBuffers));
    file.close();
    return true;
  }
  bool Predict(const vkcom::BaseEncoder* encoder, int maxLength,
                const pd::Page& p, float& marketingScore) {
    std::vector<std::vector<int>> ids;
    auto status = encoder->encode_as_ids({p.content}, &ids);
    if (!status.ok() || ids.empty()) {
      LOG(ERROR) << "encode_as_ids error!";
      return false;
    }
    std::vector<int> tokens = ids[0];
    int textLen = tokens.size();
    if (textLen > maxLength) {
      tokens = std::vector<int>(tokens.begin(), tokens.begin() + maxLength);
    }else{
      for(int i=textLen;i<maxLength;i++){
        tokens.push_back(0);
      }
    }
    auto x = *aux::PTree::BufferRN<int>(absl::Span<int>(tokens),
                                        {1, kMaxLen}, device);
    auto result = *(serving(params, x));
    auto resultLiteral = *result.ToArray();
    float rawy=resultLiteral.data<float>()[0];
    marketingScore = 1.0/(1.0+exp(0-rawy));
    if(marketingScore>=0.6){
      fprintf(output,"%s\t%.4f\n",p.idkey.c_str(),marketingScore);
    }
    return true;
  }
};

void doWork(pd::PageProducer* producer,vkcom::BaseEncoder* encoder, DistillClient* distillClient) {
  pd::Page p;
  uint64_t count = 0;
  producer->takeOnePage(p, true);
  while (p.idkey != pd::PageProducer::EOFPageHashKey) {
    float score=0;
    CHECK(distillClient->Predict(encoder,kMaxLen,p, score));
    count += 1;
    if (count % 20000 == 0) {
      LOG(INFO) << count << " pages processed!!";
    }
    producer->takeOnePage(p, true);
  }
  LOG(INFO) << count << " total pages processed!!";
}

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage("Dedup Main");
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  std::unique_ptr<vkcom::BaseEncoder> encoder;
  vkcom::Status status;
  encoder.reset(
      new vkcom::BaseEncoder(absl::GetFlag(FLAGS_tokenizer_path), 1, &status));
  if (!status.ok()) {
    LOG(ERROR) << "init tokenizer error,path:"
               << absl::GetFlag(FLAGS_tokenizer_path);
    return -1;
  }
  pd::PageProducer pageProducer;
  std::vector<std::string> pathList;
  traverseDirectory(absl::GetFlag(FLAGS_wudao_dir), pathList);
  LOG(INFO) << "got " << pathList.size() << " paths!";
  auto client = *aux::Client::GetDefault();
  aux::Device dev = client.LocalDevices()[0];
  auto serving = *pd::FlaxServing::Load(client);
  DistillClient distillClient(dev,serving,absl::GetFlag(FLAGS_distilled_output_path));
  pageProducer.initByFileList(pathList);
  CHECK(distillClient.LoadModel(absl::GetFlag(FLAGS_model_path)));
  doWork(&pageProducer,encoder.get(), &distillClient);
  pageProducer.shutdown();
  return 0;
}
