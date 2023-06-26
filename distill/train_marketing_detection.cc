#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "australis/australis.h"
#include "australis/petri.h"
#include "distill/marketing_detection.h"
#include "youtokentome/cpp/bpe.h"

ABSL_FLAG(std::string, train_data_path, "data/train.txt", "training data path");
ABSL_FLAG(std::string, test_data_path, "data/test.txt", "the test data path");
ABSL_FLAG(std::string, tokenizer_path, "vocab/tokme.model",
          "the tokenizer model path");
ABSL_FLAG(std::string, out_model_path, "md_model.jax", "model output path");
ABSL_FLAG(int, max_epoch, 50, "the maximum epoch number  of training");

const static int kBatchSize = 128;
const static int kMaxLen = 2048;

static std::vector<int> kInputBuffer(kBatchSize* kMaxLen, 0);
static std::vector<int> kTrainBuffer(kBatchSize* kMaxLen, 0);
static std::vector<int> kTargetBuffer(kBatchSize, 0);

struct TrainEx {
  int marketing;
  std::vector<int> data;
};
absl::StatusOr<std::tuple<aux::PTree, aux::PTree>> Unpack2Tuple(
    absl::StatusOr<aux::PTree> input) {
  auto tmp = *aux::PTree::DestructureTuple(std::move(input));
  if (tmp.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat("Wrong size: ", tmp.size()));
  }
  return std::tuple<aux::PTree, aux::PTree>(std::move(tmp[0]),
                                            std::move(tmp[1]));
}

absl::StatusOr<std::tuple<aux::PTree, aux::PTree, aux::PTree>> Unpack3Tuple(
    absl::StatusOr<aux::PTree> input) {
  auto tmp = *aux::PTree::DestructureTuple(std::move(input));
  if (tmp.size() != 3) {
    return absl::InvalidArgumentError(absl::StrCat("Wrong size: ", tmp.size()));
  }
  return std::tuple<aux::PTree, aux::PTree, aux::PTree>(
      std::move(tmp[0]), std::move(tmp[1]), std::move(tmp[2]));
}

static void convertLineToEx(const std::string& line,
                            const vkcom::BaseEncoder* encoder, int maxLength,
                            TrainEx& ex) {
  std::vector<std::string> vs = absl::StrSplit(line, absl::ByChar('\t'));
  int quality = 0, saleGrade = 0;
  absl::SimpleAtoi(vs[0], &quality);
  absl::SimpleAtoi(vs[1], &saleGrade);
  ex.marketing = saleGrade >= 30.0 ? 1 : 0;
  std::vector<std::vector<int>> ids;
  auto status = encoder->encode_as_ids({vs[2]}, &ids);
  if (!status.ok() || ids.empty()) {
    LOG(ERROR) << "encode_as_ids error!";
    return;
  }
  std::vector<int> tokens = ids[0];
  int textLen = tokens.size();
  if (textLen > maxLength) {
    tokens = std::vector<int>(tokens.begin(), tokens.begin() + maxLength);
  }
  ex.data = tokens;
}
void loadTestDataset(const std::string& path, const vkcom::BaseEncoder* encoder,
                     std::vector<TrainEx>& testData) {
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    TrainEx ex;
    convertLineToEx(line, encoder, kMaxLen, ex);
    testData.push_back(ex);
  }
}

void fillTrainData(const std::vector<TrainEx>& inputData) {
  CHECK_EQ(inputData.size(), kBatchSize);
  for (int k = 0; k < kBatchSize; k++) {
    for (int j = 0; j < inputData[k].data.size(); j++) {
      kTrainBuffer[k * kMaxLen + j] = inputData[k].data[j];
    }
    for (int j = inputData[k].data.size(); j < kMaxLen; j++) {
      kTrainBuffer[k * kMaxLen + j] = 0;
    }
    kTargetBuffer[k] = inputData[k].marketing;
  }
}
float evalTestDataset(const std::vector<TrainEx>& testData,
                      pd::FlaxBatchServing& batchServing, aux::Device dev,
                      const aux::PTree& param, int maxLength) {
  int nn = (testData.size() - 1) / kBatchSize + 1;
  float sumDiff = 0;
  for (int i = 0; i < nn; i++) {
    int start = i * kBatchSize;
    int end = std::min((i + 1) * kBatchSize, static_cast<int>(testData.size()));
    for (int k = start; k < end; k++) {
      int off = k - start;
      for (int j = 0; j < testData[k].data.size(); j++) {
        kInputBuffer[off * maxLength + j] = testData[k].data[j];
      }
      for (int j = testData[k].data.size(); j < maxLength; j++) {
        kInputBuffer[off * maxLength + j] = 0;
      }
    }
    for (int k = end; k < (i + 1) * kBatchSize; k++) {
      int off = k - start;
      for (int j = 0; j < maxLength; j++) {
        kInputBuffer[off * maxLength + j] = 0;
      }
    }
    auto x = *aux::PTree::BufferRN<int>(absl::Span<int>(kInputBuffer),
                                        {kBatchSize, kMaxLen}, dev);
    auto result = *(batchServing(param, x));
    auto resultLiteral = *result.ToArray();
    for (int k = start; k < end; k++) {
      const TrainEx& ex = testData[k];
      int predy = resultLiteral.data<float>()[k - start] > 0 ? 1 : 0;
      sumDiff += predy == ex.marketing ? 1 : 0;
    }
  }
  LOG(INFO) << "test accuracy: " << sumDiff / testData.size();
  return sumDiff / testData.size();
}
bool saveModel(const std::string& path,
               const std::vector<const aux::DeviceArray*>& params) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  int numBuf = params.size();
  file.write(reinterpret_cast<const char*>(&numBuf), sizeof(int));
  for (int i = 0; i < numBuf; i++) {
    const aux::DeviceArray* p = params[i];
    auto ia = *p->ToArrays();
    auto dataArr = ia.data()->data<float>();
    int numData = dataArr.size();
    file.write(reinterpret_cast<const char*>(&numData), sizeof(int));
    for (int j = 0; j < numData; j++) {
      file.write(reinterpret_cast<const char*>(&dataArr[j]), sizeof(float));
    }
  }
  file.close();
  return true;
}
int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage("Train MarketingDetection Main");
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
  int maxEpoch = absl::GetFlag(FLAGS_max_epoch);
  std::vector<TrainEx> testData;
  loadTestDataset(absl::GetFlag(FLAGS_test_data_path), encoder.get(), testData);
  LOG(INFO) << "load " << testData.size() << " test examples";
  auto client = *aux::Client::GetDefault();
  aux::Device dev = client.LocalDevices()[0];
  auto init_fn = *pd::FlaxInit::Load(client);
  auto optimizer_step_fn = *pd::FlaxOptimizerStep::Load(client);
  auto batchServing = *pd::FlaxBatchServing::Load(client);

  auto [params, opt_state] = *Unpack2Tuple(init_fn());
  LOG(INFO) << "inited model , num buffer in weights:" << params.num_buffers();
  float bestAccuracy =
      evalTestDataset(testData, batchServing, dev, params, kMaxLen);
  std::string line;
  std::vector<TrainEx> trainData;
  uint32_t step = 0;
  for (int epoch = 0; epoch < maxEpoch; epoch++) {
    LOG(INFO) << "begin epoch " << epoch;
    std::ifstream file(absl::GetFlag(FLAGS_train_data_path));
    aux::PTree lossPT;
    while (std::getline(file, line)) {
      TrainEx ex;
      convertLineToEx(line, encoder.get(), kMaxLen, ex);
      trainData.push_back(ex);
      if (trainData.size() == kBatchSize) {
        fillTrainData(trainData);
        auto x = *aux::PTree::BufferRN<int>(absl::Span<int>(kTrainBuffer),
                                            {kBatchSize, kMaxLen}, dev);
        auto y = *aux::PTree::BufferRN<int>(absl::Span<int>(kTargetBuffer),
                                            {kBatchSize, 1}, dev);

        std::tie(params, opt_state, lossPT) =
            *Unpack3Tuple(optimizer_step_fn(params, opt_state, x, y));
        step += 1;
        auto resultLiteral = *lossPT.ToArray();
        if (step % 20 == 0) {
          LOG(INFO) << "Step:" << step
                    << ",Loss: " << resultLiteral.data<float>()[0];
        }
        trainData.clear();
      }
    }
    float accuracy =
        evalTestDataset(testData, batchServing, dev, params, kMaxLen);
    if (accuracy > bestAccuracy) {
      bestAccuracy = accuracy;
      std::vector<const aux::DeviceArray*> flattened_params;
      CHECK(params.FlattenTo(&flattened_params).ok());
      saveModel(absl::GetFlag(FLAGS_out_model_path), flattened_params);
      LOG(INFO) << "new best model:" << bestAccuracy;
    }
  }
  LOG(INFO) << "Finished, best accuracy:" << bestAccuracy;
  return 0;
}
