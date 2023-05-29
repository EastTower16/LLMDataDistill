#include "dedup/hash_dumper.h"

#include <stdlib.h>

#include <algorithm>
#include <unordered_map>

#include "absl/log/log.h"
#include "dedup/page_featurizer.h"
#include "minhashcuda.h"

namespace pd {

const static uint32_t kVocabDim = 51200;
const static uint16_t kMinHashBins = 128;
const static uint32_t kDevice = 0;
const static int kBatchSize = 10240;
HashDumper::HashDumper(const std::string& outFile) : out_(outFile) {
  featurizer_.reset(new PageFeaturizer());
  indptr_.push_back(0);
  uint32_t seed = static_cast<uint32_t>(time(NULL));
  MHCUDAResult result = mhcudaSuccess;
  minhashptr_ =
      mhcuda_init(kVocabDim, kMinHashBins, seed, 0, kDevice, 2, &result);
  if (result != mhcudaSuccess) {
    LOG(FATAL) << "error mhcuda init :" << result;
    minhashptr_ = nullptr;
  }
  result_buffer_ = new uint32_t[kBatchSize * kMinHashBins * 2];
}
HashDumper::~HashDumper() {
  if (result_buffer_ != nullptr) {
    delete[] result_buffer_;
    result_buffer_ = nullptr;
  }
  if (minhashptr_ != nullptr) {
    MinhashCudaGenerator* gen =
        reinterpret_cast<MinhashCudaGenerator*>(minhashptr_);
    mhcuda_fini(gen);
    minhashptr_ = nullptr;
  }
}

bool HashDumper::process(Page& page) {
  if (!featurizer_->Featurize(page)) {
    return false;
  }

  int preCount = indptr_.back();

  std::vector<std::pair<int, float>> features(page.features.begin(),
                                              page.features.end());
  std::sort(features.begin(), features.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
              return a.first < b.first;
            });
  int num_feature = features.size();
  indptr_.push_back(preCount + num_feature);
  for (int i = 0; i < num_feature; i++) {
    int col = features[i].first;
    indices_.emplace_back(col);
    data_.emplace_back(features[i].second);
  }
  buffered_ += 1;
  if (buffered_ >= 10240) {
    // TODO
    MinhashCudaGenerator* gen =
        reinterpret_cast<MinhashCudaGenerator*>(minhashptr_);
    MHCUDAResult result = mhcudaSuccess;
    auto startTime = std::chrono::high_resolution_clock::now();
    result = mhcuda_calc(gen, data_.data(), indices_.data(), indptr_.data(),
                         indptr_.size() - 1, result_buffer_);
    auto endTime = std::chrono::high_resolution_clock::now();
    if (result != mhcudaSuccess) {
      LOG(FATAL) << "error mhcuda calc :" << result;
      return false;
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        endTime - startTime)
                        .count();
    LOG(INFO) << "minhash calc time: " << duration << " ms";
    buffered_ = 0;
    indptr_.clear();
    indptr_.push_back(0);
    indices_.clear();
    data_.clear();
  }
  return true;
}
}  // namespace pd
