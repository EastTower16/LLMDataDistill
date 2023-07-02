#include "dedup/hash_dumper.h"

#include <stdlib.h>

#include <algorithm>
#include <unordered_map>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "dedup/lsh_index.h"
#include "dedup/page_featurizer.h"
#include "minhashcuda.h"

namespace pd {

const static uint32_t kVocabDim = 51200;
const static uint16_t kMinHashBins = 128;
const static uint32_t kDevice = 0;
const static int kBatchSize = 10240;
const static int kBandNum = 8;
const static int kNumHashSlot = 99829;
const static float kDedupThreshold = 0.85;
HashDumper::HashDumper(const std::string& tokenizerPath,
                       const std::string& outFile) {
  out_ = fopen(outFile.c_str(), "w");
  featurizer_.reset(new PageFeaturizer());
  CHECK(featurizer_->Init(tokenizerPath));
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
  indexer_.reset(new pd::LshIndex(kBandNum, kNumHashSlot));
}
HashDumper::~HashDumper() {
  if (result_buffer_ != nullptr) {
    delete[] result_buffer_;
    result_buffer_ = nullptr;
  }
  if (out_ != nullptr) {
    fclose(out_);
    out_ = nullptr;
  }
  if (minhashptr_ != nullptr) {
    MinhashCudaGenerator* gen =
        reinterpret_cast<MinhashCudaGenerator*>(minhashptr_);
    mhcuda_fini(gen);
    minhashptr_ = nullptr;
  }
}

bool HashDumper::doBatch() {
  if (idkeys_.empty()) {
    LOG(INFO) << "idkey is empty,...";
    return true;
  }
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
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime)
          .count();
  LOG(INFO) << "minhash calc time: " << duration << " ms";
  for (int i = 0; i < static_cast<int>(indptr_.size() - 1); i++) {
    size_t offset = 2 * kMinHashBins * i;
    std::string key = idkeys_[i];
    pd::WeightedMinHash wmh;
    for (int k = 0; k < kMinHashBins; k++) {
      wmh.ks.push_back(result_buffer_[2 * k + offset]);
      wmh.ts.push_back(result_buffer_[2 * k + offset + 1]);
    }
    std::vector<int> hashvals;
    std::vector<std::string> cands;
    std::vector<float> sims;
    bool gotDup = false;
    if (indexer_->query(wmh, hashvals, cands, sims)) {
      for (size_t i = 0; i < cands.size(); i++) {
        if (sims[i] >= kDedupThreshold) {
          gotDup = true;
          // LOG(INFO)<<"key:["<<key<<"] dup to:["<<cands[i]<<"] with
          // sim:["<<sims[i]<<"]";
          fprintf(out_, "%s\n", key.c_str());
          break;
        }
      }
      if (!gotDup) {
        indexer_->addWeightedMinHash(key, wmh, hashvals);
      }
    }
  }
  buffered_ = 0;
  indptr_.clear();
  indptr_.push_back(0);
  indices_.clear();
  data_.clear();
  idkeys_.clear();
  return true;
}
bool HashDumper::process(Page& page) {
  if (!featurizer_->Featurize(page)) {
    LOG(INFO) << "feature error....:" << page.title;
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
  idkeys_.push_back(page.idkey);
  for (int i = 0; i < num_feature; i++) {
    int col = features[i].first;
    indices_.emplace_back(col);
    data_.emplace_back(features[i].second);
  }
  // LOG(INFO)<<"process:"<<page.title;
  buffered_ += 1;
  if (buffered_ >= 10240) {
    LOG(INFO) << "will do one batch!";
    if (!doBatch()) {
      return false;
    }
  }
  return true;
}
}  // namespace pd
