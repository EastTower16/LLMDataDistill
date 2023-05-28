#ifndef PD_DEDUP_FEATURE_DUMPER_H_
#define PD_DEDUP_FEATURE_DUMPER_H_
#include <memory>
#include <string>
#include <fstream>

#include "common/page.h"

namespace pd {
class PageFeaturizer;
class HashDumper {
 public:
  HashDumper(const std::string& outFile);
  virtual ~HashDumper();
  virtual bool process(Page& page);
  
 private:
   std::unique_ptr<PageFeaturizer> featurizer_;
   std::ofstream out_;
   std::vector<int> indptr_;
   std::vector<int> indices_;
   std::vector<float> data_;
   int buffered_;
   void* minhashptr_;
};
}  // namespace pd

#endif  // PD_DEDUP_FEATURE_DUMPER_H_

