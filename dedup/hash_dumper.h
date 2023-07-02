#ifndef PD_DEDUP_FEATURE_DUMPER_H_
#define PD_DEDUP_FEATURE_DUMPER_H_
#include <memory>
#include <string>
#include <fstream>

#include "common/page.h"

namespace pd {
class PageFeaturizer;
class LshIndex;
class HashDumper {
 public:
  HashDumper(const std::string&  tokenizerPath,const std::string& outFile);
  virtual ~HashDumper();
  virtual bool process(Page& page);
  virtual bool doBatch();
 private:
  
   std::unique_ptr<PageFeaturizer> featurizer_;
   FILE* out_=nullptr;
   std::vector<uint32_t> indptr_;
   std::vector<uint32_t> indices_;
   std::vector<float> data_;
   std::vector<std::string> idkeys_;
   int buffered_=0;
   void* minhashptr_=nullptr;
   uint32_t *result_buffer_=nullptr;
   std::unique_ptr<LshIndex> indexer_;
};
}  // namespace pd

#endif  // PD_DEDUP_FEATURE_DUMPER_H_

