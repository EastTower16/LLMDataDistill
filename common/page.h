#ifndef PD_COMMON_PAGE_H_
#define PD_COMMON_PAGE_H_
#include <string>
#include <vector>
#include <unordered_map>

namespace pd {

struct Page {
  std::string title;
  std::string url;
  std::string content;
  std::string idkey;
  std::string category;

  // filled by minhashcuda
  std::vector<std::pair<uint32_t,uint32_t>> weighted_hash_values;
  // features filled by featurizer
  std::unordered_map<int, float> features;

};

}  // namespace pd

#endif  // PD_COMMON_PAGE_H_
