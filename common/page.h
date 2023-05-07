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

  // filled by dedup
  std::vector<int> min_hashes;
  // features filled by featurizer
  std::unordered_map<int, float> features;

};

}  // namespace pd

#endif  // PD_COMMON_PAGE_H_
