#ifndef PD_DEDUP_PAGE_FEATURIZER_H_
#define PD_DEDUP_PAGE_FEATURIZER_H_
#include <memory>
#include <string>

#include "common/page.h"
namespace vkcom {
class BaseEncoder;
}  // namespace vkcom
namespace pd {
class PageFeaturizer {
 public:
  PageFeaturizer();
  virtual ~PageFeaturizer();
  virtual bool Init(const std::string& vocabPath);
  virtual bool Featurize(Page& page);

 private:
  std::unique_ptr<vkcom::BaseEncoder> encoder_;
};
}  // namespace pd

#endif  // PD_DEDUP_PAGE_FEATURIZER_H_
