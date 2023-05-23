#ifndef PD_COMMON_PAGE_PRODUCER_H_
#define PD_COMMON_PAGE_PRODUCER_H_
#include <string>
#include <vector>

namespace pd {

class PageProducer {
  public:
    virtual ~PageProducer();
    virtual bool initByFileList(const std::vector<std::string>& fileList );
    virtual bool takeOnePage(Page& page,bool waitForValidPage=false);
    virtual void shutdown();
};

}  // namespace pd

#endif  // PD_COMMON_PAGE_PRODUCER_H_
