#ifndef PD_COMMON_PAGE_PRODUCER_H_
#define PD_COMMON_PAGE_PRODUCER_H_
#include <string>
#include <vector>
#include <thread>
#include <atomic>

#include "common/page.h"
#include "blockingconcurrentqueue.h"
namespace pd {

class PageProducer {
  public:
    virtual ~PageProducer();
    virtual bool initByFileList(const std::vector<std::string>& fileList );
    virtual bool takeOnePage(struct Page& page,bool waitForValidPage=false);
    virtual void shutdown();
  private:
    moodycamel::BlockingConcurrentQueue<struct Page> page_queue_{1024};
    std::thread producer_;
    std::atomic_bool stop_=false;
    static std::string EOFPageHashKey;
};

}  // namespace pd

#endif  // PD_COMMON_PAGE_PRODUCER_H_
