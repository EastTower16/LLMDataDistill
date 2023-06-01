#include "common/page_producer.h"

#include <chrono>
#include <fstream>
#include <string>

#include "absl/log/log.h"
#include "nlohmann/json.hpp"

namespace pd {
std::string PageProducer::EOFPageHashKey ="__<eof_hash_key>__";
PageProducer::~PageProducer() {}

bool PageProducer::initByFileList(const std::vector<std::string>& fileList) {
  producer_ = std::thread([this,fileList]() {
    LOG(INFO) << "Start  process thread, num file:" << fileList.size();
    for (auto path : fileList) {
      if (this->stop_.load()) {
        break;
      }
      std::ifstream file(path);
      nlohmann::json objList = nlohmann::json::parse(file);
      LOG(INFO) << "got :" << objList.size()<<" from path: " << path;
      for (nlohmann::json::iterator it = objList.begin(); it != objList.end(); ++it) {
        if (this->stop_.load()) {
          break;
        }
        nlohmann::json& obj = *it;
        struct Page p;
        p.title = obj["title"].get<std::string>();
        p.content = obj["content"].get<std::string>();
        p.idkey = obj["titleUkey"].get<std::string>();
        p.category = obj["dataType"].get<std::string>();
        if(p.category != "博客"){
          continue;
        }
        while(!this->page_queue_.try_enqueue(p)){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // LOG(INFO)<<"processed :"<<p.title;
      }
      if (this->stop_.load()) {
        break;
      }
    }
    struct Page eof;
    eof.idkey = 	EOFPageHashKey;
    this->page_queue_.enqueue(eof);
    
  });
  return true;
}
bool PageProducer::takeOnePage(struct Page& page, bool waitForValidPage) {
  if (!waitForValidPage){
    return this->page_queue_.try_dequeue(page);
  }
  page_queue_.wait_dequeue(page);
  return true;
}
void PageProducer::shutdown() {
  stop_.store(true);
  producer_.join();
}

}  // namespace pd
