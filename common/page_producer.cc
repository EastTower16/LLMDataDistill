#include "common/page_producer.h"

#include <chrono>
#include <fstream>
#include <string>

#include "nlohmann/json.hpp"

namespace pd {

PageProducer::~PageProducer() {}
bool PageProducer::initByFileList(const std::vector<std::string>& fileList) {
  producer_ = std::thread([this,fileList]() {
    for (auto path : fileList) {
      if (this->stop_.load()) {
        break;
      }
      std::ifstream file(path);
      nlohmann::json objList = nlohmann::json::parse(file);
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
        while(!this->page_queue_.try_enqueue(p)){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
      if (this->stop_.load()) {
        break;
      }
    }
  });
  return true;
}
bool PageProducer::takeOnePage(struct Page& page, bool waitForValidPage) {
  return false;
}
void PageProducer::shutdown() {
  stop_.store(true);
  if (producer_.joinable()) {
    producer_.join();
  }
}

}  // namespace pd
