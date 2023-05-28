#include "dedup/page_featurizer.h"

#include "youtokentome/cpp/bpe.h"

namespace pd {
PageFeaturizer::PageFeaturizer(){}
PageFeaturizer::~PageFeaturizer() {}
bool PageFeaturizer::Init(const std::string& vocabPath) {
    vkcom::Status status;
    encoder_.reset(new vkcom::BaseEncoder(vocabPath,1,&status));
    if(!status.ok()){
        return false;
    }
    return true;
}
bool PageFeaturizer::Featurize(Page& page) {
    if(page.content.empty()){
        return true;
    }
    std::vector<std::vector<int>> ids;
    vkcom::Status status = encoder_->encode_as_ids({page.content}, &ids);
    if(!status.ok()){
        return false;
    }
    auto pageIds = ids[0];
    for(auto& id : pageIds){
        float w = log(2.0+id/100.0);
        auto it = page.features.find(id);
        if(it!= page.features.end()){
            page.features.insert({id, w});
        }else{
            it->second +=w;
        }
    }
    return true;
}

}  // namespace pd