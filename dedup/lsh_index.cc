#include "dedup/lsh_index.h"

namespace pd{

static int hash_func(const WeightedMinHash& wmh, int nband, int bidx, size_t mod){
    size_t hash = 0;
    size_t numPerBand = wmh.ts.size()/nband;
    for(size_t i = 0; i < numPerBand; ++i){
        hash = (hash * 60607)%mod + (wmh.ks[nband * bidx + i]*30637)%mod+ wmh.ts[nband * bidx + i];
        hash = hash % mod;
    }
    return hash;
}

static void dohash(const WeightedMinHash& wmh, int nband, size_t mod, std::vector<int>& hashvals){
    for(int i=0;i<nband;i++){
        hashvals.push_back(hash_func(wmh,nband, i, mod));
    }
}
LshIndex::LshIndex(int nband,int slotsPerBand){
    band_indexes_.resize(nband);
    for(int i=0;i<nband;i++){
        band_indexes_[i].resize(slotsPerBand);
    }
}
LshIndex::~LshIndex(){}
bool LshIndex::addWeightedMinHash(const std::string& key, const WeightedMinHash& wmh, const std::vector<int>& hashvls){
    size_t nband=band_indexes_.size();
    if(wmh.ks.size() % nband !=0 || nband ==0 ){
        return false;
    }
    if(all_hashes_.count(key)>0){
        return false;
    }
    all_hashes_.insert(std::make_pair(key, wmh));
    for(size_t i=0;i<nband;i++){
        int hashval=hash_func(wmh,nband, i, band_indexes_[0].size());
        band_indexes_[i][hashval].insert(key);
    }
    return true;
}
bool LshIndex::query(const WeightedMinHash& wmh,std::vector<int>& hashvals, std::vector<std::string>& cands,std::vector<float>& sims){
    size_t nband=band_indexes_.size();
    if(wmh.ks.size() % nband !=0 || nband ==0 ){
        return false;
    }
    dohash(wmh, nband, band_indexes_[0].size(),hashvals);
    std::unordered_set<std::string> candidates;
    for(size_t i=0;i<nband;i++){
        std::unordered_set<std::string>& iset=band_indexes_[i][hashvals[i]];
        for(size_t j=i+1;j<nband;j++){
            std::unordered_set<std::string>& jset=band_indexes_[j][hashvals[j]];
            for(auto key: jset){
                if(iset.count(key)>0){
                    candidates.insert(key);
                }
            }
        }
    }
    for(auto key : candidates){
        const WeightedMinHash& other = all_hashes_[key];
        cands.push_back(key);
        sims.push_back(wmh.jaccard(other));
    }
    return true;
}
}  // namespace pd
