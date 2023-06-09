#ifndef PD_DEDUP_LSH_INDEX_H_
#define PD_DEDUP_LSH_INDEX_H_
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "absl/container/flat_hash_set.h"
#include "absl/container/flat_hash_map.h"

namespace pd {
struct WeightedMinHash{
    std::vector<uint32_t> ks;
    std::vector<uint32_t> ts;
    float jaccard(const WeightedMinHash& other) const {
        int nn=ks.size();
        int same=0;
        for(int i=0;i<nn;i++){
            if(ks[i]==other.ks[i] && ts[i]== other.ts[i]){
                same +=1;
            }
        }
        return static_cast<float>(same)/ static_cast<float>(nn);
    }

};
class LshIndex {
 public:
  LshIndex(int nband,int slotsPerBand);
  virtual ~LshIndex();
  virtual bool addWeightedMinHash(const std::string& key, const WeightedMinHash& wmh,const std::vector<int>& hashvls);
  virtual bool query(const WeightedMinHash& wmh,std::vector<int>& hashvals,std::vector<std::string>& cands, std::vector<float>& sims);
 private:
   absl::flat_hash_set<std::string> hashkeys_;
   std::vector<std::string> all_keys_;
   std::vector<WeightedMinHash> hash_values_;
   std::vector<std::vector<absl::flat_hash_set<int> >> band_indexes_;
};
}  // namespace pd

#endif  // PD_DEDUP_LSH_INDEX_H_
