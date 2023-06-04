#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_split.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/log/log.h"
#include "absl/log/check.h"
#include "dedup/lsh_index.h"

ABSL_FLAG(std::string, minhash_path, "/g/wudao_features.txt",
          "calced weighted min hash output file path");
ABSL_FLAG(int, nband, 8, "the band number of lsh");
ABSL_FLAG(int, band_slots, 99829, "hash number of same band");
ABSL_FLAG(float, dup_threshold, 0.75, "the sim hash threshold to judge as duplicate");
ABSL_FLAG(std::string, dup_key_path, "/g/dup_keys.txt",
          "the output duplicated keys file path");
int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  pd::LshIndex index(absl::GetFlag(FLAGS_nband),
                     absl::GetFlag(FLAGS_band_slots));
  std::ifstream inputFile(absl::GetFlag(FLAGS_minhash_path));
  if (!inputFile) {
    LOG(WARNING) << "Failed to open the file:"
                 << absl::GetFlag(FLAGS_minhash_path);
    return 1;
  }
  float dupThreshold=absl::GetFlag(FLAGS_dup_threshold);
  std::string line;
  uint32_t dupCount=0;
  uint32_t totalCount=0;
  std::ofstream outputFile(absl::GetFlag(FLAGS_dup_key_path));
  while (std::getline(inputFile, line)) {
    absl::StripTrailingAsciiWhitespace(&line);
    std::vector<std::string> ss = absl::StrSplit(line,' ');
    const std::string& key = ss[0];
    CHECK_EQ(static_cast<int>(ss.size()), 257)<<line;
    pd::WeightedMinHash wmh;
    for(int i=0;i<128;i++){
        absl::SimpleAtoi(ss[1+i*2],&wmh.ks[i]);
        absl::SimpleAtoi(ss[2+i*2],&wmh.ts[i]);
    }
    std::vector<int> hashvals;
    std::vector<std::string> cands;
    std::vector<float> sims;
    bool gotDup=false;
    totalCount +=1;
    if(index.query(wmh,hashvals,cands,sims)){
        for(size_t i=0;i<cands.size();i++){
            if(sims[i] >= dupThreshold){
                gotDup=true;
                LOG(INFO)<<"key:["<<key<<"] dup to:["<<cands[i]<<"] with sim:["<<sims[i]<<"]";
                outputFile<<key<<std::endl;
                break;
            }
        }
        if(gotDup){
            dupCount +=1;
        }else{
            index.addWeightedMinHash(key, wmh, hashvals);
        }
    }
    if(totalCount % 20000 ==0){
        LOG(INFO)<<totalCount<<" items indexed, with dup:"<<dupCount;
    }
  }
  LOG(INFO)<<totalCount<<" items indexed, with dup:"<<dupCount;
  inputFile.close();
  outputFile.close();
  return 0;
}
