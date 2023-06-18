#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_split.h"
#include "absl/strings/str_join.h"
#include "absl/strings/numbers.h"

#include "youtokentome/cpp/bpe.h"

ABSL_FLAG(std::string, tagged_corpus_path, "/g/chatgpt_output.txt",
          "the tagged corpuse path");
ABSL_FLAG(std::string, output_path, "/g/tagged_dataset.txt",
          "segmented corpuse output path");
ABSL_FLAG(std::string, tokenizer_path, "vocab/tokme.model",
          "the tokenizer model path");
ABSL_FLAG(int, max_length, 4096, "the maximum length of text");

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage("Gen Ex Main");
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);
  std::unique_ptr<vkcom::BaseEncoder> encoder;
  vkcom::Status status;
  encoder.reset(new vkcom::BaseEncoder(absl::GetFlag(FLAGS_tokenizer_path),1,&status));
  if(!status.ok()){
    LOG(ERROR)<<"init tokenizer error,path:"<<absl::GetFlag(FLAGS_tokenizer_path);
    return -1;
  }
  int qualityGrades[10]={0};
  int saleGrades[10]={0};
  std::ofstream ofs(absl::GetFlag(FLAGS_output_path));
  int maxLength = absl::GetFlag(FLAGS_max_length);
  std::ifstream inputFile(absl::GetFlag(FLAGS_tagged_corpus_path));
  if (inputFile) {
    std::string line;
    while (std::getline(inputFile, line)) {
      std::vector<std::string> vs = absl::StrSplit(line, absl::ByChar('\t'));
      int quality=0, saleGrade =0;
      absl::SimpleAtoi(vs[0],&quality);
      absl::SimpleAtoi(vs[1],&saleGrade);
      int g = quality/10;
      if(g>=10){
        g = 9;
      }
      qualityGrades[g]+=1;
      g = saleGrade /10;
      if(g>=10){
        g = 9;
      }
      saleGrades[g]+=1;
      std::vector<std::vector<int>> ids;
      status = encoder->encode_as_ids({vs[2]}, &ids);
      if(!status.ok() || ids.empty()){
        LOG(ERROR)<<"encode_as_ids error!";
        continue;
      }
      std::vector<int> tokens=ids[0];
      int textLen = tokens.size();
      if(textLen > maxLength){
        tokens = std::vector<int>(tokens.begin(),tokens.begin()+maxLength);
      }
      std::string text = absl::StrJoin(tokens, " ");
      ofs<<quality<<"\t"<<saleGrade<<"\t"<<text<<std::endl;
    }
  } else {
    LOG(ERROR) << "Failed to open the file:"
              << absl::GetFlag(FLAGS_tagged_corpus_path);
  }
  ofs.close();
  LOG(INFO)<<"quality grades distributions......";
  for(int i=0;i<10;i++){
    LOG(INFO)<<"quality [ "<<i*10<<"-"<<(i+1)*10-1 <<"]:"<<qualityGrades[i];
  }
  LOG(INFO)<<"sale grades distributions......";
  for(int i=0;i<10;i++){
    LOG(INFO)<<"sale [ "<<i*10<<"-"<<(i+1)*10-1 <<"]:"<<saleGrades[i];
  }
  return 0;
}


