import os
import json


def loadBlacklist(paths):
    retSet = set([])
    for p in paths:
        with open(p,"r",encoding="utf-8") as inp:
            for line in inp:
                ss = line.strip().split('\t')
                retSet.add(ss[0])
    return retSet

def processCorpus(inputdir, dataTypes, blacklistPath,outputPath):
    total =0
    valid = 0
    prefilter =0
    tooShort =0
    blackSet= loadBlacklist(blacklistPath)
    types = set(dataTypes)
    print(f"got {len(blackSet)} keys in blacklist")
    with open(outputPath,"w",encoding='utf8') as outp:
        #list inputdir
        for file in os.listdir(inputdir):
            if file.endswith(".json"):
                with open(os.path.join(inputdir, file),"r") as inp:
                    data = inp.read()
                    objarr = json.loads(data)
                    for obj in objarr:
                        title = obj['title']
                        text = obj['content']
                        type= obj['dataType']
                        ukey = obj['uniqueKey']
                        total +=1
                        if total%100000==0:
                            print(f"processed {total}, prefilter:{prefilter}, valid:{valid}, short :{tooShort}.....")
                        
                    
                        if type in types:
                            prefilter +=1
                            if ukey not in blackSet:
                                if len(title) < 5 or  len(text) <10:
                                    tooShort +=1
                                    continue
                                json.dump(obj,outp, ensure_ascii=False)
                                outp.write("\n")
                                valid +=1
  
    print(f"processed {total}, prefilter:{prefilter}, valid:{valid}, short :{tooShort}.....")
                    


if __name__ == '__main__':
    # processCorpus("/g/wudao/",'博客',["/g/duplicate_keys.txt","/g/distilled_pages.txt"],'/g/distilled_blog_pages.jsonl')
    # processCorpus("/g/wudao/",['百科'],[],'/g/baike_pages.jsonl')
    # processCorpus("/g/wudao/",['经验','小红书攻略','健康','亲子','医学问答','生活','理论','期货','观点','党建','信托','国学','评论','法律','百家号文章','科普文章', '孕育常识'],[],'/g/other2_pages.jsonl')
    processCorpus("/g/wudao/",["经济","娱乐","文化","军事","游戏","汽车","科技","农业","体育","国际","教育","社会","旅行","房产","股票","豆瓣话题","资讯","新闻"],["/g/duplicate_keys.txt"],'/g/other1_pages.jsonl')
