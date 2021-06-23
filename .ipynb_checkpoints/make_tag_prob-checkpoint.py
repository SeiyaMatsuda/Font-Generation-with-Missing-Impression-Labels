import tqdm
import glob
import os
import word2vec
from mylib import pickle_dump,label_preprocess
class make_tag_prob:
    def __init__(self):
        self.tag_files=sorted(glob.glob('./Myfont/dataset/taglabel/*'))
        self.Embedding_model=word2vec.word2vec()
        self.taglabel={}
        self.token_ap_prob={}

    def Embedding(self,token):
        try:
            self.Embedding_model[token]
            TF = True
        except KeyError:
            # print(idx,token)
            TF = False
        return TF
    def Calculate_prob(self):
        for i, t_f in tqdm.tqdm(enumerate(self.tag_files)):
            with open(t_f, 'r', encoding='utf-8') as f:
                text = f.read()
                Font_name = os.path.basename(t_f)
                tag = self.label_preprocess(text)
                tag_list = []
                for idx, token in enumerate(tag):
                    if self.Embedding(token) == True:
                        tag_list.append(token)
                    else:
                        continue
                for t in tag_list:
                    try:
                        self.token_ap_prob[t] += (1 / len(self.tag_files)) * (1 / len(tag_list))
                    except KeyError:
                        self.token_ap_prob[t]= (1 / len(self.tag_files)) * (1 / len(tag_list))
                self.taglabel[Font_name] = tag_list
        impression_word=sorted(self.taglabel.items())
        prob_list=[]
        for idx, i_w in tqdm.tqdm(enumerate(impression_word)):
            prob_list.append([self.token_ap_prob[token] for token in i_w[1]])
        return prob_list
if __name__=="__main__":
    make_tag_prob=make_tag_prob()
    prob_list=make_tag_prob.Calculate_prob()
    base_dir="./Myfont/dataset/tag_prob"
    if os.path.isdir(base_dir)==False:
        os.makedirs(base_dir)
    pickle_dump(prob_list, os.path.join(base_dir,"tag_prob.pickle"))


