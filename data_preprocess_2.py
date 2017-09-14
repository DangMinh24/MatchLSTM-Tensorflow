###########Convert all context and question from tokens to index
########### 1/Read cid,qid,c_toks,q_toks,ans_s,ans_e in data_word_lvl
########### 2/Read known_list, mark all word not in known_list to <unk>
########### 3/Fixed_context=400, Fixed_question=30. Padding with fixed_size with <pad>
########### 4/Use word2index to convert from word to index
########### 5/Filter all sentence that have answ_e > fixed_context_length
########### Note: This approach cost lot of more time than that actually need

##########  Approach 2:
##########  1/ Create 2 dict(): cid2c and qid2q
##########  2/ Using cid2c and qid2q, create cid2index_c and qid2index_q:
##########      -Using known_list  to mark all word in context and question that not in known_list as <unk>
##########      -Padding all context and question into fixed_length
##########  3/ Read cid,qid,c_toks,q_toks,ans_s,ans_e in data_word_lvl
##########  Use cid,qid to access indexed_context, indexed_question
##########  4/ Filter all sentence that have answ_e>fixed_context_length

import pickle
from tqdm import tqdm
import os
import sys
HOME_PATH=os.path.dirname(__file__)
DATA_PATH=os.path.join(HOME_PATH,"data")
TRAIN_DATA_PATH=os.path.join(DATA_PATH,"train")
VAL_DATA_PATH=os.path.join(DATA_PATH,"val")
UTIL_PATH=os.path.join(DATA_PATH,"utility")

UNK="<unk>"
PAD="<pad>"
def filtering_words(data,known_list_path):
    known_list=pickle.load(open(known_list_path,"rb"))
    new_data=[]
    for iter,(cid,qid,c_toks,q_toks,ans_s,ans_e) in enumerate(data):
        filtered_c_toks=[]
        for w in c_toks:
            if w in known_list:
                filtered_c_toks.append(w)
            else:
                filtered_c_toks.append(UNK)
        filtered_q_toks=[]
        for w in q_toks:
            if w in known_list:
                filtered_q_toks.append(w)
            else:
                filtered_q_toks.append(UNK)
        print(filtered_c_toks)
        print(filtered_q_toks)
        print()
        new_data.append((cid,qid,filtered_c_toks,filtered_q_toks,ans_s,ans_e))
    return new_data
def create_shortcut_contextNquestion(data):
    cid2c_dict=dict()
    qid2q_dict=dict()
    cid_set=set()
    qid_set=set()
    for cid,qid,c_toks,q_toks,ans_s,ans_e in data:
        if cid not in cid_set:
            cid2c_dict[cid]=c_toks
            cid_set.add(cid)
        if qid not in qid_set:
            qid2q_dict[qid]=q_toks
            qid_set.add(qid)
    return cid2c_dict,qid2q_dict
def filterNpadding(cid2c,qid2q,known_list,fixed_context,fixed_question):
    for id,toks in tqdm(cid2c.items(),desc="\n"):
        for iter,w in enumerate(toks):
            if w not in known_list:
                toks[iter]=UNK
        if len(toks)>=fixed_context:
            toks=toks[:fixed_context]
        else:
            padding_size = fixed_context - len(toks)
            toks = toks + [PAD] * padding_size
        cid2c[id]=toks
    for id,toks in tqdm(qid2q.items(),desc="\n"):
        for iter,w in enumerate(toks):
            if w not in known_list:
                toks[iter]=UNK
        if len(toks)>=fixed_question:
            toks=toks[:fixed_question]
        else:
            padding_size = fixed_question - len(toks)
            toks = toks + [PAD] * padding_size
        qid2q[id]=toks
    return cid2c,qid2q
def convert2index(cid2padding_c,qid2padding_q,word2index):

    for id,toks in cid2padding_c.items():
        toks_index=[word2index[w] for w in toks]
        cid2padding_c[id]=toks_index
    for id,toks in qid2padding_q.items():
        toks_index=[word2index[w] for w in toks]
        qid2padding_q[id]=toks_index
    return cid2padding_c,qid2padding_q

def create_pairs_index(data_pair,cid2indexed_c,qid2indexed_q):
    new_data_pairs=[]
    for cid,qid,c_toks,q_toks,answ_s,answ_e in data_pair:
        indexed_c_toks=cid2indexed_c[cid]
        indexed_q_toks=qid2indexed_q[qid]
        new_data_pairs.append((cid,qid,indexed_c_toks,indexed_q_toks,answ_s,answ_e))
    return new_data_pairs

def filter_indexed_data(data_indexed,maximum_fixed_context_length):
    filter_data=[]
    for cid,qid,c_toks,q_toks,answ_s,answ_e in data_indexed:
        if answ_e>=maximum_fixed_context_length:
            continue
        else:
            filter_data.append((cid,qid,c_toks,q_toks,answ_s,answ_e))
    return filter_data
if __name__=="__main__":
    data_word_lvl_path = os.path.join(TRAIN_DATA_PATH,"data_word_lvl")
    data = pickle.load(open(data_word_lvl_path,"rb"))

    known_list_path=os.path.join(UTIL_PATH,"known_list")
    known_list=pickle.load(open(known_list_path,"rb"))

    fixed_context_length=400
    fixed_question_length=30
    word2index_path=os.path.join(UTIL_PATH,"w2i")
    w2i=pickle.load(open(word2index_path,"rb"))

    ##########Create qid2q and cid2c
    cid2c,qid2q= create_shortcut_contextNquestion(data)
    cid2padding_c,qid2padding_q=filterNpadding(cid2c,qid2q,known_list,
                               fixed_context_length,fixed_question_length)


    ##########Saving cid2padding_c, qid2padding_q
    cid2padding_c_path=os.path.join(DATA_PATH,"cid2padding_c")
    qid2padding_q_path=os.path.join(DATA_PATH,"qid2padding_q")

    cid2padding_c_file=open(cid2padding_c_path,"wb")
    qid2padding_q_file=open(qid2padding_q_path,"Wb")
    pickle.dump(cid2padding_c,cid2padding_c_file)
    pickle.dump(qid2padding_q,qid2padding_q_file)

    cid2padding_c_file.close()
    qid2padding_q_file.close()

    #########Convert to index
    cid2padding_c_file = open(cid2padding_c_path, "rb")
    qid2padding_q_file = open(qid2padding_q_path, "rb")
    cid2padding_c=pickle.load(cid2padding_c_file)
    qid2padding_q=pickle.load(qid2padding_q_file)
    cid2indexed_c,qid2indexed_q=convert2index(cid2padding_c,qid2padding_q,w2i)


    #########Saving cid2indexed_c,qid2indexed_q
    cid2indexed_c_path=os.path.join(DATA_PATH,"cid2indexed_c")
    qid2indexed_q_path=os.path.join(DATA_PATH,"qid2indexed_q")
    cid2indexed_c_file=open(cid2indexed_c_path,"wb")
    qid2indexed_q_file=open(qid2indexed_q_path,"wb")
    pickle.dump(cid2indexed_c,cid2indexed_c_file)
    pickle.dump(qid2indexed_q,qid2padding_q_file)
    cid2indexed_c_file.close()
    qid2indexed_q_file.close()

    ########Create data_indexed
    cid2indexed_c_file=open(cid2indexed_c_path,"rb")
    qid2indexed_q_file=open(qid2indexed_q_path,"rb")
    cid2indexed_c=pickle.load(cid2indexed_c_file)
    qid2indexed_q=pickle.load(qid2padding_q_file)
    data_indexed_pairs=create_pairs_index(data,cid2indexed_c,qid2indexed_q)
    #######Saving data_indexed
    data_indexed_path=os.path.join(TRAIN_DATA_PATH,"data_indexed")
    pickle.dump(data_indexed_pairs,open(data_indexed_path,"wb"))

    #######Filter data_indexed
    data_indexed_path=os.path.join(TRAIN_DATA_PATH,"data_indexed")
    data_indexed=pickle.load(open(data_indexed_path,"rb"))
    data_filtered=filter_indexed_data(data_indexed,fixed_context_length)
    #######Saving data_filtered
    data_filtered_path=os.path.join(TRAIN_DATA_PATH,"data_filtered")
    pickle.dump(data_filtered,open(data_filtered_path,"wb"))


    ##########Validation:
    val_data_path=os.path.join(VAL_DATA_PATH,"val_data_word_lvl")
    val_data=pickle.load(open(val_data_path,"rb"))
    fixed_context_length=400
    fixed_question_length=30
    val_cid2c,val_qid2q= create_shortcut_contextNquestion(val_data)
    val_cid2padding_c,val_qid2padding_q=filterNpadding(val_cid2c,val_qid2q,known_list,
                               fixed_context_length,fixed_question_length)
    ########Saving validation
    val_cid2padding_c_path=os.path.join(VAL_DATA_PATH,"val_cid2padding_c")
    val_qid2padding_q_path=os.path.join(VAL_DATA_PATH,"val_qid2padding_q")

    val_cid2padding_c_file=open(val_cid2padding_c_path,"wb")
    val_qid2padding_q_file=open(val_qid2padding_q_path,"wb")
    pickle.dump(val_cid2padding_c,val_cid2padding_c_file)
    pickle.dump(val_qid2padding_q,val_qid2padding_q_file)

    #########Convert validation to index
    val_cid2padding_c_path=os.path.join(VAL_DATA_PATH,"val_cid2padding_c")
    val_qid2padding_q_path=os.path.join(VAL_DATA_PATH,"val_qid2padding_q")
    val_cid2padding_c_file = open(val_cid2padding_c_path, "rb")
    val_qid2padding_q_file = open(val_qid2padding_q_path, "rb")
    val_cid2padding_c=pickle.load(val_cid2padding_c_file)
    val_qid2padding_q=pickle.load(val_qid2padding_q_file)

    val_cid2indexed_c,val_qid2indexed_q=convert2index(val_cid2padding_c,val_qid2padding_q,w2i)
    val_cid2indexed_c_path=os.path.join(VAL_DATA_PATH,"val_cid2indexed_c")
    val_qid2indexed_q_path=os.path.join(VAL_DATA_PATH,"val_qid2indexed_q")
    val_cid2indexed_c_file=open(val_cid2indexed_c_path,"wb")
    val_qid2indexed_q_file=open(val_qid2indexed_q_path,"wb")
    pickle.dump(val_cid2indexed_c,val_cid2indexed_c_file)
    pickle.dump(val_qid2indexed_q,val_qid2indexed_q_file)

    #########Create validation_pair
    val_cid2indexed_c_file=open(val_cid2indexed_c_path,"rb")
    val_qid2indexed_q_file=open(val_qid2indexed_q_path,"rb")
    val_cid2indexed_c=pickle.load(val_cid2indexed_c_file)
    val_qid2indexed_q=pickle.load(val_qid2indexed_q_file)
    val_indexed_pairs=create_pairs_index(val_data,val_cid2indexed_c,val_qid2indexed_q)
    ##########Saving validation_indexed
    val_indexed_path=os.path.join(VAL_DATA_PATH,"val_indexed")
    pickle.dump(val_indexed_pairs,open(val_indexed_path,"wb"))

    #######Filter val_indexed
    val_indexed_path=os.path.join(VAL_DATA_PATH,"val_indexed")
    val_indexed=pickle.load(open(val_indexed_path,"rb"))
    fixed_context_length=400
    val_filtered=filter_indexed_data(val_indexed,fixed_context_length)
    #######Saving data_filtered
    val_filtered_path=os.path.join(VAL_DATA_PATH,"val_filtered")
    pickle.dump(val_filtered,open(val_filtered_path,"wb"))