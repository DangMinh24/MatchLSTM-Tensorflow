########Create data pair for validation or train set
########Read from original json file:=>Convert into qid,cid,q_context,c_context,a_start_character_level,a_length
import pickle
import json
import os
from nltk import StanfordTokenizer
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.pyplot import hist
import sys
from configargparse import ArgParser

HOME_PATH=os.path.dirname(__file__)
DATA_PATH=os.path.join(HOME_PATH,"data")
TOKENIZER_PATH=os.path.join(DATA_PATH,"tokenizer")
TOKENIZER_CORE_PATH=os.path.join(TOKENIZER_PATH,"stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar")
UTIL_PATH=os.path.join(HOME_PATH,"utility")
def create_pair(data_path):
    data_file=open(data_path,"r")
    data=json.load(data_file)["data"]
    result=[]
    cid=[]
    c_count=0
    qid=[]
    c_str=[]
    q_str=[]
    a_s=[]
    a_str=[]
    for article in data:
        for pa in article["paragraphs"]:
            context=pa["context"]
            context=context.lower()
            cid.append(c_count)
            for qa in pa["qas"]:
                qid_=qa["id"]
                question=qa["question"]
                question=question.lower()
                qid.append(qid_)
                for answ in qa["answers"]:
                    answ_s=answ["answer_start"]
                    answ_str=answ["text"]
                    result.append((c_count,qid_,context,question,answ_s,answ_str))
            c_count += 1
    return result

def max_contextNquestion_length(data_wordlvl_path):
    data=pickle.load(open(data_wordlvl_path,"rb"))
    hist_length_c=[]
    hist_length_q=[]
    for cid,qid,tok_c,tok_q,ans_s,ans_e in data:
        hist_length_c.append(len(tok_c))
        hist_length_q.append(len(tok_q))
    return hist_length_c,hist_length_q
if __name__=="__main__":
    ########Add argument list(num_worker)
    parser=ArgParser()
    parser.add_argument("-workers","--num_workers",default=1)
    parser.add_argument("-tok_path","--tok_dir",default=TOKENIZER_CORE_PATH)


    args=parser.parse_args()
    workers=int(args.num_workers)
    ########Check tokenizer:
    tokenizer=StanfordTokenizer(args.tok_dir)

    train_data_path=os.path.join(DATA_PATH,"SQuAD-v1.1-train.json")
    known_list_path=os.path.join(UTIL_PATH,"known_list")
    train_data_pairs=create_pair(train_data_path)

    TRAIN_DATA_PATH=os.path.join(DATA_PATH,"train")
    train_save_file=open(os.path.join(TRAIN_DATA_PATH,"data_"),"wb")
    #########Saving
    print("Extracting train features in character level...")
    pickle.dump(train_data_pairs,train_save_file)
    train_save_file.close()
    print("Saving in %s"%os.path.join(TRAIN_DATA_PATH,"data_"))
    print("Done")

    def func_(element):
        cid, qid, c_str, q_str, answ_s, answ_str = element
        tokenized_c = tokenizer.tokenize(c_str)
        tokenized_q = tokenizer.tokenize(q_str)
        sub_context = c_str[0:answ_s]
        tokenized_sub_context = tokenizer.tokenize(sub_context)
        answ_s_wordlvl = len(tokenized_sub_context)
        tokenized_answ = tokenizer.tokenize(answ_str)
        answ_e_wordlvl = answ_s_wordlvl + len(tokenized_answ)
        print("s:%d \t e:%d"%(answ_s_wordlvl,answ_e_wordlvl))
        print(tokenized_c[answ_s_wordlvl:answ_e_wordlvl])
        return cid, qid, tokenized_c, tokenized_q, answ_s_wordlvl, answ_e_wordlvl

    print()
    ###########To wordlevel
    print("Creating train data word-level...")
    train_save_file=open(os.path.join(TRAIN_DATA_PATH,"data_"),"rb")
    data=pickle.load(train_save_file)
    pool=Pool(workers)
    result_map=pool.map(func_,data)
    pool.close()
    pool.join()
    train_word_lvl_file=open(os.path.join(TRAIN_DATA_PATH,"data_word_lvl"),"wb")
    pickle.dump(result_map,file=train_word_lvl_file)
    train_word_lvl_file.close()
    print("Saving in %s"%os.path.join(TRAIN_DATA_PATH,"data_word_lvl"))
    print("Done")

    print()
    #################Validation
    print("Extracting validation's features in character level...")
    val_data_path=os.path.join(DATA_PATH,"SQuAD-v1.1-dev.json")
    val_pairs=create_pair(val_data_path)

    VAL_DATA_PATH=os.path.join(DATA_PATH,"val")
    val_save_file=open(os.path.join(VAL_DATA_PATH,"val_data_"),"wb")
    pickle.dump(val_pairs,val_save_file)
    val_save_file.close()
    print("Saving in %s"%os.path.join(VAL_DATA_PATH,"val_data_"))
    print("Done")

    print()
    #########Word level
    print("Creating train data word-level...")
    val_save_file = open(os.path.join(VAL_DATA_PATH, "val_data_"), "rb")
    val_data=pickle.load(val_save_file)
    pool=Pool(workers)
    result_map=pool.map(func_,val_data)
    pool.close()
    pool.join()
    val_data_word_lvl_file=open(os.path.join(VAL_DATA_PATH,"val_data_word_lvl"),"wb")
    pickle.dump(result_map,val_data_word_lvl_file)
    val_data_word_lvl_file.close()
    print("Saving in %s"%os.path.join(VAL_DATA_PATH,"val_data_word_lvl"))
    print("Done")
