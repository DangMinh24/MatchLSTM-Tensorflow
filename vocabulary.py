from nltk import StanfordTokenizer
import os
import json
from multiprocessing import Pool
import pickle
import sys
from configargparse import ArgParser
HOME_PATH=os.path.dirname(__file__)
DATA_PATH=os.path.join(HOME_PATH,"data")
TOKENIZER_PATH=os.path.join(DATA_PATH,"tokenizer")
TOKENIZER_CORE_PATH=os.path.join(TOKENIZER_PATH,"stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar")

def build_WordList(data_path,use_dev_set=False):
    contextNquestion=[]

    data_path=os.path.join(data_path,"SQuAD-v1.1-train.json")
    data=json.load(open(data_path,"r"))["data"]

    if use_dev_set==True:
        data_dev_path=os.path.join(data_path,"SQuAD-v1.1-dev.json")
        data_dev=json.load(open(data_dev_path,"r"))["data"]


    for iter,article in enumerate(data):
        for pa in article["paragraphs"]:
            context=pa["context"]
            context=context.lower()
            contextNquestion.append(context)
            for qa in pa["qas"]:
                question=qa["question"]
                question=question.lower()
                contextNquestion.append(question)

    ########Append vocabulary size with dev_set
    if use_dev_set==True:
        for iter,article in enumerate(data):
            for pa in article["paragraphs"]:
                context=pa["context"]
                context=context.lower()
                contextNquestion.append(context)
                for qa in pa["qas"]:
                    question=qa["question"]
                    question=question.lower()
                    contextNquestion.append(question)
    return contextNquestion

if __name__=="__main__":
    ######Reading Args through ConfigArgParser
    parser=ArgParser()

    parser.add_argument("-workers","--num_workers",default=1)
    parser.add_argument("-use_dev","--use_dev",default=False,)
    parser.add_argument("-tok_path","--tok_dir",default=TOKENIZER_CORE_PATH)

    args=parser.parse_args()

    try:
        tokenizer=StanfordTokenizer(args.tok_dir)
    except LookupError:
        raise LookupError("Can't find Stanford-corenlp in %s" % args.tok_dir)


    def func_(element):
        tokenized_list = tokenizer.tokenize(element)
        print(tokenized_list[:5])
        return tokenized_list

    contextNquestion=build_WordList(DATA_PATH,use_dev_set=args.use_dev)
    print("Use %d worker(s) to create vocabulary..."%int(args.num_workers))
    pool=Pool(processes=int(args.num_workers))
    result_map=pool.map(func_,contextNquestion)
    pool.close()
    pool.join()

    print("Creating set of vocabulary...")
    vocabulary=[]
    for subset in result_map:
        vocabulary.extend(subset)
    print("Done")
    print("There are %d words in vocabulary (NOT INCLUDE <PAD> AND <UNKNOWN>)"\
          %len(set(vocabulary)))


    if args.use_dev:
        path_save=os.path.join(
                    os.path.join(DATA_PATH,"utility"),"vocabulary_train_dev")
        pickle.dump(set(vocabulary),open(path_save,"wb"))
    else:
        path_save = os.path.join(
            os.path.join(DATA_PATH, "utility"), "vocabulary_train")
        pickle.dump(set(vocabulary), open(path_save, "wb"))


