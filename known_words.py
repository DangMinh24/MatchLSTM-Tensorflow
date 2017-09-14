#######Restrict vocabulary of wiki depend on vocabulary of glove
#######Vocabulary of glove's about 400k
#######Assuming any words that outside the vocabulary is not known

import pickle
import os
from gensim.models import KeyedVectors
from gensim.scripts import glove2word2vec
import sys
from configargparse import ArgParser

HOME_PATH=os.path.dirname(__file__)
DATA_PATH=os.path.join(HOME_PATH,"data")
GLOVE_PATH=os.path.join(DATA_PATH,"glove")
UTIL_PATH=os.path.join(DATA_PATH,"utility")


def preprocess_raw_glove(glove_path,save_path):
    glove2word2vec.glove2word2vec(glove_path,save_path)

def filter_vocabulary(vocabulary_path,glove_path):
    glove_w2v=KeyedVectors.load_word2vec_format(glove_path,binary=False)
    glove_vocabulary=glove_w2v.vocab.keys()
    vocab=pickle.load(open(vocabulary_path,"rb"))
    known_list=[]
    for w in vocab:
        if w in glove_vocabulary:
            known_list.append(w)
    return known_list

if __name__=="__main__":
    ######## Add Arguments List:(w2v_size,use_dev=False)
    parser=ArgParser()
    parser.add_argument("-glove_dim","--glove_dimension",default="50")
    parser.add_argument("-use_dev",default=False)
    args=parser.parse_args()

    glove_dimension=int(args.glove_dimension)
    glove_file = "glove.6B.%dd.txt" % glove_dimension
    glove_w2v_file="glove.6B.%dd.w2v.txt"%glove_dimension
    glove_path = os.path.join(GLOVE_PATH, glove_file)
    glove_w2v_path=os.path.join(GLOVE_PATH,glove_w2v_file)
    ########Check if glove w2v is exist?

    if os.path.exists(glove_w2v_file)==False:
        print("Creating glove_w2v-%d dimension..."%glove_dimension)
        preprocess_raw_glove(glove_path,glove_w2v_path)

    if args.use_dev==True:
        vocabulary_path=os.path.join(UTIL_PATH,"vocabulary_train_dev")
    else:
        vocabulary_path=os.path.join(UTIL_PATH,"vocabulary_train")

    save_known_list_path=os.path.join(UTIL_PATH,"known_list")
    known_list=filter_vocabulary(vocabulary_path,glove_w2v_path)

    #######Save file
    pickle.dump(known_list,open(save_known_list_path,"wb"))
    print("Saving in %s"%save_known_list_path)
    print("Done")
    print()

