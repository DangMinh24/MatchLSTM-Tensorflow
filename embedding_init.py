############Create embedding numpy
############Using w2i and i2w (for train and val seperately)
import pickle
import numpy as np
from gensim.models import KeyedVectors
import os
import sys
from configargparse import ArgParser

HOME_PATH=os.path.dirname(__file__)
DATA_PATH=os.path.join(HOME_PATH,"data")
UTIL_PATH=os.path.join(DATA_PATH,"utility")
GLOVE_PATH=os.path.join(DATA_PATH,"glove")

def embedding_init(w2i_path,i2w_path,glove_path,unknown_vector=None,padding_vector=None,dimension=50):
    glove=KeyedVectors.load_word2vec_format(glove_path,binary=False)
    w2i=pickle.load(open(w2i_path,"rb"))
    i2w=pickle.load(open(i2w_path,"rb"))
    i2w_list=list(i2w.items())
    sorted_i2w=sorted(i2w_list,key=lambda x: int(x[0]))
    embedding_np=[]
    for i,w in sorted_i2w:
        if w =="<unk>":
            w_vector = np.array([-10e6]*dimension)
        elif w=="<pad>":
            w_vector=np.array([0]*dimension)
        else:
            w_vector=glove[w]
        embedding_np.append(w_vector)
    return np.array(embedding_np)
if __name__=="__main__":
    #############add Argument list [glove_w2v_dimension=50]
    parser=ArgParser()
    parser.add_argument("-glove_dim","--glove_dimension",default=50)
    args=parser.parse_args()

    #############Check if glove exist??
    print("Creating embedding matrix...")
    glove_dimension=int(args.glove_dimension)
    glove_name="glove.6B.%dd.w2v.txt"%glove_dimension
    glove_file_path=os.path.join(GLOVE_PATH,glove_name)
    w2i_path=os.path.join(UTIL_PATH,"w2i")
    i2w_path=os.path.join(UTIL_PATH,"i2w")
    embedding_np=embedding_init(w2i_path,i2w_path,glove_file_path)

    save_embedding_path=os.path.join(UTIL_PATH,"embedding_np")
    #####Saving file
    pickle.dump(embedding_np,open(save_embedding_path,"wb"))
    print("Saving in %s"%save_embedding_path)
    print("Done")
    print()