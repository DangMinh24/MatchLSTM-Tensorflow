#############This file process word and index
#############1/ Create word2index
#############2/ Create index2word
#############3/ Add Unknown and Padding to word2index and index2word
import pickle
import os
import sys

HOME_PATH=os.path.dirname(__file__)
DATA_PATH=os.path.join(HOME_PATH,"data")
UTIL_PATH=os.path.join(DATA_PATH,"utility")

def w2i_i2w_init(known_list_path):
    UKN="<unk>"
    PAD="<pad>"
    known_list_=pickle.load(open(known_list_path,"rb"))
    known_list_=[UKN]+[PAD]+known_list_
    w2i=dict()
    i2w=dict()
    for iter,w in enumerate(known_list_):
        i2w[iter]=w
        w2i[w]=iter
    return w2i,i2w

if __name__=="__main__":
    ########Check if known_list file is exists?
    print("Creating word2index and index2word...")
    known_list_path=os.path.join(UTIL_PATH,"known_list")
    if not os.path.exists(known_list_path):
        raise FileExistsError("File not exist")

    w2i,i2w=w2i_i2w_init(known_list_path)
    #########Saving dictionary:
    pickle.dump(w2i,open(os.path.join(UTIL_PATH,"w2i"),"wb"))
    pickle.dump(i2w,open(os.path.join(UTIL_PATH,"i2w"),"wb"))
    print("Save in %s"%UTIL_PATH)
    print("Done")
    print()