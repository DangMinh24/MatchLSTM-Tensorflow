from configargparse import ArgParser
import os
HOME_PATH=os.path.dirname(__file__)
DATA_PATH=os.path.join(HOME_PATH,"data")
TRAIN_DATA_PATH=os.path.join(DATA_PATH,"train")
VAL_DATA_PATH=os.path.join(DATA_PATH,"val")
UTIL_PATH=os.path.join(DATA_PATH,"utility")
GLOVE_PATH=os.path.join(DATA_PATH,"glove")
TOKENIZER_PATH=os.path.join(DATA_PATH,"tokenizer")
TOKENIZER_CORE_PATH=os.path.join(TOKENIZER_PATH,"stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar")

if __name__=="__main__":
    parser=ArgParser(default_config_files="default_config")
    parser.add_argument("-workers", "--num_workers", default=1)
    parser.add_argument("-use_dev", "--use_dev", default=False)
    parser.add_argument("-tok_path", "--tok_dir", default=TOKENIZER_CORE_PATH)
    parser.add_argument("-glove_dim", "--glove_dimension", default=50)
    args=parser.parse_args()

    workers = int(args.num_workers)
    use_dev = str(args.use_dev)
    tok_path = args.tok_dir
    glove_dim=int(args.glove_dimension)
    #Create vocabulary:
    os.system("python3 vocabulary.py -workers %d -use_dev %s -tok_path \"%s\""%(workers,use_dev,tok_path) )

    #Filtering Vocabulary:
    os.system("python3 known_words.py -glove_dim %d"%glove_dim)

    #Create w2i and i2w
    os.system("python3 indexNword.py")

    #Create embedding_init:
    os.system("python3 embedding_init.py -glove_dim %d"%glove_dim)

    #Preprocessing_1:
    os.system("python3 data_preprocess_1.py -workers %d -tok_path %s"%(workers,tok_path))

    #Preprocessing_2:
    os.system("python3 data_preprocess_2.py ")
