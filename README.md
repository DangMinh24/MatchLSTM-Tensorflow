# MatchLSTM-Tensorflow
  
## [Reference papers by Shuohang Wang and Jing Jiang](https://arxiv.org/abs/1608.07905)

# General:
Shuohang Wang and Jing Jiang use previous researchs (MatchLSTM on [textual entailment](https://en.wikipedia.org/wiki/Textual_entailment)) applying Question Answering problem (on [SQuAD-dataset](https://rajpurkar.github.io/SQuAD-explorer/)), which got some promised results.

# Goals:
There are many reimplementations of MatchLSTM (and [author's implementation](https://github.com/shuohangwang/SeqMatchSeq) which using Torch7). However, there is a shortage
of reimplementations in tensorflow, which can give us a better view the basic ideas and simple structure of this interesting model.

### Requirments:
* Downloads train set and dev set from [SQuAD data set](https://rajpurkar.github.io/SQuAD-explorer/), save in data/ directory
* Downloads all required python libraries:
```
  sudo pip3 install -r requirements.txt
```
* Downloads StanfordCoreNLP and put into data/tokenizer (or specific your tokenizer's path)

* Downloads [Glove](https://nlp.stanford.edu/projects/glove/) word2vec and put into data/glove

# Code Flows:
### Preprocessing:

This will take some time to finish preprocessing stage

```python
  python3 preprocess.py
```
(Including these steps below):

* Create vocabulary
```python
    python3 vocabulary.py 
```

* Preprocess Glove word2vec and filter vocabulary
```python
    python3 known_words.py
```

* Representing word in number format (create word2index and index2word) - Any words out of filtered vocabulary size should marked as <UNK>
```python
    python3 indexNword.py
```

* Create embedding matrix
```python
    python3 embedding_init
```

* From original json file, extract features for each **question_context** pair in __character-level__. Convert to **word-level** 
```python
  python3 data_preprocess_1.py
```

* From word-level data, filtering unknown tokens, and padding to fixed size (max_context=400,max_question=30)
```python
  python3 data_preprocess_2.py
```
  
### Main flow of the model:
This is the core of the program

```python
  python3 model.py 
```

I try to reimplement all the steps of orginal papers with include:

1. Encoder Layer: Encoding **question** and **context** into fixed vector by BasicLSTM cells
2. MatchLSTM Layer: Which is customized from LSTM so that it can apply attention mechanism, compute weighted_vector (in 2 flow directions)
  * **Note**: According to authors, parameters in MatchLSTM should be shared (or reuse) in reversed direction, which will require you to know how to deal with parameters sharing in LSTM. My current code is somehow like a hacking code, which is quite clumsy. If you have any question about the code, please send me an [email](dangtm24@gmail.com) 
3. PointerNet-Boundary (Decoder): Also be customized from RNN cell, so that it can apply another attention mechanism before computing the outputs
  * **Note 1**: I reimplement model in **Boundary Mode** - which only attempt to predict start_index and end_index. However, if you want to reimplement model in **Sequence Mode**, you can customize your 3. slightly
4. Evaluate: Use standard evaluation method as 
  * EM(Exact Match): Only exactly predicted answer can be consider as True, else False
  * F1_score : This method to calculate the precision and recall
  * **Note**:This code is mainly derived from [SQuAD homepage](https://rajpurkar.github.io/SQuAD-explorer/). However, give yourself a peek to know how it works.

**Important:** All training process and validation process is implemented through mini-batch (which will overcome memory limit problem on your GPU/CPU). Try to reduce size of mini-batch so that model can work appropriately with your system


