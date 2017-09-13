import re
import string
from collections import Counter
import random

import tensorflow as tf
from tensorflow.python.ops import math_ops,array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
# from tensorflow.python.ops.rnn_cell import BasicRNNCell
import numpy as np
import pickle
from tqdm import tqdm
max_context=400
max_question=30
dimension=50
hidden_size=128

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
def ExactMatch(ground,predict):
    return (normalize_answer(ground)==normalize_answer(predict))
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def mask_label(label,fixed_length_context=400):
    result=np.zeros(shape=fixed_length_context)
    result[label]=1
    return result
def length_(sequence_numpy,pad_id=1):
    bool_index=np.array(sequence_numpy)!=pad_id
    length=np.sum(bool_index)
    return length
########################################
data_train_path="/home/dangtran/Desktop/Python_Projects/Kaggle/Squad_Match_LSTM/data/train/data_filtered"
train=pickle.load(open(data_train_path,"rb"))

data_val_path="/home/dangtran/Desktop/Python_Projects/Kaggle/Squad_Match_LSTM/data/val/val_filtered"
val=pickle.load(open(data_val_path,"rb"))
###############
cid2indexed_c_path="/home/dangtran/Desktop/Python_Projects/Kaggle/Squad_Match_LSTM/data/train/cid2indexed_c"
cid2indexed_c=pickle.load(open(cid2indexed_c_path,"rb"))
embedding_np_path="/home/dangtran/Desktop/Python_Projects/Kaggle/Squad_Match_LSTM/data/utility/embedding_np"
embedding_np=np.load(embedding_np_path)

val_cid2padding_c_path="/home/dangtran/Desktop/Python_Projects/Kaggle/Squad_Match_LSTM/data/val/val_cid2padding_c"

########################################

embedding_np_path="/home/dangtran/Desktop/Python_Projects/Kaggle/Squad_Match_LSTM/data/utility/embedding_np"
embedding_np=np.load(embedding_np_path)
embedding_np_size=embedding_np.shape


embedding_matrix_=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=embedding_np_size),
                             trainable=False)
embedding_input_=tf.placeholder(shape=embedding_np_size,dtype=tf.float32)
embedding_init_=embedding_matrix_.assign(embedding_input_)

p_=tf.placeholder(shape=(None,max_context),dtype=tf.int32)
q_=tf.placeholder(shape=(None,max_question),dtype=tf.int32)
p_vector_=tf.nn.embedding_lookup(embedding_matrix_,p_)
q_vector_=tf.nn.embedding_lookup(embedding_matrix_,q_)
p_length_=tf.placeholder(shape=(None),dtype=tf.int32)
q_length_=tf.placeholder(shape=(None),dtype=tf.int32)

label_s_=tf.placeholder(shape=(None),dtype=tf.int32)
label_e_=tf.placeholder(shape=(None),dtype=tf.int32)
label_s_vector_=tf.placeholder(shape=(None,max_context),dtype=tf.float32)
label_e_vector_=tf.placeholder(shape=(None,max_context),dtype=tf.float32)

##########Encoder:
# list_p_vector_=tf.unstack(p_vector_,axis=1)
# print(len(list_p_vector_))
with tf.variable_scope("encoding_passage"):
    encoder_passage_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
    encoded_q,(q_state,_)=tf.nn.dynamic_rnn(encoder_passage_cell,q_vector_,q_length_,dtype=tf.float32)
with tf.variable_scope("encoding_question"):
    encoder_question_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
    encoded_p,(p_state,_)=tf.nn.dynamic_rnn(encoder_question_cell,p_vector_,p_length_,dtype=tf.float32)


W_q_=tf.Variable(tf.random_normal(shape=(hidden_size,hidden_size),dtype=tf.float32))
class Match_LSTM_cell(tf.nn.rnn_cell.RNNCell):
    def __init__(self,encoded_q_vectors,w_q,num_units,forget_bias=1.0,activation=None,reuse=None,name="1"):
        # super(Match_LSTM_cell, self).__init__(_reuse=reuse)
        self._num_units=num_units
        self._forget_bias=forget_bias
        self._activation=activation or math_ops.tanh
        self._encoded_q_vectors=encoded_q_vectors
        self._w_matrix=w_q
        self._name=name
        self.reuse=reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs,state):
        c,h=state
        # ######Check if variable Exist             //Bad performance
        # try:
        #     with tf.variable_scope("share_parameter", reuse=True):
        #         W_p = tf.get_variable(name="W_p", shape=(self._num_units, self._num_units))
        # except ValueError:
        #     with tf.variable_scope("share_parameter"):
        #         W_p = tf.get_variable(name="W_p", shape=(self._num_units, self._num_units),dtype=tf.float32,
        #                               initializer=tf.random_uniform_initializer)
        #         W_r = tf.get_variable(name="W_r", shape=(self._num_units, self._num_units),dtype=tf.float32,
        #                               initializer=tf.random_uniform_initializer)
        #         b_p = tf.get_variable(name="b_p", shape=(self._num_units),dtype=tf.float32,
        #                               initializer=tf.ones_initializer)
        #         w = tf.get_variable(name="w", shape=(self._num_units, 1),dtype=tf.float32,
        #                             initializer=tf.random_uniform_initializer)
        #         b = tf.get_variable(name="b", shape=(max_question),dtype=tf.float32,
        #                             initializer=tf.ones_initializer)
        #         W_concat = tf.get_variable(name="W_concat", shape=(self._num_units * 3, self._num_units * 4),
        #                                    dtype=tf.float32,
        #                                    initializer=tf.random_uniform_initializer)
        #         b_concat = tf.get_variable(name="b_concat", shape=(self._num_units * 4),
        #                                    dtype=tf.float32,
        #                                    initializer=tf.ones_initializer)
        #
        # with tf.variable_scope("share_parameter",reuse=True):
        #     W_p = tf.get_variable(name="W_p", shape=(self._num_units, self._num_units))
        #     W_r = tf.get_variable(name="W_r", shape=(self._num_units, self._num_units))
        #     b_p = tf.get_variable(name="b_p", shape=(self._num_units))
        #     w = tf.get_variable(name="w", shape=(self._num_units, 1))
        #     b = tf.get_variable(name="b", shape=(max_question))
        #     W_concat = tf.get_variable(name="W_concat", shape=(self._num_units * 3, self._num_units * 4))
        #     b_concat = tf.get_variable(name="b_concat", shape=(self._num_units * 4))
        #
        #
        #
        # #######Finding g
        # flatten_q=tf.reshape(self._encoded_q_vectors,(-1,self._num_units))
        # subg_q=tf.matmul(flatten_q,self._w_matrix)
        # subg_q=tf.reshape(subg_q,(-1,max_question,self._num_units))
        #
        # subg_p=tf.matmul(inputs,W_p)
        # subg_r=tf.matmul(c,W_r)
        # subg_=subg_r+subg_p+b_p
        # subg_=tf.tile(tf.expand_dims(subg_,axis=1),(1,max_question,1))
        # g=self._activation(subg_+subg_q)
        #
        # #######Finding alpha
        # flatten_g=tf.reshape(g,(-1,self._num_units))
        # w_g=tf.matmul(flatten_g,w)
        # w_g=tf.reshape(w_g,(-1,max_question))
        #
        # alpha=w_g+b
        # alpha=tf.reshape(alpha,(-1,max_question,1))
        # alpha=tf.nn.softmax(alpha,dim=1)
        #
        # #######Finding z
        # reshape_alpha=tf.transpose(alpha,(0,2,1))
        # alpha_h=tf.reshape(tf.matmul(reshape_alpha,self._encoded_q_vectors),(-1,self._num_units))
        # z=tf.concat([inputs,alpha_h],axis=-1)
        #
        # #######Core LSTM
        # sigmoid=math_ops.sigmoid
        # concat_tmp_=tf.concat([z,h],-1)
        # concat=tf.matmul(concat_tmp_,W_concat)+b_concat
        # # concat=_linear([z,c],self._num_units*4,bias=True)
        # i,j,f,o=array_ops.split(concat,num_or_size_splits=4,axis=1)
        # new_c=(c*sigmoid(f+self._forget_bias) + sigmoid(i)*self._activation(j))
        # new_h=self._activation(new_c)*sigmoid(o)
        # new_state=(new_c,new_h)
        # return new_h,new_state


        ###############Use _linear only:
        if self.reuse !=True:
            ###########Finding  g:
            flatten_q=tf.reshape(self._encoded_q_vectors,(-1,self._num_units))
            subg_q=tf.matmul(flatten_q,self._w_matrix)
            subg_q=tf.reshape(subg_q,(-1,max_question,self._num_units))

            with tf.variable_scope("sub_g_hidden_part"):
                subg_h=_linear([inputs,h],self._num_units,bias=True)
            subg_h_tiled=tf.tile(tf.expand_dims(subg_h,1),(1,max_question,1))
            g=math_ops.tanh(subg_q+subg_h_tiled)

            ##########Finding alpha:
            flatten_g=tf.reshape(g,(-1,self._num_units))
            with tf.variable_scope("pre_alpha"):
                pre_alpha=_linear([flatten_g],1,bias=True)
                pre_alpha=tf.reshape(pre_alpha,(-1,max_question,1))
            alpha=tf.nn.softmax(pre_alpha,dim=1)

            ##########Finding z:
            reshaped_alpha=tf.transpose(alpha,(0,2,1))
            subz_q=tf.reshape(tf.matmul(reshaped_alpha,self._encoded_q_vectors),(-1,self._num_units))
            z=tf.concat([inputs,subz_q],axis=-1)

            ##########LSTM CORE:
            sigmoid=math_ops.sigmoid
            with tf.variable_scope("linear"):
                concat=_linear([z,h],self._num_units*4,bias=True)
            i,j,f,o=array_ops.split(concat,num_or_size_splits=4,axis=1)
            new_c=(c*sigmoid(f+self._forget_bias)+sigmoid(i)*self._activation(j))
            new_h=self._activation(new_c)*sigmoid(o)
            new_state=(new_c,new_h)
            return new_h,new_state
        elif self.reuse==True:
            ############Get existed parameters
            # W_subg_hidden = tf.Variable(tf.zeros(shape=(self._num_units*2, self._num_units),dtype=tf.float32))
            # b_subg_hidden = tf.Variable(tf.zeros(shape=(self._num_units),dtype=tf.float32))
            #
            # W_pre_alpha=tf.Variable(tf.zeros(shape=(self._num_units,1),dtype=tf.float32))
            # b_pre_alpha=tf.Variable(tf.zeros(shape=(1),dtype=tf.float32))
            #
            # W_linear=tf.Variable(tf.zeros(shape=(self._num_units*3,self._num_units*4),dtype=tf.float32))
            # b_linear=tf.Variable(tf.zeros(shape=(self._num_units*4),dtype=tf.float32))

            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="rnn/sub_g_hidden_part"):
                sub_name="rnn/sub_g_hidden_part"
                if var.name==sub_name+"/kernel:0":
                    W_subg_hidden=var
                    print("True")
                if var.name==sub_name+"/bias:0":
                    b_subg_hidden=var
                    print("True")
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="rnn/pre_alpha"):
                sub_name="rnn/pre_alpha"
                if var.name==sub_name+"/kernel:0":
                    W_pre_alpha=var
                    print("True")
                if var.name==sub_name+"/bias:0":
                    b_pre_alpha=var
                    print("True")
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="rnn/linear"):
                sub_name="rnn/linear"
                if var.name==sub_name+"/kernel:0":
                    W_linear=var
                    print("True")
                if var.name==sub_name+"/bias:0":
                    b_linear=var
                    print("True")

            ###########Finding g:
            flatten_q = tf.reshape(self._encoded_q_vectors, (-1, self._num_units))
            subg_q = tf.matmul(flatten_q, self._w_matrix)
            subg_q = tf.reshape(subg_q, (-1, max_question, self._num_units))

            concat_h_p=tf.concat([inputs,h],axis=-1)

            subg_h=tf.matmul(concat_h_p,W_subg_hidden)+b_subg_hidden
            subg_h_tiled=tf.tile(tf.expand_dims(subg_h,dim=1),(1,max_question,1))
            g=math_ops.tanh(subg_q+subg_h_tiled)

            ##########Finding alpha
            flatten_g=tf.reshape(g,(-1,self._num_units))
            pre_alpha=tf.matmul(flatten_g,W_pre_alpha)+b_pre_alpha
            pre_alpha=tf.reshape(pre_alpha,(-1,max_question,1))
            alpha=tf.nn.softmax(pre_alpha,dim=1)

            ##########Finding z:
            reshaped_alpha=tf.transpose(alpha,(0,2,1))
            subz_q=tf.reshape(tf.matmul(reshaped_alpha,self._encoded_q_vectors),(-1,self._num_units))
            z=tf.concat([inputs,subz_q],axis=-1)

            ##########LSTM CORE:
            sigmoid=math_ops.sigmoid
            concat_input_=tf.concat([z,h],axis=-1)
            concat=tf.matmul(concat_input_,W_linear)+b_linear

            i,j,f,o=array_ops.split(concat,num_or_size_splits=4,axis=1)
            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)
            new_state = (new_c, new_h)
            return new_h, new_state

encoded_p_list=tf.unstack(encoded_p,axis=1)
init_state=(tf.zeros_like(encoded_p_list[0]),tf.zeros_like(encoded_p_list[0]))
fw_cell=Match_LSTM_cell(encoded_q,W_q_,num_units=128,forget_bias=1.0,activation=math_ops.tanh,name="fw",reuse=False)
h_fw,h_fw_states=tf.nn.dynamic_rnn(fw_cell,inputs=encoded_p,sequence_length=p_length_,dtype=tf.float32,initial_state=init_state)

fw_cell.reuse=True
reversed_encoded_p=array_ops.reverse_sequence(input=encoded_p,seq_lengths=p_length_,seq_axis=1,batch_axis=0)
reversed_h_bw,h_bw_states=tf.nn.dynamic_rnn(fw_cell,inputs=reversed_encoded_p,sequence_length=p_length_,dtype=tf.float32,initial_state=init_state)
h_bw=array_ops.reverse_sequence(input=reversed_h_bw,seq_lengths=p_length_,seq_axis=1,batch_axis=0)

h=tf.concat([h_fw,h_bw],axis=-1)

# h_tmp=tf.placeholder(shape=(None,max_context,dimension*2),dtype=tf.float32)   // Use for test only
# h_tmp_list=tf.unstack(h_tmp,axis=1)                                           // Use for test only
# flatten_=tf.reshape(h,(-1,hidden_size*2))
# tmp=tf.matmul(flatten_,w_h_)


w_h_=tf.Variable(tf.random_normal(shape=(hidden_size*2,hidden_size),dtype=tf.float32))
zero_input=tf.zeros_like(h)
class Pointer_Net_cell_boundary(tf.nn.rnn_cell.RNNCell):
    ############In this cell, Input for this cell is some kind special
    ############We don't need to have input for separate cells
    ############However, we must have a input vector in dynamic_rnn,
    # => We can use a zero_input to refill the cell
    def __init__(self,h,w_h,num_units,forget_bias=1.0,activation=None,reuse=None,name="1"):
        # super(Pointer_Net_cell_boundary,self).__init__(_reuse=reuse)
        self._num_units=num_units
        self._forget_bias=forget_bias
        self._activation=activation or math_ops.tanh
        self._H=h
        self._W_h=w_h
        self._name=name
        self._logit_list=[]
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs,state):
        c,h=state
        # try :         ##########Bad performance
        #     with vs.variable_scope("share_parameter_2",reuse=True):
        #         W_a=tf.get_variable("W_a",shape=(self._num_units,self._num_units))
        # except ValueError:
        #     with vs.variable_scope("share_parameter_2"):
        #         W_a=tf.get_variable("W_a",shape=(self._num_units,self._num_units),
        #                             dtype=tf.float32,
        #                             initializer=tf.random_uniform_initializer)
        #         b_a=tf.get_variable("b_a",shape=(self._num_units),
        #                             dtype=tf.float32,
        #                             initializer=tf.ones_initializer)
        #         v=tf.get_variable("v",shape=(self._num_units,1),
        #                           dtype=tf.float32,
        #                           initializer=tf.random_uniform_initializer)
        #         c_v=tf.get_variable("c_v",shape=(max_context),
        #                             dtype=tf.float32,
        #                             initializer=tf.ones_initializer)
        #         W_concat=tf.get_variable("W_concat",shape=(self._num_units*3,self._num_units*4),
        #                                  dtype=tf.float32,
        #                                  initializer=tf.random_uniform_initializer)
        #         c_concat=tf.get_variable("c_concat",shape=(self._num_units*4),
        #                                  dtype=tf.float32,
        #                                  initializer=tf.ones_initializer)
        # with vs.variable_scope("share_parameter_2",reuse=True):
        #     W_a = tf.get_variable("W_a", shape=(self._num_units, self._num_units))
        #     b_a = tf.get_variable("b_a", shape=(self._num_units))
        #     v = tf.get_variable("v", shape=(self._num_units, 1))
        #     c_v = tf.get_variable("c_v", shape=(max_context))
        #     W_concat = tf.get_variable("W_concat", shape=(self._num_units * 3, self._num_units * 4))
        #     c_concat = tf.get_variable("c_concat", shape=(self._num_units * 4))
        #
        # ###########Calculate f
        # flatten_H=tf.reshape(self._H,(-1,self._num_units*2))
        # subf_h=tf.matmul(flatten_H,self._W_h)
        # subf_h=tf.reshape(subf_h,(-1,max_context,self._num_units))
        #
        # subf_a=tf.matmul(c,W_a)+b_a
        # subf_a_tiled=tf.tile(tf.expand_dims(subf_a,axis=1),(1,max_context,1))
        #
        # f=math_ops.tanh(subf_h+subf_a_tiled)
        #
        # ##########Calculate beta
        # flatten_f=tf.reshape(f,(-1,self._num_units))
        # subbeta_=tf.matmul(flatten_f,v)
        # subbeta_=tf.reshape(subbeta_,(-1,max_context))
        # logit=subbeta_+c_v
        # beta=tf.reshape(logit,(-1,max_context,1))
        # beta=tf.nn.softmax(beta,dim=1)
        #
        # ##########Core LSTM:
        # sigmoid=math_ops.sigmoid
        # reshaped_beta=tf.transpose(beta,(0,2,1))
        # input_beta=tf.reshape(tf.matmul(reshaped_beta,self._H),(-1,self._num_units*2))
        # input_=tf.concat([input_beta,h],axis=-1)
        # concat=tf.matmul(input_,W_concat)+c_concat
        # i,j,f,o=array_ops.split(concat,num_or_size_splits=4,axis=1)
        # new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        # new_h= self._activation(new_c)*sigmoid(o)
        #
        # new_state=(new_c,new_h)
        # self._beta_list.append(logit)
        #
        # return new_h,new_state

        #########Use _linear only:
        #####Find f:
        flatten_h=tf.reshape(self._H,(-1,self._num_units*2))
        subf_h=tf.matmul(flatten_h,self._W_h)
        subf_h=tf.reshape(subf_h,shape=(-1,max_context,self._num_units))

        with tf.variable_scope("sub_f_hidden_part"):
            subf_hidden=_linear([h],self._num_units,bias=True)
        subf_hidden_tiled=tf.tile(tf.expand_dims(subf_hidden,axis=1),(1,max_context,1))

        f=math_ops.tanh(subf_h+subf_hidden_tiled)

        #####Find beta:
        flatten_f=tf.reshape(f,(-1,self._num_units))
        with tf.variable_scope("pre_beta"):
            pre_beta=_linear([flatten_f],1,True)
        pre_beta=tf.reshape(pre_beta,(-1,max_context,1))
        logit=tf.reshape(pre_beta,(-1,max_context))
        self._logit_list.append(logit)
        beta=tf.nn.softmax(pre_beta,dim=1)

        #####Core LSTM:
        sigmoid=math_ops.sigmoid
        reshaped_beta=tf.transpose(beta,(0,2,1))
        input_=tf.reshape(tf.matmul(reshaped_beta,self._H),(-1,self._num_units*2))
        with tf.variable_scope("linear_v2"):
            concat=_linear([input_,h],self._num_units*4,bias=True)
        i,j,f,o=array_ops.split(concat,num_or_size_splits=4,axis=1)
        new_c=(c*sigmoid(f+self._forget_bias)+sigmoid(i)*self._activation(j))
        new_h=self._activation(new_c)*sigmoid(o)
        new_state=(new_c,new_h)
        return new_h,new_state


decoder_cell=Pointer_Net_cell_boundary(h,w_h_,num_units=128)
output_,decoder_state=tf.nn.static_rnn(decoder_cell,inputs=[label_s_vector_,label_e_vector_],dtype=tf.float32,initial_state=init_state)
logit_s=decoder_cell._logit_list[0]
logit_e=decoder_cell._logit_list[1]

# loss_s=tf.nn.softmax_cross_entropy_with_logits(logits=logit_s,labels=label_s_vector_)
# loss=tf.reduce_mean(loss_s)
# train_step=tf.train.AdamOptimizer().minimize(loss)

loss_s=tf.nn.softmax_cross_entropy_with_logits(logits=logit_s,labels=label_s_vector_)
loss_e=tf.nn.softmax_cross_entropy_with_logits(logits=logit_e,labels=label_e_vector_)
loss=tf.reduce_mean(loss_s+loss_e)
train_step=tf.train.AdamOptimizer().minimize(loss)

prob_s=tf.nn.softmax(logit_s)
prob_e=tf.nn.softmax(logit_e)

final_ps=tf.argmax(prob_s,dimension=1)
final_pe=tf.argmax(prob_e,dimension=1)

def evaluate(ep,Session,val_data, cid2padding_c_path, val_batch_size=64):
    cid2padding_c = pickle.load(open(cid2padding_c_path, "rb"))
    range_ = np.arange(0, len(val_data), val_batch_size)
    predicts_answ = []
    grounds_answ = []
    EM_score = []
    F1_score = []
    final_predict=[]
    final_ground=[]
    for iter, i in enumerate(tqdm(range_, desc="Val %d:"%ep)):
        if iter == len(val_data) // val_batch_size:
            # continue
            mini_batch = val_data[i:]
            cid_list=[]
            qid_list=[]
            c_list=[]
            q_list=[]
            answ_s_list=[]
            answ_e_list=[]
            mask_p_list=[]
            mask_q_list=[]
            answ_s_vector_list=[]
            answ_e_vector_list=[]
            for cid,qid,c,q,answ_s,answ_e in mini_batch:
                cid_list.append(cid)
                qid_list.append(qid)
                c_list.append(c)
                q_list.append(q)
                answ_s_list.append([answ_s])
                answ_e_list.append([answ_e])
                answ_s_vector=mask_label(answ_s)
                answ_e_vector=mask_label(answ_e)
                answ_s_vector_list.append(answ_s_vector)
                answ_e_vector_list.append(answ_e_vector)
                length_q=length_(q)
                length_p=length_(c)
                mask_p_list.append(length_p)
                mask_q_list.append(length_q)
            cid_list=np.array(cid_list)
            qid_list=np.array(qid_list)
            c_list=np.array(c_list)
            q_list=np.array(q_list)
            answ_s_list=np.array(answ_s_list)
            answ_e_list=np.array(answ_e_list)
            answ_s_vector_list=np.array(answ_s_vector_list)
            answ_e_vector_list=np.array(answ_e_vector_list)
            mask_p_list = np.array(mask_p_list)
            mask_q_list = np.array(mask_q_list)
            feed_train={
                p_:c_list,
                q_:q_list,
                p_length_:mask_p_list,
                q_length_:mask_q_list,
                label_s_:answ_s_list,
                label_e_:answ_e_list,
                label_s_vector_:answ_s_vector_list,
                label_e_vector_:answ_e_vector_list
                }
            predict_s=Session.run(final_ps,feed_train)
            predict_e=Session.run(final_pe,feed_train)
            for ps,pe,cid,ts,te in zip(predict_s,predict_e,cid_list,answ_s_list.ravel(),answ_e_list.ravel()):
                context=cid2padding_c[cid]
                answ_true=" ".join(context[ts:te])
                answ_predict=" ".join(context[ps:pe])
                final_predict.append(answ_predict)
                final_ground.append(answ_true)
        else:
            mini_batch = val_data[i:i + val_batch_size]
            cid_list=[]
            qid_list=[]
            c_list=[]
            q_list=[]
            answ_s_list=[]
            answ_e_list=[]
            mask_p_list=[]
            mask_q_list=[]
            answ_s_vector_list=[]
            answ_e_vector_list=[]
            for cid,qid,c,q,answ_s,answ_e in mini_batch:
                cid_list.append(cid)
                qid_list.append(qid)
                c_list.append(c)
                q_list.append(q)
                answ_s_list.append([answ_s])
                answ_e_list.append([answ_e])
                answ_s_vector=mask_label(answ_s)
                answ_e_vector=mask_label(answ_e)
                answ_s_vector_list.append(answ_s_vector)
                answ_e_vector_list.append(answ_e_vector)
                length_q=length_(q)
                length_p=length_(c)
                mask_p_list.append(length_p)
                mask_q_list.append(length_q)
            cid_list=np.array(cid_list)
            qid_list=np.array(qid_list)
            c_list=np.array(c_list)
            q_list=np.array(q_list)
            answ_s_list=np.array(answ_s_list)
            answ_e_list=np.array(answ_e_list)
            answ_s_vector_list=np.array(answ_s_vector_list)
            answ_e_vector_list=np.array(answ_e_vector_list)
            mask_p_list = np.array(mask_p_list)
            mask_q_list = np.array(mask_q_list)
            feed_train={
                p_:c_list,
                q_:q_list,
                p_length_:mask_p_list,
                q_length_:mask_q_list,
                label_s_:answ_s_list,
                label_e_:answ_e_list,
                label_s_vector_:answ_s_vector_list,
                label_e_vector_:answ_e_vector_list
                }
            predict_s=Session.run(final_ps,feed_train)
            predict_e=Session.run(final_pe,feed_train)
            for ps,pe,cid,ts,te in zip(predict_s,predict_e,cid_list,answ_s_list.ravel(),answ_e_list.ravel()):
                context=cid2padding_c[cid]
                answ_true=" ".join(context[ts:te])
                answ_predict=" ".join(context[ps:pe])
                final_predict.append(answ_predict)
                final_ground.append(answ_true)
    for p,t in zip(final_predict,final_ground):
        EM_score.append(ExactMatch(p,t))
        F1_score.append(f1_score(p,t))
    print("EM %f" % np.average(np.array(EM_score)))
    print("F1_score %f" % np.average(np.array(F1_score)))
    return np.average(np.array(EM_score)), np.average(np.array(F1_score))

#######Init
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
feed_embedding={embedding_input_:embedding_np}
sess.run(embedding_init_,feed_dict=feed_embedding)

#######Test 1 element:
# data_tmp=train[0]
# cid,qid,c,q,answ_s,answ_e=data_tmp
# cid=np.array([cid])
# qid=np.array([qid])
# c=np.array([c])
# q=np.array([q])
# answ_s_vector=np.array([mask_label(answ_s)])
# answ_e_vector=np.array([mask_label(answ_e)])
# answ_s=np.array([[answ_s]])
# answ_e=np.array([[answ_e]])
# length_p=np.array([length_(c)])
# length_q=np.array([length_(q)])
#
# feed_train={p_:c,
#             q_:q,
#             p_length_:length_p,
#             q_length_:length_q,
#             label_s_:answ_s,
#             label_e_:answ_e,
#             label_s_vector_:answ_s_vector,
#             label_e_vector_:answ_e_vector}
#
# for i in range(300):
#     loss_=sess.run(loss,feed_train)
#     print(loss_)
#     sess.run(train_step,feed_train)

#######Test batch element:
# data_tmp=train[:10]
# cid_list=[]
# qid_list=[]
# c_list=[]
# q_list=[]
# answ_s_list=[]
# answ_e_list=[]
# c_length_list=[]
# q_length_list=[]
# answ_s_vector_list=[]
# answ_e_vector_list=[]
# for cid,qid,c,q,answ_s,answ_e in data_tmp:
#     cid_list.append(cid)
#     qid_list.append(qid)
#     c_list.append(c)
#     q_list.append(q)
#     answ_s_list.append([answ_s])
#     answ_e_list.append([answ_e])
#     answ_s_vector=mask_label(answ_s)
#     answ_e_vector=mask_label(answ_e)
#     answ_s_vector_list.append(answ_s_vector)
#     answ_e_vector_list.append(answ_e_vector)
#     length_q=length_(q)
#     length_p=length_(c)
#     c_length_list.append(length_p)
#     q_length_list.append(length_q)
# cid_list=np.array(cid_list)
# qid_list=np.array(qid_list)
# c_list=np.array(c_list)
# q_list=np.array(q_list)
# answ_s_list=np.array(answ_s_list)
# answ_e_list=np.array(answ_e_list)
# c_length_list=np.array(c_length_list)
# q_length_list=np.array(q_length_list)
# answ_s_vector_list=np.array(answ_s_vector_list)
# answ_e_vector_list=np.array(answ_e_vector_list)
# feed_train={
#     p_:c_list,
#     q_:q_list,
#     p_length_:c_length_list,
#     q_length_:q_length_list,
#     label_s_:answ_s_list,
#     label_e_:answ_e_list,
#     label_s_vector_:answ_s_vector_list,
#     label_e_vector_:answ_e_vector_list
#     }
# loss_=sess.run(loss,feed_train)
# # print(loss_)
# for i in range(100):
#     loss_=sess.run(loss,feed_train)
#     print(loss_)
#     sess.run(train_step,feed_train)


############Actually train:
batch_size=32
# train=train[:1000]
range_=np.arange(0,len(train),batch_size)
# evaluate(-1,sess,val,val_cid2padding_c_path)
epoch=15
for ep in range(epoch):
    random.shuffle(train)
    for iter,i in enumerate(tqdm(range_,desc="Epoch %d"%ep)):
        if iter==len(train)//batch_size:
            # continue
            mini_batch=train[i:]
            cid_list=[]
            qid_list=[]
            c_list=[]
            q_list=[]
            answ_s_list=[]
            answ_e_list=[]
            c_length_list=[]
            q_length_list=[]
            answ_s_vector_list=[]
            answ_e_vector_list=[]
            for cid,qid,c,q,answ_s,answ_e in mini_batch:
                # cid_list.append(cid)
                # qid_list.append(qid)
                c_list.append(c)
                q_list.append(q)
                answ_s_list.append([answ_s])
                answ_e_list.append([answ_e])
                answ_s_vector=mask_label(answ_s)
                answ_e_vector=mask_label(answ_e)
                answ_s_vector_list.append(answ_s_vector)
                answ_e_vector_list.append(answ_e_vector)
                length_q=length_(q)
                length_p=length_(c)
                c_length_list.append(length_p)
                q_length_list.append(length_q)
            cid_list=np.array(cid_list)
            qid_list=np.array(qid_list)
            c_list=np.array(c_list)
            q_list=np.array(q_list)
            answ_s_list=np.array(answ_s_list)
            answ_e_list=np.array(answ_e_list)
            c_length_list=np.array(c_length_list)
            q_length_list=np.array(q_length_list)
            answ_s_vector_list=np.array(answ_s_vector_list)
            answ_e_vector_list=np.array(answ_e_vector_list)
            feed_train={
                p_:c_list,
                q_:q_list,
                p_length_:c_length_list,
                q_length_:q_length_list,
                label_s_:answ_s_list,
                label_e_:answ_e_list,
                label_s_vector_:answ_s_vector_list,
                label_e_vector_:answ_e_vector_list
                }
            loss_=sess.run(loss,feed_train)
            sess.run(train_step,feed_train)
        else:
            mini_batch = train[i:i+batch_size]
            cid_list = []
            qid_list = []
            c_list = []
            q_list = []
            answ_s_list = []
            answ_e_list = []
            c_length_list = []
            q_length_list = []
            answ_s_vector_list = []
            answ_e_vector_list = []
            for cid, qid, c, q, answ_s, answ_e in mini_batch:
                # cid_list.append(cid)
                # qid_list.append(qid)
                c_list.append(c)
                q_list.append(q)
                answ_s_list.append([answ_s])
                answ_e_list.append([answ_e])
                answ_s_vector = mask_label(answ_s)
                answ_e_vector = mask_label(answ_e)
                answ_s_vector_list.append(answ_s_vector)
                answ_e_vector_list.append(answ_e_vector)
                length_q = length_(q)
                length_p = length_(c)
                c_length_list.append(length_p)
                q_length_list.append(length_q)
            cid_list = np.array(cid_list)
            qid_list = np.array(qid_list)
            c_list = np.array(c_list)
            q_list = np.array(q_list)
            answ_s_list = np.array(answ_s_list)
            answ_e_list = np.array(answ_e_list)
            c_length_list = np.array(c_length_list)
            q_length_list = np.array(q_length_list)
            answ_s_vector_list = np.array(answ_s_vector_list)
            answ_e_vector_list = np.array(answ_e_vector_list)
            feed_train = {
                p_: c_list,
                q_: q_list,
                p_length_: c_length_list,
                q_length_: q_length_list,
                label_s_: answ_s_list,
                label_e_: answ_e_list,
                label_s_vector_: answ_s_vector_list,
                label_e_vector_: answ_e_vector_list
            }
            loss_ = sess.run(loss, feed_train)
            sess.run(train_step, feed_train)
    evaluate(ep,sess,val,val_cid2padding_c_path)
