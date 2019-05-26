# -*- coding: utf-8 -*-

import yaml
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import model_from_yaml
import keras
np.random.seed(1337)  # For Reproducibility
import jieba
import sys
import tensorflow as tf
sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 100
cpu_count = multiprocessing.cpu_count()


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('./lstm_model/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

def load_model():
    #tf.get_default_graph()
    #print('loading model......')
    with open('./lstm_model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    #print('loading weights......')
    model.load_weights('./lstm_model/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model

def lstm_predict(string, model):

    data=input_transform(string)
    data.reshape(1,-1)
    #print data
    result=model.predict_proba(data)
    #print(result)
    keras.backend.clear_session()
    return result[0][0]

if __name__=='__main__':
    model = load_model()
    #train()
    #string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    # string='酒店的环境非常好，价格也便宜，值得推荐'
    # string='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    # string='我是傻逼'
    # string='你是傻逼'
    # string='屏幕较差，拍照也很粗糙。'
    # string='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    string='东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'

    print(lstm_predict(string,load_model()))
