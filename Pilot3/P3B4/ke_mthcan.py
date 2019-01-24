import os
import numpy as np
import keras
from keras.layers import Input, Embedding, Dropout, Activation, TimeDistributed, Permute, Lambda, Conv1D, Dot, Dense
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.engine.topology import Layer
import random
import sys
import time

class target_att(Layer):
    def __init__(self,attention_size,**kwargs):
        self.attention_size = attention_size
        super(target_att,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.Q = K.variable(np.random.rand(1,1,self.attention_size)*0.1-0.05)
        self.trainable_weights = [self.Q]        
        super(target_att,self).build(input_shape)

    def call(self,x):
        return K.batch_dot(self.Q,K.permute_dimensions(x,(0,2,1)))/np.sqrt(self.attention_size)

    def compute_output_shape(self,input_shape):
        return (1,1,input_shape[1])

class hcan(object):
    #def __init__(self,embedding_matrix,num_classes,max_sents,max_words,
    #             attention_size=512,dropout_keep=0.9,activation=tf.nn.elu,lr=0.0001,optimizer='adam'):    
    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,
                 attention_heads=8,attention_size=512,dropout_rate=0.1, activation= 'elu', lr= 0.0001, optimizer= 'adam'):

        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.vocab_size = embedding_matrix.shape[0]
        self.embedding_size = embedding_matrix.shape[1]
        self.ms = max_sents
        self.mw = max_words

        #word hierarchy - embeddings
        single_line = Input(shape=(max_words,))
        single_line_r = Lambda(lambda x: K.slice(x,[0,0],[1,K.cast(K.sum(K.sign(x)),'int32')]))(single_line)
        word_embeds = Embedding(self.vocab_size,self.embedding_size,weights=[self.embedding_matrix])(single_line_r)
        word_embeds = Lambda(lambda x: K.dropout(x,dropout_rate))(word_embeds)

        #word hierarchy - self attention
        A = Conv1D(attention_size,3,activation='elu',padding='same')(word_embeds)
        B = Conv1D(attention_size,3,activation='elu',padding='same')(word_embeds)
        C = Conv1D(attention_size,3,activation='elu',padding='same')(word_embeds)

        AB = Dot(2)([A,B])
        AB = Lambda(lambda x:x/np.sqrt(attention_size))(AB)
        AB = Activation('softmax')(AB)
        AB = Dropout(dropout_rate)(AB)
        AB = Permute([2,1])(AB)
        word_out = Dot(1)([AB,C])

        #word hierarchy - target attention
        A = Conv1D(attention_size,3,activation='elu',padding='same')(word_out)
        AB = target_att(attention_size)(A)
        AB = Activation('softmax')(AB)
        AB = Dropout(dropout_rate)(AB)
        AB = Permute([2,1])(AB)
        line_embed = Dot(1)([AB,word_out])
        line_embed = Lambda(lambda x: K.squeeze(K.squeeze(x,0),0))(line_embed)
        word_hierarchy = Model(inputs=single_line,outputs=line_embed)

        #line hierarchy - embeddings
        single_doc = Input(shape=(max_words,))
        single_doc_r = Lambda(lambda x: K.slice(x,[0,0],[K.cast(K.sum(K.sign(K.sum(K.sign(x),1))),'int32'),max_words]))(single_doc)
        single_doc_r = Lambda(lambda x: K.expand_dims(x,0))(single_doc_r)
        line_embeds = TimeDistributed(word_hierarchy)(single_doc_r)
        line_embeds = Lambda(lambda x: K.dropout(K.expand_dims(K.transpose(x),0),dropout_rate))(line_embeds)

        #line hierarchy - self attention
        A = Conv1D(attention_size,3,activation='elu',padding='same')(line_embeds)
        B = Conv1D(attention_size,3,activation='elu',padding='same')(line_embeds)
        C = Conv1D(attention_size,3,activation='elu',padding='same')(line_embeds)

        AB = Dot(2)([A,B])
        AB = Lambda(lambda x:x/np.sqrt(attention_size))(AB)
        AB = Activation('softmax')(AB)
        AB = Dropout(dropout_rate)(AB)
        AB = Permute([2,1])(AB)
        line_out = Dot(1)([AB,C])

        #line hierarchy - target attention
        A = Conv1D(attention_size,3,activation='elu',padding='same')(line_out)
        AB = target_att(attention_size)(A)
        AB = Activation('softmax')(AB)
        AB = Dropout(dropout_rate)(AB)
        AB = Permute([2,1])(AB)
        doc_embed = Dot(1)([AB,line_out])
        doc_embed = Lambda(lambda x: K.squeeze(K.squeeze(x,0),0))(doc_embed)
        line_hierarchy = Model(inputs=single_doc,outputs=doc_embed)

        #doc embeddings and classify
        doc_inputs = Input(shape=(max_sents,max_words))
        doc_inputs_r = Lambda(lambda x: K.permute_dimensions(x,[1,0,2]))(doc_inputs)
        doc_embeds = TimeDistributed(line_hierarchy)(doc_inputs_r)
        doc_embeds = Lambda(lambda x: K.dropout(K.permute_dimensions(x,[1,0]),dropout_rate))(doc_embeds)
        preds = []
        for c in num_classes:
            preds.append(Dense(c,activation='softmax')(doc_embeds))

        #build model
        #adam = Adam(lr=0.0001,beta_1=0.9,beta_2=0.99)
        if optimizer == 'adam':
            opt = keras.optimizers.Adam( lr = lr, beta_1= 0.9, beta_2= 0.99 )
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD( lr= lr )
        elif optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta( lr= lr )
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSProp( lr= lr )
 
        self.model = Model(inputs=doc_inputs,outputs=preds)
        self.model.compile( loss= 'sparse_categorical_crossentropy' ,optimizer= opt, metrics= [ 'acc' ] )
        self.model.summary()

if __name__ == "__main__":

    import pickle
    from sklearn.model_selection import train_test_split

    #params
    batch_size = 64
    lr = 0.0001
    epochs = 10
    train_samples = 5000
    test_samples = 1000
    vocab_size = 5000
    max_lines = 150
    max_words = 30
    num_classes = [5,10,20,5,2]
    embedding_size = 300

    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    X = np.random.randint(0,vocab_size,(train_samples+test_samples,max_lines,max_words))

    #optional masking
    min_lines = 30
    min_words = 5
    mask = []
    for i in range(train_samples+test_samples):
        doc_mask = np.ones((1,max_lines,max_words))
        num_lines = np.random.randint(min_lines,max_lines)
        for j in range(num_lines):
            num_words = np.random.randint(min_words,max_words)
            doc_mask[0,j,:num_words] = 0
        mask.append(doc_mask)

    mask = np.concatenate(mask,0)
    X[mask.astype(np.bool)] = 0

    #test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_trains = [np.random.randint(0,c,train_samples) for c in num_classes]
    y_tests = [np.random.randint(0,c,test_samples) for c in num_classes]

    #train model
    model = hcan(vocab,num_classes,max_lines,max_words)
    model.model.fit(X_train,y_trains,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_tests))

