
import numpy as np
import tensorflow as tf
from tflstm import *
from tflstm import lstm_op as lstm

class lstm_net(object):
    
    def __init__(self,T, D, H, vocab_size, num_classes, h0s,l2_reg_lambda=0.00001):
    
        #T = seuqence length, length of each of the text input
        #D = embedding size
        #H = an array containing the hidden sizes
        #regularization loss will be added later
        
        #placeholders for the input and output
        self.x_input = tf.placeholder(tf.int32, [None, T])
        self.y_input = tf.placeholder(tf.float32, [None, num_classes])
        self.l_input = tf.placeholder(tf.float32, [None])
        
        self.Weights = {}
        reg_loss = 0
        #embedding layer
        with tf.device('/cpu:0'):
            W_embed = tf.Variable(
                tf.random_uniform([vocab_size, D], -1.0, 1.0)
                )
            emx = tf.nn.embedding_lookup(W_embed, self.x_input)
            self.Weights['W_embed'] = W_embed
            reg_loss += tf.nn.l2_loss(W_embed)
            
        #LSTM Layers
        N_layers = len(H)
        #Note:
            #hidden states are zero initialized,
            #could be explored further
            #should be an external input with placeholders.
        #h0s = {}
        D0 = D
        lstms = {}
        for i in range(N_layers):
            #h0s[i] = tf.zeros([None,H[i]], dtype=tf.float32)
            lstms[i] = tf_lstm_cell(D0,H[i])
            D0 = H[i]
            reg_loss += tf.nn.l2_loss(lstms[i].Wx)
            reg_loss += tf.nn.l2_loss(lstms[i].Wh)
            reg_loss += tf.nn.l2_loss(lstms[i].b)
            
        self.Weights['lstms'] = lstms
        
        lstm_output = tf.nn.relu(lstm_layers(lstms,emx,h0s,layer_description=lstm))
        self.lstm_output = tf.reduce_mean(lstm_output,2)
        
        #output layer
        W_affine = tf.Variable(tf.truncated_normal(
                                        [T,num_classes],
                                        stddev=0.1),
                                  dtype=tf.float32)
        
        b_affine = tf.Variable(tf.truncated_normal(
                                        [num_classes],
                                        stddev=0.1),
                                  dtype=tf.float32)
        
        self.Weights['Waffine'] = W_affine
        self.Weights['baffine'] = b_affine
        reg_loss += tf.nn.l2_loss(W_affine)
        reg_loss += tf.nn.l2_loss(b_affine)
        
        self.scores = tf.matmul(self.lstm_output,W_affine)+b_affine
        self.predictions = tf.argmax(self.scores, 1)
        
        losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.y_input)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * reg_loss
        
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.y_input, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
