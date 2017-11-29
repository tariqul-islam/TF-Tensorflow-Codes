#Using Tesnorflow 0.9

import numpy as np
import tensorflow as tf

class tf_lstm_cell(object):
    '''
    In an LSTM cell shapes of X,and Out put are respectively (N,D) and (NxH)
    4 different weight vectors of shape (DxH) is required for the input
    Usually they are declared in a single matrix of shape (Dx4H)
    Same argument goes for the weight vectors for the previous state with input and output (NxH).
    '''

    def __init__(self,D,H,
                weight_init=False,Wx=None,Wh=None,b=None,
                dtype=tf.float32):
        if weight_init==True:
            if Wx is not None:
                if (D,4*H)==Wx.shape:
                    self.Wx=tf.Variable(Wx,dtype=tf.float32)
                else:
                    raise ValueError("Wx shape is not equal to [D,4*H]")
            else:
                raise ValueError("Wx is None. Expected [D,4*H] dimension matrix")
            
            if Wh is not None:
                if (H,4*H)==Wh.shape:
                    self.Wh=tf.Variable(Wh,dtype=tf.float32)
                else:
                    raise ValueError("Wh shape is not equal to [H,4*H]")
            else:
                raise ValueError("Wh is None. Expected [H,4*H] dimension matrix")
            
            if b is not None:
                if (4*H,)==b.shape:
                    self.b=tf.Variable(b,dtype=tf.float32)
                else:
                    raise ValueError("b shape is not equal to [4H,]")
            else:
                raise ValueError("b is None. Expected [4*H,] dimension matrix")
        else:
            self.Wx = tf.Variable(tf.truncated_normal(
                                        [D,4*H],
                                        stddev=0.1),
                                  dtype=tf.float32)
            self.Wh = tf.Variable(tf.truncated_normal(
                                        [H,4*H],
                                        stddev=0.1),
                                  dtype=tf.float32)
            self.b = tf.Variable(tf.truncated_normal(
                                        [4*H],
                                        stddev=0.1),
                                  dtype=tf.float32)
        

#layer description of
#the regular lstm operation
def lstm_op(lstm,x,prev_h,prev_c):
    """
        lstm = weigth variable of lstm
        x = input to lstm
        prev_h = previous hidden state
        prev_c = previous cell state
    """
    D,H = lstm.Wx.get_shape()
    H = int(H/4)
    a = tf.matmul(x,lstm.Wx) + tf.matmul(prev_h,lstm.Wh) + lstm.b
    ax = tf.sigmoid(a[:,:3*H])
    next_c = ax[:,H:2*H] * prev_c + ax[:,:H] * tf.tanh(a[:,3*H:])
    next_h = ax[:,2*H:3*H] * tf.tanh(next_c)
    return next_h, next_c

#LSTM with ReLU Actiavtion, It Does Not work    
def lstm_op_relu(lstm,x,prev_h,prev_c):
    """
        lstm = weigth variable of lstm
        x = input to lstm
        prev_h = previous hidden state
        prev_c = previous cell state
    """
    D,H = lstm.Wx.get_shape()
    H = int(H/4)
    a = tf.matmul(x,lstm.Wx) + tf.matmul(prev_h,lstm.Wh) + lstm.b
    ax = tf.sigmoid(a[:,:3*H])
    next_c = ax[:,H:2*H] * prev_c + ax[:,:H] * tf.tanh(a[:,3*H:])
    next_h = tf.nn.relu(ax[:,2*H:3*H] * tf.tanh(next_c))
    return next_h, next_c

#LSTM Layer    
def lstm_layer(lstm,x,h0,layer_description=lstm_op):
    _,T,_ = x.get_shape()
    next_h = h0
    next_c = tf.zeros_like(h0)
    hidden_states = []
    for i in range(T):
        next_h,c = layer_description(lstm,x[:,i,:],next_h,next_c)
        hidden_states.append(next_h)
    hidden_states = tf.pack(hidden_states,axis=1)
    return hidden_states

#LSTM Network    
def lstm_layers(lstms,x,h0s,layer_description=lstm_op):
    """
        lstms = array of lstm weight variable
        x = input
        h0s = array of hidden states
    """
    next_h = x
    for i in range(len(lstms)):
        next_h = lstm_layer(lstms[i],next_h,h0s[i],layer_description)
    return next_h
    
