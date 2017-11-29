import numpy as np
import tensorflow as tf

from operations import dn_encoder, dn_decoder

#dn_encoder(x, name, filter_size, out_channel, stride=1, do_bias=True, do_activation=True, do_batch_norm=True, inference_variable=None)
#dn_decoder(x, name, filter_size, out_channel, stride=2, do_bias=True, do_activation=True, do_batch_norm=True, inference_variable=None)

class encoder_decoder_net():
    #this class defines a symmetric encoder decoder network without skip connection
    def __init__(self, batch_size=2, filters_size=3, encoder_channels=[64,128,256,512,512,512,512,512], encoder_strides=[2,2,2,2,1,1,1,1], reuse=False, reg=0.0):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        
        self.input_x = tf.placeholder(tf.float32,[batch_size,128,128,1])
        self.input_y = tf.placeholder(tf.float32,[batch_size,128,128,1])
        
        filter_size = 3;
        
        self.x = {} 
        
        
        self.layer_names = []
        
        l_name='inp'
        self.x[l_name] = self.input_x
        self.layer_names.append(l_name)
        
        l_name='proc'
        self.x[l_name] = 2*(self.x[self.layer_names[-1]]-1)
        
        in_shape = self.input_x.get_shape()
        in_channel = int(in_shape[3])
        
        for i in range(len(encoder_channels)):
            l_name = 'g_encoder_'+str(i)
            self.x[l_name] = dn_encoder(x=self.x[self.layer_names[-1]],
                                        name=l_name, 
                                        filter_size=filter_size, 
                                        out_channel=encoder_channels[i], 
                                        stride=encoder_strides[i])
            self.layer_names.append(l_name)
                           
                           
        
        for i in range(len(encoder_channels)-1,-1,-1):
            l_name = 'g_decoder_'+str(i)
            self.x[l_name] = dn_decoder(x=self.x[self.layer_names[-1]],
                                        name=l_name,
                                        filter_size=filter_size, 
                                        out_channel=encoder_channels[i], 
                                        stride=encoder_strides[i])
            self.layer_names.append(l_name)
                           
        
        l_name='g_final_conv'
        self.x[l_name] = dn_encoder(x=self.x[self.layer_names[-1]],
                      name=l_name,
                      filter_size=filter_size,
                      out_channel=in_channel,
                      stride=1,
                      do_activation=False, 
                      do_batch_norm=False)
        self.layer_names.append(l_name)
        
        l_name='g_final_output'             
        self.x[l_name] = 0.5*(tf.nn.tanh(self.x[self.layer_names[-1]])+1)
        self.layer_names.append(l_name)
        
        #output of the network
        self.out = self.x[self.layer_names[-1]]
        
        #L2 Loss function of the network
        #loss = tf.nn.l2_loss(self.input_y-self.out)
        loss = tf.reduce_sum(tf.abs(self.input_y-self.out))
        self.loss = loss+reg*l2_loss
        
        
class encoder_decoder_skipnet():
    #this class defines a symmetric encoder decoder network with skip connection
    def __init__(self, batch_size=1, filters_size=3, encoder_channels=[64,128,256,512,512,512,512,512], encoder_strides=[2,2,2,2,1,1,1,1], reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        
        self.input_x = tf.placeholder(tf.float32,[batch_size,128,128,1])
        self.input_y = tf.placeholder(tf.float32,[batch_size,128,128,1])
        
        filter_size = 3;
        
        self.x = {} 
        
        
        self.layer_names = []
        
        l_name='inp'
        self.x[l_name] = self.input_x
        self.layer_names.append(l_name)
        
        l_name='proc'
        self.x[l_name] = 2*(self.x[self.layer_names[-1]]-1)
        
        in_shape = self.input_x.get_shape()
        in_channel = int(in_shape[3])
        
        for i in range(len(encoder_channels)):
            l_name = 'g_encoder_'+str(i)
            self.x[l_name] = dn_encoder(x=self.x[self.layer_names[-1]],
                                        name=l_name, 
                                        filter_size=filter_size, 
                                        out_channel=encoder_channels[i], 
                                        stride=encoder_strides[i])
            self.layer_names.append(l_name)
            
            print self.x[l_name].get_shape()
                           
        l_name = 'g_flat'
        self.x[l_name] = dn_encoder(x=self.x[self.layer_names[-1]],
                                        name=l_name, 
                                        filter_size=filter_size, 
                                        out_channel=encoder_channels[-1], 
                                        stride=1)               
        self.layer_names.append(l_name)
        print self.x[l_name].get_shape()
        
        for i in range(len(encoder_channels)-1,-1,-1):
            l_name = 'g_decoder_'+str(i)
            l2_name = 'g_encoder_'+str(i)
            
            
            op = tf.concat([self.x[self.layer_names[-1]], self.x[l2_name]], 3)
            
            self.x[l_name] = dn_decoder(x=op,
                                        name=l_name,
                                        filter_size=filter_size, 
                                        out_channel=encoder_channels[i], 
                                        stride=encoder_strides[i])
            
            self.layer_names.append(l_name)
            print 'op', op.get_shape()
            print self.x[l_name].get_shape()
            
            
                           
        
        l_name='g_final_conv'
        self.x[l_name] = dn_encoder(x=self.x[self.layer_names[-1]],
                      name=l_name,
                      filter_size=filter_size,
                      out_channel=in_channel,
                      stride=1,
                      do_activation=False, 
                      do_batch_norm=False)
        self.layer_names.append(l_name)
        
        l_name='g_final_output'             
        self.x[l_name] = 0.5*(tf.nn.tanh(self.x[self.layer_names[-1]])+1)
        self.layer_names.append(l_name)
        
        #output of the network
        self.out = self.x[self.layer_names[-1]]
        
        #L2 Loss function of the network
        loss = tf.nn.l2_loss(self.input_y-self.out)
        #loss = tf.reduce_sum(tf.abs(self.input_y-self.out))
        self.loss = loss#+reg*l2_loss
        

