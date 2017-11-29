import tensorflow as tf

#encoder
def dn_encoder(x, name, filter_size, out_channel, stride=1, do_bias=True, do_activation=True, do_batch_norm=True, inference_variable=None):
    data_format='NHWC' #all the codes are written using this format
    
    in_shape = x.get_shape()
    in_channel = int(in_shape[3])
    
    shape = [filter_size, filter_size, in_channel, out_channel]
    
    W = tf.get_variable(name=name+"_W", shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    
    
    strides = [1, stride, stride, 1]
    op = tf.nn.conv2d(input=x, filter=W, strides=strides, padding='SAME',
                      data_format=data_format, name=name+"_conv")
    
    if do_bias:
        b = tf.get_variable(name=name+"_b", shape=out_channel, 
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        op = tf.nn.bias_add(op,b,data_format=data_format,name=name+'_bias')
    
    
    if do_batch_norm:
        if inference_variable is None:
            mu,sigmasq = tf.nn.moments(op,axes=[0,1,2], name=name+"_moments")
        else:
            pass
        
        gamma = tf.get_variable(name=name+"_gamma", shape=out_channel, 
                                initializer=tf.truncated_normal_initializer(stddev=0.01)) #scale
        beta = tf.get_variable(name=name+"_beta", shape=out_channel, 
                                initializer=tf.truncated_normal_initializer(stddev=0.01)) #offset

        epsilon = 10**-8
        
        
        
        op = (op - mu) / tf.sqrt(sigmasq + epsilon)
        op = gamma * op + beta
    
    if do_activation:
        op = tf.nn.relu(op,name=name+'_activation')
                
    return op

#decoder
def dn_decoder(x, name, filter_size, out_channel, stride=2, do_bias=True, do_activation=True, do_batch_norm=True, inference_variable=None):
    data_format = 'NHWC' #all the codes are written using this format
    
    in_shape = x.get_shape()
    in_channel = in_shape[3]
    
    shape = [filter_size, filter_size, out_channel, in_channel]
    
    W = tf.get_variable(name=name+"_W", shape=shape,
                        initializer=tf.truncated_normal_initializer(stddev=0.001))
    
    strides = [1, stride, stride, 1]
    output_shape = [int(in_shape[0]), int(in_shape[1])*stride, int(in_shape[2])*stride, out_channel]
    op = tf.nn.conv2d_transpose(value=x, filter=W, output_shape=output_shape, strides=strides, padding='SAME',
                      data_format=data_format, name=name+"_conv")
    
    if do_bias:
        b = tf.get_variable(name=name+"_b", shape=out_channel, 
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        op = tf.nn.bias_add(op,b,data_format=data_format,name=name+'_bias')
    
    
    
    if do_batch_norm:
        if inference_variable is None:
            mu,sigmasq = tf.nn.moments(op,axes=[0,1,2], name=name+"_moments")
        else:
            pass
        
        gamma = tf.get_variable(name=name+"_gamma", shape=out_channel, 
                                initializer=tf.truncated_normal_initializer(stddev=0.01)) #scale
        
        beta = tf.get_variable(name=name+"_beta", shape=out_channel, 
                                initializer=tf.truncated_normal_initializer(stddev=0.01)) #offset

        epsilon = 10**-8
        
        op = (op - mu) / tf.sqrt(sigmasq + epsilon)
        op = gamma * op + beta
        
    if do_activation:
        op = tf.nn.relu(op,name=name+'_activation')
            
    return op

