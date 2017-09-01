import tensorflow as tf

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import nn

#cell path values: (basic_lstm_cell, gru_cell/gates)
def sequence_layer(train_steps, batch_size, in_length, out_length, num_inputs, \
                   num_layers, hidden_size, cell_maker, cell_path, beta, global_dropout, reuse):

    inputs = []
    regularizers = 0
    logits = 0
    
    initial_states = []
    final_states = []
    
    targets = tf.placeholder(tf.float32, shape=(batch_size, train_steps, out_length), name="targets")
    
    for channel in range(num_inputs):
        input_str = 'input_{0}'.format(channel)
        input_name = '{0}:0'.format(input_str)
        
        channel_input = tf.placeholder(tf.float32, shape=(batch_size, train_steps, in_length), name=input_str)
        inputs.append(channel_input)

        with tf.variable_scope("cell") as scope:
            internal_cells = [cell_maker() for _ in range(num_layers)]
            multicell = tf.contrib.rnn.MultiRNNCell(internal_cells)
            if global_dropout:
                multicell = tf.nn.rnn_cell.DropoutWrapper(multicell, output_keep_prob=global_dropout)
        
        zero_state = multicell.zero_state(batch_size, tf.float32)
        initial_states.append(zero_state)
        
        scope_str = 'channel_{0}'.format(channel)
        
        static = False
        
        with tf.variable_scope(scope_str, reuse=reuse) as scope:
            if static:
                #The static_rnn expects a sequence (list) of batch_size * in_length
                reshaped = tf.reshape(channel_input, [train_steps, batch_size, in_length])
                channel_input = tf.unstack(reshaped, num=train_steps, axis=0)
                output, final_state = tf.nn.static_rnn(\
                                multicell, channel_input, initial_state=zero_state)
                final_states.append(final_state)
                output = tf.reshape(tf.stack(axis=1, values=output), [batch_size, train_steps, hidden_size])
            else:
                #if dynamic_rnn set time major [max_time,batch_size, ...]
                #channel_input = tf.reshape(channel_input, [train_steps, batch_size, in_length])
                output, final_state = tf.nn.dynamic_rnn(\
                                multicell, channel_input, initial_state=zero_state, time_major=False)
                #output = tf.reshape(output, [batch_size, train_steps, hidden_size])
                
            final_states.append(final_state)
            
            weight_str = 'softmax_weight_{0}'.format(channel)
            bias_str = 'softmax_bias_{0}'.format(channel)
            softmax_weight = tf.get_variable(weight_str, [hidden_size, out_length], dtype=tf.float32)
            
            regularizer = tf.nn.l2_loss(softmax_weight)
            regularizers = regularizer + regularizers
            softmax_bias = tf.get_variable(bias_str, [1, out_length], dtype=tf.float32)
            
        with tf.variable_scope(scope_str, reuse=True) as scope:
            #add the cell's kernels to the reqularizer
            for cell_num in range(num_layers):
                kernel_str = 'rnn/multi_rnn_cell/cell_{0}/{1}/kernel'.format(cell_num, cell_path)
                kernel = tf.get_variable(kernel_str)
                regularizer = tf.nn.l2_loss(kernel)
                regularizers = regularizer + regularizers
        
        #a bit of unstacking and restacking to get our logits
        channel_outputs = tf.unstack(output, num=batch_size, axis=0)
        
        channels = []
        
        for b_output in channel_outputs:
            channel_result = tf.matmul(b_output, softmax_weight)
            channel_result = channel_result + softmax_bias
            channels.append(channel_result)
            
        channel_logits = tf.stack(channels)
        
        with tf.variable_scope("effect", reuse=reuse) as scope:
            effect_str = 'effect_{0}'.format(channel)
            effect = tf.get_variable(effect_str, [], dtype=tf.float32)
            tf.assign(effect, 1.0/num_inputs)
            
        channel_logits = channel_logits * effect
        
        logits = channel_logits + logits
        
    loss = tf.losses.mean_squared_error(
        predictions=logits,
        labels=targets
    )
    
    #add in the regularization loss
    cost = loss + beta * regularizers
    
    saver_dict = dict(
        inputs = inputs,
        inital_state = initial_states,
        final_state = final_states,
        cost = cost,
        preds = logits
    )
    
    return loss, cost, logits, saver_dict

#separate out so we can easily add different optimizers to the graph
def adam_train(cost):
    
    lr = tf.Variable(0.005, trainable=False)
    
    new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")

    lr_update = tf.assign(lr, new_lr)

    max_grad_norm = 5

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          max_grad_norm)

    optimizer = tf.train.AdamOptimizer(lr)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    return train_op, grads