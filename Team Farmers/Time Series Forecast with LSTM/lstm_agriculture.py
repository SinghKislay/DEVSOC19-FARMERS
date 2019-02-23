import tensorflow as tf
from input_pipe import data_input_pipeline
import numpy as np

dataset =data_input_pipeline('./tfRecords.train')



def feeder(data,epochs):
    iterator = data.make_initializable_iterator()
    it_init=iterator.initializer
    x_input,label =iterator.get_next()
    
    
    _input1=tf.unstack( x_input,num=4,axis=1)
    labels=tf.unstack(label,num=4,axis=1)
    logits=lstm_blacbox(_input1)

    with tf.name_scope("loss"):
        loss=tf.losses.mean_squared_error(labels,logits,weights=1.0)
        adam_optimizer=tf.train.AdamOptimizer(learning_rate=.02).minimize(loss)
        init_op=tf.global_variables_initializer()
        tf.summary.scalar("loss",loss)
        merged=tf.summary.merge_all()
        saver=tf.train.Saver()
    checkpoint_path="./graph/agro_model"

    with tf.Session() as sess:
        sess.run(init_op)
        writer=tf.summary.FileWriter("./lstm_logs",sess.graph)
        sess.run(it_init)
        
        j=0
        for j in range(epochs):
                summary=sess.run(merged)  
                _,l=sess.run([adam_optimizer,loss])   
                print(" loss: {:.10f}".format( l))
                writer.add_summary(summary,j)
                j=j+1
                    
                
        saver.save(sess, checkpoint_path)





def lstm_blacbox(x):
    
    
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(16,forget_bias=1.0)
    
    outputs,_= tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
    return [tf.layers.dense(outputs[0],1,activation='sigmoid'),tf.layers.dense(outputs[1],1,activation='sigmoid'),tf.layers.dense(outputs[2],1,activation='sigmoid'),tf.layers.dense(outputs[3],1,activation='sigmoid')]

feeder(dataset,180)
