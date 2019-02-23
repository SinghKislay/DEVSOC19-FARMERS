import tensorflow as tf
import numpy as np
import os

sample=np.array([[[1273859294/20000000,63,165.53],[1283858686/20000000,64,162.53],[1293852390/20000000,65,175.53],[1293855456/20000000,68,165.53]]])

with tf.Session() as sess:
    new_saver=tf.train.import_meta_graph('C:/Users\Kislay/Desktop/Hackathon_2018/graph/agro_model.meta')
    new_saver.restore(sess,tf.train.latest_checkpoint('./graph'))
    graph = tf.get_default_graph()
    dataset_init_op = graph.get_operation_by_name('dataset_init')
    x_input=graph.get_tensor_by_name("p1:0")
    logit=graph.get_tensor_by_name("l:0")
    sess.run(dataset_init_op,feed_dict={x_input:sample})
    log=sess.run(logit)
    

l=list(log)
r=0
for i in range(4):
    r=r+1
    print("lstm{}: {:.2f} consumption per metric tons".format(r,l[i][0][0]*100000))
