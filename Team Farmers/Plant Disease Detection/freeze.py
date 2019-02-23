import tensorflow as tf

with tf.Session() as session:
     my_saver = tf.train.import_meta_graph('corn.tflearn.meta')
     my_saver.restore(session,tf.train.latest_checkpoint('.'))

     [print(n.name) for n in session.graph_def.node]
     frozen_graph =tf.graph_util.convert_variables_to_constants(session,
                   session.graph_def,
                   ['output_node/Softmax'])
     with open('frozen_model_shape.pb', 'wb') as f:
          f.write(frozen_graph.SerializeToString())

