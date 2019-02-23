import tensorflow as tf
import sys
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

output_frozen_graph_name='./frozen_model.pb'



input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name,"rb") as f:

    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(

         input_graph_def, ["input/X"],["output_node/Softmax"],tf.float32.as_datatype_enum

)                               

f = tf.gfile.FastGFile('final_corn.pb',"w")
f.write(output_graph_def.SerializeToString())
