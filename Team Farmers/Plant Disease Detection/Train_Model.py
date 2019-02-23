import tensorflow as tf
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import cv2
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression

Train_Dir = 'Train_Corn'
IMG_SIZE = 210
LR = 0.01

MODEL_NAME = 'corn.tflearn'

def label_image(img):

    word_label = img.split('.')[0]
    label = word_label.split(' ')[0]
    
    if label == 'Disease_1':
        return [1,0,0]
    elif label=='Disease_2':
        return [0,1,0]
    elif label=='Healthy':
        return [0,0,1]

def create_train():

    for img in tqdm(os.listdir(Train_Dir)):
        training_data = []
        label = label_image(img)
        path = os.path.join(Train_Dir,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data 

train_data = create_train()
print(train_data)

tf.reset_default_graph()

convnet = input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')

convnet = conv_2d(convnet,32,3,activation='relu')
convnet = max_pool_2d(convnet,2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)


convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')


convnet = fully_connected(convnet, 3, activation='softmax',name="output_node")
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir ='log')
train = train_data[:-2000]
test  = train_data[-2000:]

X = np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train_data]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input':X},{'targets':Y},n_epoch=20,validation_set=({'input':test_x},{'targets':test_y}),snapshot_step=2,show_metric=True)


with convnet.graph.as_default():
     del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]

model.save(MODEL_NAME)

#
#saver = model.load('./graph.chkp')
#graph = tf.get_default_graph()
#input_graph_def = graph.as_graph_def()
#sess = tf.Session()
#saver.restore(sess,'graph.chkp')

#output_node_names="targets"
#output_graph_def = graph_util.convert_variables_to_constants(
 #                  sess,
  #                 input_graph_def,
   #                output_node_names
#)

#output_graph = "/corn.pb"
#with tf.gfile.GFile(output_graph,"wb") as f:
 #     f.write(output_graph_def.SerializeToString())
#sess.close()
  






