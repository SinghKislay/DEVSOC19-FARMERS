import tensorflow as tf
import numpy as np
from tqdm import tqdm
from random import shuffle
import cv2
import os

Test_Dir = 'Test'

interpreter = tf.contrib.lite.Interpreter("corn_graph_shape.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)


def label_image(img):

    word_label = img.split('.')[0]
    label = word_label.split(' ')[0]
    
    if label == 'Disease_1':
        return [1,0,0]
    elif label=='Disease_2':
        return [0,1,0]
    elif label=='Healthy':
        return [0,0,1]

for img in tqdm(os.listdir(Test_Dir)):
    label = label_image(img)
    path = os.path.join(Test_Dir,img)
    img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(210,210))
    input_data_array = [np.array(img),np.array(label)]
    input_data = np.array(input_data_array[0],dtype=np.float32).reshape(-1,210,210,1)
    
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("Actual Label : ",label)
    print("Predicted label :", output_data)
    











#input_shape = input_details[0]['shape']

#for data in test:
#    input_data = data[0]
#input_data = np.array(np.random.random_sample(input_shape),dtype=np.float32)
#interpreter.set_tensor(input_details[0]['index'],input_data)

#interpreter.invoke()
#output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data)
