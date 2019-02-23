import tensorflow as tf
import numpy as np
import os


def parse(serialized):
    features={
        'gdp1':tf.FixedLenFeature([],tf.float32),
        'exchange1':tf.FixedLenFeature([],tf.float32),
        'rainfall1':tf.FixedLenFeature([],tf.float32),
        'label1':tf.FixedLenFeature([],tf.float32),
        'gdp2':tf.FixedLenFeature([],tf.float32),
        'exchange2':tf.FixedLenFeature([],tf.float32),
        'rainfall2':tf.FixedLenFeature([],tf.float32),
        'label2':tf.FixedLenFeature([],tf.float32),
        'gdp3':tf.FixedLenFeature([],tf.float32),
        'exchange3':tf.FixedLenFeature([],tf.float32),
        'rainfall3':tf.FixedLenFeature([],tf.float32),
        'label3':tf.FixedLenFeature([],tf.float32),
        'gdp4':tf.FixedLenFeature([],tf.float32),
        'exchange4':tf.FixedLenFeature([],tf.float32),
        'rainfall4':tf.FixedLenFeature([],tf.float32),
        'label4':tf.FixedLenFeature([],tf.float32),
    }
    parsed_example = tf.parse_single_example(serialized=serialized,features=features)

    gdp1=parsed_example['gdp1']
    exchange1=parsed_example['exchange1']
    rainfall1=parsed_example['rainfall1']
    label1=parsed_example['label1']/100000
    gdp2=parsed_example['gdp2']
    exchange2=parsed_example['exchange2']
    rainfall2=parsed_example['rainfall2']
    label2=parsed_example['label2']/100000
    gdp3=parsed_example['gdp3']
    exchange3=parsed_example['exchange3']
    rainfall3=parsed_example['rainfall3']
    label3=parsed_example['label3']/100000
    gdp4=parsed_example['gdp4']
    exchange4=parsed_example['exchange4']
    rainfall4=parsed_example['rainfall4']
    label4=parsed_example['label4']/100000

    x_input=tf.stack([[gdp1,exchange1,rainfall1],[gdp2,exchange2,rainfall2],[gdp3,exchange3,rainfall3],[gdp4,exchange4,rainfall4]])
    label=tf.stack([[label1],[label2],[label3],[label4]])
    
    return x_input,label


def data_input_pipeline(filenames,batch_size=5):
    dataset=tf.data.TFRecordDataset(filenames=filenames)
    dataset=dataset.shuffle(40)
    dataset=dataset.map(parse,num_parallel_calls=4)
    dataset=dataset.repeat(400)
    dataset=dataset.batch(batch_size)
    dataset=dataset.prefetch(2)
    

    return dataset