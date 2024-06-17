from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import argparse
import warnings
import time

import tensorflow_datasets as tfds
import logging
import json
import os


batch_size = 32
image_size = 224


class_names = {}

def arg_parser():
    
    """ 
    This function sets up and returns command-line argument parsing for a scriptand allows the user to specify the model path, input image path, the number of top predictions to return, and the path to a JSON file mapping labels to names
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./image_classifier.h5', action="store", type = str, help='path to the classifier model')
    parser.add_argument('--input',dest='imagepath', default='./test_images/cautleya_spicata.jpg', action="store", type = str, help='path to the image to be classified')
    parser.add_argument('--top_k', dest='top_k', default='5', action="store", type=int, help='Return the top K most likely classes')
    parser.add_argument('--category_names', dest='category_names',default='./label_map.json', action="store", type=str, help='Path to the JSON file with mapping of labels to flower names')
    return parser.parse_args()


def process_image(image):
    
    """ 
    This function resizes an image to 224x224 pixels, normalizes its pixel values to the range [0, 1], and converts it to a NumPy array
    """
        
    image_size = 224
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()


def predict(image_path, model, top_k):
    if top_k <= 0:
        print('Invalid value, top_k must be >0')
        exit()

    image = Image.open(image_path)
    image = np.asarray(image)
    processed_img = process_image(image)
    expanded_img = np.expand_dims(processed_img, axis=0)
    
    predicted_probabilities = model.predict(expanded_image)
    probs = np.sort(predicted_probabilities)[-top_k:len(predicted_probabilities)]
    probs_list = probs.tolist()
    classes = np.argpartition(predicted_probabilities, -top_k)[-top_k:]
    classes = classes.tolist() #create a list of int classes
    names = [class_names.get(str(i + 1)).capitalize() for i in (classes)]
    return probabilities, names

def main():
    args = arg_parser()
    
    model = tf.keras.models.load_model(args.model ,custom_objects={'KerasLayer':hub.KerasLayer})
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    image_path = args.imagepath
    top_k = int(args.top_k)
    probs, classes = predict(image_path, model,top_k)
    class_labels = [class_names[str(ind)] for ind in classes]
    
    print('File: ' + img_path)
    
    print (f'\n Top {top_k} Classes \n')

    for i, probs, class_label in zip(range(1, top_k+1), probs, class_labels):
        print(i)
        print('Label:', class_label)
        print('Class name:', class_names[str(class_label+1)].title())
        print('Probability:', probs)
        print('----------')

if __name__ == "__main__":
    main()
