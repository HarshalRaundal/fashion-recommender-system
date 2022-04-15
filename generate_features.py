from tqdm import tqdm
import pickle
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50 , preprocess_input
import numpy as np
import tensorflow as tf


model_path = 'models/1649875344'
model = tf.saved_model.load(model_path)


def extract_features(filename, model):
  img  = image.load_img(filename,target_size = (224,224))
  img_arr = image.img_to_array(img)
  expanded_img = np.expand_dims(img_arr , axis=0)
  preprocessed_input = preprocess_input(expanded_img)
#   result = model.predict(preprocessed_input).flatten()
  result = model(preprocessed_input, training=False).numpy().flatten()
  normalized_res = result/np.linalg.norm(result)
  return normalized_res

folder_location = 'data\myntradataset\images'

def generate_filenames(folder_location):
  filenames = []

  for i , file in enumerate(os.listdir(folder_location)):
    # print('saving file {} \n'.format(file))
    filenames.append(file)
    

  # for i in range(10):
  #   print(filenames[i])
  
  return filenames

def generate_features(folder_location_of_data):
  features = []
  filenames = generate_filenames(folder_location_of_data)
  i =0 
  for file in tqdm(filenames):
    # print('extracting features from {} \n'.format(file))
    features.append(extract_features(os.path.join(folder_location_of_data,file) , model))
   

  # print(len(filenames) , len(features))

  pickle.dump(features , open('sample_embeddings.pkl' ,'wb'))
  pickle.dump(filenames , open('sample_filenames.pkl' ,'wb') )
