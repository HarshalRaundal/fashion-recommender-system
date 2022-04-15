import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50 , preprocess_input
import pickle
from annoy_file import find_neighbours
from connect_database import get_images_from_storage 


features_file = 'sample_embeddings.pkl'
images_names_file = 'sample_filenames.pkl'
data_location  ='data\sample_data'

@st.cache  # 👈 This function will be cached
def load_files():
    model_path = 'models/1649875344'
    # load model
    model = tf.saved_model.load(model_path)
    # load filenames
   
    features_list = pickle.load(open(features_file,'rb'))
    filenames_list = pickle.load(open(images_names_file,'rb'))

    # initialize_app()
    return model , features_list , filenames_list 
    

@st.cache
def extract_features(filename, model):
  img  = image.load_img(filename,target_size = (224,224))
  img_arr = image.img_to_array(img)
  expanded_img = np.expand_dims(img_arr , axis=0)
  preprocessed_input = preprocess_input(expanded_img)
  result = model(preprocessed_input, training=False).numpy().flatten()
  normalized_res = result/np.linalg.norm(result)
  return normalized_res

@st.cache
def find_nearest_neighbours(img_arr , features):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(features)
    distances, indices = neighbors.kneighbors([img_arr])
    return indices


def process_input(filename ,model ,features_list):
    extracted_features = extract_features(filename , model)

    indices = find_neighbours(extracted_features ,features_list)
    
    return indices 


def upload_handler (uploaded_file):
    try:
        with open(os.path.join('uploads' , uploaded_file.name) ,'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


st.title('Fashion Recommendor system')

bar =st.progress(0)
i = 0

model , features_list , filenames_list = load_files()

bar.progress(10)

uploaded_file = st.file_uploader("Choose an image ")

bar.progress(20)
if uploaded_file is not None:
    if upload_handler(uploaded_file):
        display_image = Image.open(uploaded_file)
        st.image(display_image)
       
        #empty space
        st.markdown('#')
        st.write('## Recommended Products ')
        st.markdown('#')
        
        bar.progress(30)
        # model work
        indices = process_input(os.path.join('uploads' , uploaded_file.name) , model , features_list)
        # print(indices)

        bar.progress(50)
        # get top 5 images
        col1 ,col2 ,col3 ,col4 , col5 = st.columns(5)
    
        # retrive images from firebase using filenames

        filenames = [filenames_list[indices[i]] for i in range(6)]

        # for file in filenames:
        #     print(file)     

        img_links = get_images_from_storage(filenames)
        
        try:
            with col1:
                # st.image(filenames_list[indices[1]])
                st.image(img_links[0])
                bar.progress(60)
            with col2:
                st.image(img_links[1])
                # st.image(filenames_list[indices[2]])
                bar.progress(70)
            with col3:
                st.image(img_links[2])
                # st.image(filenames_list[indices[3]])
                bar.progress(80)
            with col4:
                st.image(img_links[3])
                # st.image(filenames_list[indices[4]])
                bar.progress(90)
            with col5:
                st.image(img_links[4])
                # st.image(filenames_list[indices[5]])
        except:
            st.write('Files not found')
        bar.progress(95)
    else:
        st.header('some error occured in file ')
    
    
bar.progress(100)
