import pickle
import firebase_admin
import datetime
# import tensorflow as tf
from firebase_admin import credentials ,storage
import os
import numpy as np
from tqdm import tqdm

def initialize_app():
    if not firebase_admin._apps:
        cred = credentials.Certificate("key.json")
        app = firebase_admin.initialize_app(cred ,{ 'storageBucket':'ml-fashion-recommender.appspot.com'})
        

local_path= 'data\myntradataset\images'
folder_name = 'images'

def add_Data_to_storage(folder_name ,local_path):
    initialize_app()
    bucket = storage.bucket()
    for file in tqdm(os.listdir(local_path)):
        # print('file' ,file)
        image_path = os.path.join(local_path , file)
        # print("image_path: " , image_path)
        image_blob = bucket.blob(f'{folder_name}/{file}')
        image_blob.upload_from_filename(image_path)
        


# add_Data_to_storage('sample_images' , 'data\sample_data')

def get_images_from_storage(filenames):
    try:
        initialize_app()
        img_links = []
        for file in filenames:
            # print(f'trying to fetch {file} ')
            bucket = storage.bucket()
            folder_name_in_storage ='sample_images'
            blob = bucket.get_blob(f'{folder_name_in_storage}/{file}')
            if blob != NONE:
                img_arr=  np.frombuffer(blob.download_as_string() , np.uint8)
                img_links.append(blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET'))
            else:
                img_links.append('')

        return img_links
    except:
         raise Exception('Error retriving images from database')

