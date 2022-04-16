# fashion-recommender-system
Fashion Recommender system is a web app which helps users find similar images as they want to search . It is similar to reverse image search and works on a dataset of 40k images and finds images matching the given input image .

## Project: fashion-recommender-system

### Install

This project requires **Python** and the following Python libraries installed:

- [annoy==1.17.0](https://algorithmia.com/algorithms/spotify/Annoy/docs)
- [numpy~=1.19.2](https://numpy.org/)
- [image](https://pypi.org/project/image/)
- [tensorflow](https://www.tensorflow.org/)
- [streamlit==1.8.1](https://streamlit.io/)
- [pickle-mixin==1.0.2](https://pypi.org/project/pickle-mixin/)
- [firebase_admin==5.1.0](https://pypi.org/project/firebase-admin/)

You will also need to have software installed to run and execute a [Visula Studio Code](https://code.visualstudio.com/).

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which already has the above packages and more included. 


### Run

In a terminal or command window, navigate to the top-level project directory `fashion-recommender-system/` (that contains this README) and run one of the following commands:

```
streamlit run main.py
```  


This will open the Jupyter Notebook software and project file in your browser.

### Data

[kaggle dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
44000 products with category labels and images.

**INPUT**
Clothing image 

**OUTPUT**
Five images from data which are most similar to input image

### Main Features

Highlights -
- ResNet (features extraction)
- CNN & Transfer Learning
- Annoy (Spotify music recommendation library)
- Streamlit webapp
- Firebase Storage

## Model
model = ResNet50(weights='imagenet' , include_top=False, input_shape = (224,224,3)) <br>
model.trainable = False<br>
model = tf.keras.Sequential([model,  GlobalMaxPool2D()])<br>
model.summary()<br>

## Result

[output](https://drive.google.com/file/d/1BcDSAW3sWzcH-PT7ELGA1qt2AGrZJHxW/view?usp=sharing)


