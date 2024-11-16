import time
import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
# from utilss import * # Explicit imports

from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# from PIL import Image

def gen_labels():
    train = '../Data/Train'
    train_generator = ImageDataGenerator(rescale = 1/255)

    train_generator = train_generator.flow_from_directory(train,
                                                        target_size = (300,300),
                                                        batch_size = 32,
                                                        class_mode = 'sparse')
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    return labels

def preprocess(image):
    image = np.array(image.resize((300, 300), Image.ANTIALIAS))
    image = np.array(image, dtype='uint8')
    image = np.array(image)/255.0

    return image

def model_arc():
    model=Sequential()

    #Convolution blocks
    model.add(Conv2D(32, kernel_size = (3,3), padding='same',input_shape=(300,300,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=2)) 

    model.add(Conv2D(64, kernel_size = (3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2)) 

    model.add(Conv2D(32, kernel_size = (3,3), padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=2)) 

    #Classification layers
    model.add(Flatten())

    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(6,activation='softmax'))

    return model

# Generate labels
labels = gen_labels()

# Streamlit HTML elements
st.markdown('''
    <div style="padding: 20px;">
        <center><h1>Garbage Segregation</h1></center>
    </div>
    <div>
        <center><h3>Please upload Waste Image to find its Category</h3></center>
    </div>
''', unsafe_allow_html=True)

# Upload option
opt = st.selectbox("How do you want to upload the image for classification?",
                   ('Please Select', 'Upload image via link', 'Upload image from device'))

image = None
if opt == 'Upload image from device':
    file = st.file_uploader('Select', type=['jpg', 'png', 'jpeg'])
    if file:
        image = Image.open(file)

elif opt == 'Upload image via link':
    img_url = st.text_input('Enter the Image Address')
    if st.button('Submit'):
        try:
            image = Image.open(urllib.request.urlopen(img_url))
        except Exception:
            st.error("Invalid image address!")

# Predict category
if image:
    st.image(image, width=300, caption='Uploaded Image')
    if st.button('Predict'):
        img = preprocess(image)
        model = model_arc()
        model.load_weights("../weights/model.h5")
        prediction = model.predict(img[np.newaxis, ...])
        category = labels[np.argmax(prediction[0], axis=-1)]
        st.info(f'The uploaded image is classified as "{category} waste".')



# import time
# import streamlit as st
# import numpy as np
# from PIL import Image
# import urllib.request
# from utils import *

# labels = gen_labels()

# html_temp = '''
#     <div style =  padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
#     <center><h1>Garbage Segregation</h1></center>
    
#     </div>
#     '''

# st.markdown(html_temp, unsafe_allow_html=True)
# html_temp = '''
#     <div>
#     <h2></h2>
#     <center><h3>Please upload Waste Image to find its Category</h3></center>
#     </div>
#     '''
# st.set_option('deprecation.showfileUploaderEncoding', False)
# st.markdown(html_temp, unsafe_allow_html=True)
# opt = st.selectbox("How do you want to upload the image for classification?\n", ('Please Select', 'Upload image via link', 'Upload image from device'))
# if opt == 'Upload image from device':
#     file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
#     st.set_option('deprecation.showfileUploaderEncoding', False)
#     if file is not None:
#         image = Image.open(file)

# elif opt == 'Upload image via link':

#   try:
#     img = st.text_input('Enter the Image Address')
#     image = Image.open(urllib.request.urlopen(img))
    
#   except:
#     if st.button('Submit'):
#       show = st.error("Please Enter a valid Image Address!")
#       time.sleep(4)
#       show.empty()

# try:
#   if image is not None:
#     st.image(image, width = 300, caption = 'Uploaded Image')
#     if st.button('Predict'):
#         img = preprocess(image)

#         model = model_arc()
#         model.load_weights("../weights/model.h5")

#         prediction = model.predict(img[np.newaxis, ...])
#         st.info('Hey! The uploaded image has been classified as " {} waste " '.format(labels[np.argmax(prediction[0], axis=-1)]))
# except Exception as e:
#   st.info(e)
#   pass
