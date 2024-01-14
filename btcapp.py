import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import cv2
import os


st.set_page_config(layout="wide")
st.title('Brain Tumor Classification')
st.write('This app is designed to classify brain tumors from MRI images.')

cmodel = st.selectbox('Select Model', ['CNN', 'SVM', 'Random Forest'])


def modelLoad(model_name):
    try:
        if(model_name=='CNN'):
            return joblib.load('models/rf_model.joblib')
        elif(model_name=='SVM'):
            return joblib.load('models/rf_model.joblib')
        else:
            return joblib.load('models/rf_model.joblib')
    except:
        st.error('Model not found.')
        exit()

def predictImg(cmodel,model,img):
    if cmodel == 'CNN':
        img = cv2.resize(img, (200, 200))
        img = img.reshape(1, 200, 200, 1) / 255.0
        p = model.predict(img)
        return p[0][0]
    else:
        img = cv2.resize(img, (200, 200))
        img = img.reshape(1,-1)/255
        p = model.predict(img)
        return p[0]



model = modelLoad(cmodel)

st.write('Using {} Model'.format(cmodel))
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    c1,c2,c3,c4,c5 = st.columns(5)
    with c3:
        st.image(uploaded_file, caption='Uploaded MRI.')
    
    st.markdown('<p align="center">Classifying using {}...</p>'.format(cmodel), unsafe_allow_html=True)
    st.markdown('<p align="center"><u>{}</u></p>'.format(uploaded_file.name), unsafe_allow_html=True)
    
    file_bytes = uploaded_file.read()
    np_image = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_image, 0)
    
    prediction = predictImg(cmodel,model,img)

    class_label = 'Positive Tumor' if prediction > 0.5 else 'No Tumor'
    
    st.markdown('<p align="center">This MRI scan is of a {}.</p>'.format(class_label), unsafe_allow_html=True)


