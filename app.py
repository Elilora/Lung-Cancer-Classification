import os
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, plot_confusion_matrix, classification_report, confusion_matrix, accuracy_score


st.title("Lung Cancer Detection using Image Classification with Machine Learning")
st.text("Upload a Lung CT Scan for image classification as benign, Malignant or Normal ")

Categories = ['Bengin cases','Malignant cases','Normal cases']
for category in Categories:
    class_num = Categories.index(category)
model = pickle.load(open('img_model.p','rb'))

uploaded_file = st.file_uploader("Choose a Lung CT scan", type=["jpg", "png", "jpeg"])

def detection(image, model):
    image = np.array(image)
    img_resize=resize(image,(150,150,3))
    l=[img_resize.flatten()]
    df=pd.DataFrame(l) #dataframe
    x=df.iloc[:,:] #input data 
    probability=model.predict(x)
    #for ind,val in enumerate(Categories):
        #print(f'{val} = {probability[0][ind]*100}%')
        #print("The predicted image is : "+Categories[model.predict(l)[0]])
        #j= Categories[model.predict(l)[0]]
        #print(f'Is the image a {Categories[model.predict(l)[0]]} ?(y/n)')
    return probability


if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded CT Scan.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = detection(image, model)
        if label == 0:
            st.write("The CT scan is a benign case")
        elif label == 1:
            st.write("The CT scan is a Malignant case")
        else:
            st.write("The CT scan is a Normal case")