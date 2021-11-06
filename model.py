import os
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
import pickle

target_arr = []
flat_data_arr = []

DATADIR = 'C:/Users/HP/Downloads/The IQ-OTHNCCD lung cancer dataset'
CATEGORIES = ['Bengin cases','Malignant cases','Normal cases']
for category in CATEGORIES:
    class_num = CATEGORIES.index(category) # label Encoding the values
    # create a path merging  all the images
    path = os.path.join(DATADIR, category) 
    print(path)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path,img))
        img_resize = resize(img_array, (150, 150, 3)) # NOrmalizes the value from 0 to 2
        flat_data_arr.append(img_resize.flatten())
        target_arr.append(CATEGORIES.index(category))
        
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)


#unique data distribution
unique, count = np.unique(target, return_counts=True)
plt.bar(CATEGORIES,count)
plt.show()

df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data

# split data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 150, stratify=y)


model=DecisionTreeClassifier(random_state=50)

#model training
model.fit(x_train,y_train)

#model testing
y_predicted=model.predict(x_test)

#np.argmax(y_prediction)

pickle.dump(model,open('img_model.p','wb'))
print("Pickle is dumped successfully")


       
