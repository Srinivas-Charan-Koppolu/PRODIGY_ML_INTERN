# GOOGLE COLAB - PYTHON

# Downloading Raw DataSet
# !wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip


# Unzipping Data
# !unzip kagglecatsanddogs_5340.zip


# Extracting UseFull DataSet From Images
# Using Pickles to store the Data Object 
# This helps to reduce the Model Training Time
"""
from os import chdir, getcwd, listdir
import cv2
import numpy as np
import pickle

path = "/content/PetImages"
categories = ['Cat', 'Dog']

data = []
for category in categories:
  category_path = path + "/" + category
  chdir(category_path)
  index = categories.index(category)

  for img in listdir():
    imgpath = category_path + "/" + img
    try:
      pet = cv2.resize(cv2.imread(imgpath,0),(50,50))
      image = np.array(pet).flatten()
      data.append([image,index])
    except:
      pass

print(data)
chdir(path)

f = open("CatDogData.pickle","wb")
pickle.dump(data,f)
f.close()
"""



# -- -- -- -- -- M O D E L -- -- -- -- M A K I N G -- -- -- -- -- #
# Dogs VS Cats - Supervised Learning Model - Support Vector Machine

import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

f = open("/content/PetImages/CatDogData.pickle","rb")
data = pickle.load(f)
f.close() 

print(len(data))

features = []
labels = []

for feature, label in data:
  features.append(feature)
  labels.append(label)


x_train, x_test, y_train, y_test = train_test_split( features, labels, train_size=0.8 )

model = SVC(C=1,kernel='poly',gamma='auto')

model.fit(x_train, y_train)

predict = model.predict(x_test)

accuracy = model.score(x_test, y_test)
print("Accuracy: ",accuracy)
print("Prediction: ",predict)

print("\n\n")
categories = ['Cat', 'Dog']
for index in range(len(x_test)//2):
  # Printing Some Data To Check Out Accuracy
  category = categories[ predict[index] ]
  pet = x_test[index].reshape(50,-1)
  f = plt.figure()
  # f.set_figwidth(4)
  f.set_figheight(2)
  plt.imshow(pet,cmap='gray')
  # plt.xlabel( (category) )
  print(category)
  plt.show()
  print("\n\n")




