import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
import pickle





# Setting Knobs
##################

path = 'myData'
testRatio = 0.2
validationRatio = 0.2
imageDim = (28, 28, 3)

batchSizeVal = 50
epochVal = 100
stepsPerEpochVal = 2000

##################

images = []
classNo = []
myList = os.listdir(path)
print(len(myList))
noOfClasses = len(myList)
print("importing classes...")

for x in range(0, noOfClasses):
    myPicList = os.listdir(path + '/' + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + '/' + str(x) + '/' + y)
        curImg = cv2.resize(curImg, (28, 28))
        images.append(curImg)
        classNo.append(x)
    print(x,  end=' ')
print('Total Number Of Images = ', len(images))
print('Total Number of Labels = ', len(classNo))

images = np.array(images)
classNo = np.array(classNo)
print(images.shape)

# Splitting Data

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validationRatio)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train == 0)[0]))
print(numOfSamples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title('Number Of Images For Each Class')
plt.xlabel('class ID')
plt.ylabel('number of Images')
plt.show()


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(x_train[30])
# img = cv2.resize(img,(400,00))
# cv2.imshow("preprocessed", img)
# cv2.waitKey(0)

x_train = np.array(list(map(preProcessing, x_train)))
x_test = np.array(list(map(preProcessing, x_test)))
x_validation = np.array(list(map(preProcessing, x_validation)))


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

#Augmentation

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(x_train)

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDim[0],imageDim[1],1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())


history = model.fit_generator(dataGen.flow(x_train,y_train,
                                         batch_size=batchSizeVal),
                                         steps_per_epoch= stepsPerEpochVal,
                                         epochs=epochVal,
                                         validation_data =(x_validation,y_validation),
                                         shuffle=1)


#Loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')

#Accuracy
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

score = model.evaluate(x_test,y_test, verbose=0)
print('Test score = ', score[0])
print('Test Accuracy = ', score[1])


pickle_out = open('digit_model_trained_10.p', 'wb')
pickle.dump(model, pickle_out)
pickle_out.close()






