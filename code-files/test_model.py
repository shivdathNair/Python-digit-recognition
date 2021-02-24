import numpy as np
import cv2
import pickle
import time

#################
width = 640
height = 480
noFrames = 60
start = time.time()
threshold = 0.8
#################

cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
    print('could not open selected camera')

cap.set(3,width)
cap.set(3,height)

pickle_in = open('digit_model_trained_10.p','rb')
model = pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while(True):

    success, imgOriginal = cap.read()
    end = time.time()
    seconds = int(end - start)
    fps = int(noFrames / seconds)
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(28,28))
    img = preProcessing(img)
    #cv2.imshow('preview', img)
    img = img.reshape(1,28,28,1)

    #predicttoin
    classInd = int(model.predict_classes(img))
    prediction = model.predict(img)
    probVal = np.amax(prediction)
    print(classInd , '-' , probVal)

    if probVal > threshold:
        cv2.putText(imgOriginal,str(classInd) + ' '+ str(probVal),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
    cv2.imshow('original Image',imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
