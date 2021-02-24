import cv2


#################
width = 640
height = 480
start = cv2.getTickCount()
#################

cap = cv2.VideoCapture(0)
tracker = cv2.TrackerMOSSE_create()
success, img = cap.read()
bbox = cv2.selectROI('tracking',img,False)
tracker.init(img,bbox)

if not (cap.isOpened()):
    print('could not open selected camera')

cap.set(3,width)
cap.set(3,height)

def drawBox(img,bbox):
    x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img, 'status - tracking', (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

while(True):

    success, img = cap.read()
    success, bbox = tracker.update(img)

    if success:
        drawBox(img,bbox)
    else:
        cv2.putText(img, 'status - lost', (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)


    fps = int((cv2.getTickFrequency()/(cv2.getTickCount()-start))*100)
    cv2.putText(img, ('fps - ' + str(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    cv2.imshow('tracking', img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

