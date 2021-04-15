
import cv2 as cv
import matplotlib.pyplot as plt


# Loading the model 

configFile = r'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozenModel = r'frozen_inference_graph.pb'
model = cv.dnn_DetectionModel(frozenModel,configFile)

# Creating the class file

classFile = r'coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames, len(classNames)) #classNames ==> Contains all the 80 classes from the coco dataset

#  Setting up the configuration of the model

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


#  Capturing the Video

cap = cv.VideoCapture(r'Video_path')
#cap = cv.VideoCapture(0) ==> webcam
if not cap.isOpened():
    raise IOError("Cannot Open Video")


#Setting the font scale and font style

font_scale = 1.1
font = cv.FONT_HERSHEY_COMPLEX
# Reading each frame and detecting the objects in it

while True:
    ret,frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame,confThreshold=0.62)
    if len(ClassIndex)!= 0:
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            if ClassInd <= 80:
                cv.rectangle(frame,boxes,(255,0,0),2)
                cv.putText(frame,classNames[ClassInd-1],(boxes[0]+10,boxes[1]+40),font, fontScale=font_scale, color = (0,255,0),thickness = 3)
    cv.imshow("output",frame)
    if cv.waitKey(2) & 0xFF == ord('q'):
        break  #breaks out of the loop as soon as we press 'q' from our keyboard
cap.release()
cv.destroyAllWindows()


# # For an image
'''

img = cv.imread(r'Image_path')
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(rgb);


ClassIndex, confidence, bbox = model.detect(rgb,confThreshold=0.5)

font_scale = 1.1
font = cv.FONT_HERSHEY_COMPLEX
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv.rectangle(img,boxes,(255,0,0),2)
    cv.putText(img,classNames[ClassInd-1],(boxes[0]+10,boxes[1]+40),font, fontScale=font_scale, color = (0,255,0),thickness = 3)
cv.imshow('output',img);
cv.waitKey(0)

'''