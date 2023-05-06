import cv2
import numpy as np
import face_recognition
import os
import pickle

savedClassNames=savedEncodeList=[]
f = open('encodes.bin','rb')
savedEncodeList = [pickle.load(f)]
savedClassNames = [pickle.load(f)]
print(savedEncodeList)
print(savedClassNames)


path = 'ImageDir'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    if cl not in savedClassNames:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete!')

if bool(savedClassNames) and bool(savedEncodeList):
    print("equal")
    savedClassNames = classNames
    savedEncodeList = encodeListKnown
else:
    print("extend")
    savedClassNames.extend(classNames)
    savedEncodeList.extend(encodeListKnown)
    
print(savedEncodeList)
print(savedClassNames)

file = open("encodes.bin","wb")
pickle.dump(savedEncodeList,file)
pickle.dump(savedClassNames,file)
