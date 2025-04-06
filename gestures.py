"""
    Created on Wed Jun  9 20:00:00 2021
    Author: Joshua David Salcedo Monroy
    Email: a22110109@ceti.mx
    Version: 1.0
    Model: Random Forest
"""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd 
import joblib


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model = joblib.load('gestures.pkl')

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks is not None:
               
        
              for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                  x1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
                  y1=int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
               
                  x2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                  y2=int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
               
                  x3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
                  y3=int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
                  
                  x4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
                  y4=int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)
                  
                  x5=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
                  y5=int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)
                  
                  x6=int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                  y6=int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)

                  handedness = results.multi_handedness[i]
                  hand_label = handedness.classification[0].label
                  
                  angulos=[]
                  arregloangulos=[]
                  
                  cv2.circle(frame,(x1,y1),3,(255,0,0),4)
                  cv2.circle(frame,(x2,y2),3,(0,255,0),4)
                  cv2.circle(frame,(x3,y3),3,(0,0,255),4)
                  cv2.circle(frame,(x4,y4),3,(0,255,255),4)
                  cv2.circle(frame,(x5,y5),3,(0,0,0),4)
                  cv2.circle(frame,(x6,y6),3,(255,0,255),4)
                  cv2.line(frame,[x1,y1],[x2,y2],(255,255,0),thickness = 2)
                  cv2.line(frame,[x2,y2],[x3,y3],(255,255,0),thickness = 2)
                  cv2.line(frame,[x3,y3],[x4,y4],(255,255,0),thickness = 2)
                  cv2.line(frame,[x4,y4],[x5,y5],(255,255,0),thickness = 2)
                  cv2.line(frame,[x5,y5],[x6,y6],(255,255,0),thickness = 2)
                  cv2.line(frame,[x6,y6],[x1,y1],(255,255,0),thickness = 2)
                  
                  v=[x2,y2]
                  a=[x1,y1]
                  m=[x6,y6]
                  n=[x5,y5]
                  am=[x4,y4]
                  r=[x3,y3]
                  
                  v0=np.array(v)-np.array(a)
                  v1=np.array(m)-np.array(a)
                  
                  v2=np.array(a)-np.array(m)
                  v3=np.array(n)-np.array(m)
                  
                  v4=np.array(m)-np.array(n)
                  v5=np.array(am)-np.array(n)
                  
                  v6=np.array(n)-np.array(am)
                  v7=np.array(r)-np.array(am)
                  
                  v8=np.array(am)-np.array(r)
                  v9=np.array(v)-np.array(r)
                  
                  v10=np.array(r)-np.array(v)
                  v11=np.array(a)-np.array(v)
                  
                
                  
                  angle1=np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))#rad
                  grado1= np.degrees(angle1)
                  grado1=round(grado1)
                  cv2.putText(frame,str(grado1),(x1+15,y1),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1)
                  
                  angle2=np.math.atan2(np.linalg.det([v2,v3]),np.dot(v2,v3))#rad
                  grado2= np.degrees(angle2)
                  grado2=round(grado2)
                  cv2.putText(frame,str(grado2),(x6+15,y6),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,255),1)
                  
                  angle3=np.math.atan2(np.linalg.det([v4,v5]),np.dot(v4,v5))#rad
                  grado3= np.degrees(angle3)
                  grado3=round(grado3)
                  cv2.putText(frame,str(grado3),(x5+15,y5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1)
                  
                  angle4=np.math.atan2(np.linalg.det([v6,v7]),np.dot(v6,v7))#rad
                  grado4= np.degrees(angle4)
                  grado4=round(grado4)
                  cv2.putText(frame,str(grado4),(x4+10,y4),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,255),1)
                  
                  angle5=np.math.atan2(np.linalg.det([v8,v9]),np.dot(v8,v9))#rad
                  grado5= np.degrees(angle5)
                  grado5=round(grado5)
                  cv2.putText(frame,str(grado5),(x3+5,y3-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,0,255),1)
                  
                  angle6=np.math.atan2(np.linalg.det([v10,v11]),np.dot(v10,v11))#rad
                  grado6= np.degrees(angle6)
                  grado6=round(grado6)
                  cv2.putText(frame,str(grado6),(x2,y2-10),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
             
                  result = model.predict([[grado1,grado2,grado3,grado4,grado5,grado6]])
                
                  if hand_label == "Right":
                    if result[0] == 0:
                      cv2.putText(frame, "Uno", (400,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                    if result[0] == 1:
                      cv2.putText(frame, "Dos", (400,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                    if result[0] == 2:
                      cv2.putText(frame, "Tres", (400,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                      
  
                  
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
