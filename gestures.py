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
cap = cv2.VideoCapture(0)

model = joblib.load('gestures.pkl')

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    while True:
      ret, frame = cap.read()
      if ret == False:
        break

      height, width = frame.shape[:2]
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
                      
            angle2=np.math.atan2(np.linalg.det([v2,v3]),np.dot(v2,v3))#rad
            grado2= np.degrees(angle2)
            grado2=round(grado2)
                      
            angle3=np.math.atan2(np.linalg.det([v4,v5]),np.dot(v4,v5))#rad
            grado3= np.degrees(angle3)
            grado3=round(grado3)
                      
            angle4=np.math.atan2(np.linalg.det([v6,v7]),np.dot(v6,v7))#rad
            grado4= np.degrees(angle4)
            grado4=round(grado4)
                      
            angle5=np.math.atan2(np.linalg.det([v8,v9]),np.dot(v8,v9))#rad
            grado5= np.degrees(angle5)
            grado5=round(grado5)
                      
            angle6=np.math.atan2(np.linalg.det([v10,v11]),np.dot(v10,v11))#rad
            grado6= np.degrees(angle6)
            grado6=round(grado6)
                
            result = model.predict([[grado1,grado2,grado3,grado4,grado5,grado6]])
                    
            if hand_label == "Right":
              if result[0] == 0:
                print("Uno")
              if result[0] == 1:
                print("Dos")
              if result[0] == 2:
                print("Tres")                  

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
