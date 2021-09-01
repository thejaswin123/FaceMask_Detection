import time
from selenium import webdriver
cd='C:\\Users\\theja\\Downloads\\chromedriver_win32\\chromedriver.exe'
driver=webdriver.Chrome(cd)
email = ""
password = ""



driver.get("https://www.tinkercad.com/dashboard")
time.sleep(2)

google = driver.find_element_by_xpath('//*[@id="content"]/div/main/ng-component/main/section/div/div/div/div/div[1]/a[2]/span[2]')
google.click()

driver.find_element_by_xpath('//*[@id="userName"]').send_keys(email)
user_next=driver.find_element_by_xpath('//*[@id="verify_user_btn"]')
user_next.click()  
time.sleep(2)

driver.find_element_by_xpath('//*[@id="password"]').send_keys(password)
pwd_next = driver.find_element_by_xpath('//*[@id="btnSubmit"]')
pwd_next.click()
time.sleep(5)

driver.get('https://www.tinkercad.com/things/1rnpxvBCIBk')
time.sleep(1)

#Tinker this
driver.find_element_by_xpath('//button[@class="btn btn-lg btn-primary"]').click()
time.sleep(2)


#Code
driver.find_element_by_id('CODE_EDITOR_ID').click()
time.sleep(2)

#Serial Monitor
driver.find_element_by_id('SERIAL_MONITOR_ID').click()
time.sleep(2)

#Simulate
driver.find_element_by_id('SIMULATION_ID').click()

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import json


# Model reconstruction from JSON file
with open(r"model_architecture_FaceMask_Detection.json", 'r') as f:
    model=model_from_json(f.read())

# Load weights into the new model"""
model.load_weights(r"FaceMask_Detection.h5")


import cv2
import numpy as np

label = {0:"With Mask",1:"Without Mask"}
color_label = {0: (0,255,0),1 : (0,0,255)}

cap = cv2.VideoCapture(0) 

cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')


while True:
    (rval, frame) = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray,1.1,4)
    
    for x,y,w,h in faces:
        face_image = frame[y:y+h,x:x+w]
        resize_img  = cv2.resize(face_image,(150,150))
        normalized = resize_img/255.0
        reshape = np.reshape(normalized,(1,150,150,3))
        reshape = np.vstack([reshape])
        result = model.predict(reshape)
        result = result[0][0]
        
        if result <= 0.5:
            cv2.rectangle(frame,(x,y),(x+w,y+h),color_label[0],3)
            cv2.rectangle(frame,(x,y-50),(x+w,y),color_label[0],-1)
            cv2.putText(frame,label[0],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            #Input Box
            serial_input = driver.find_element_by_xpath('//div[@class="code_panel__serial__bottom js-code_panel__serial__bottom js-code_editor__serial-monitor__bottom clearfix"]/input')
            serial_input.send_keys(0) #  input high
            driver.find_element_by_xpath('//div[@class="code_panel__serial__bottom js-code_panel__serial__bottom js-code_editor__serial-monitor__bottom clearfix"]/div/a/div').click()
            time.sleep(0.5)
        elif result > 0.5:
            cv2.rectangle(frame,(x,y),(x+w,y+h),color_label[1],3)
            cv2.rectangle(frame,(x,y-50),(x+w,y),color_label[1],-1)
            cv2.putText(frame,label[1],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            #Input Box
            serial_input = driver.find_element_by_xpath('//div[@class="code_panel__serial__bottom js-code_panel__serial__bottom js-code_editor__serial-monitor__bottom clearfix"]/input')
            serial_input.send_keys(1) #  input low
            driver.find_element_by_xpath('//div[@class="code_panel__serial__bottom js-code_panel__serial__bottom js-code_editor__serial-monitor__bottom clearfix"]/div/a/div').click()
            time.sleep(0.5)
            
    cv2.imshow('LIVE',   frame)
    key = cv2.waitKey(10)
    
    if key==27:
        break

cap.release()

cv2.destroyAllWindows()
