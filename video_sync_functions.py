# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:08:23 2021

@author: muthu
"""

from datetime import datetime , time , timedelta
from decimal import Decimal as D
import pytz
import pandas as pd
import cv2
import numpy as np
import pytesseract
import random
from scipy import stats

#change it to argument

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\muthu\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

#%%
def get_digits(frame , top , bottom , left , right ):
    
    hsv = cv2.cvtColor(frame[top:bottom, left:right, :] , cv2.COLOR_BGR2HSV)
    kernel = np.ones((1, 1), np.uint8)
    hsv = cv2.dilate(hsv, kernel, iterations=1)
    hsv= cv2.erode(hsv, kernel, iterations=1)
    
        # define range of white color in HSV
        # change it according to your need !
    sensitivity = 30
    lower_white = np.array([0,0,255-sensitivity], dtype=np.uint8)
    upper_white = np.array([255,sensitivity,255], dtype=np.uint8)
    
    
    # Threshold the HSV image to get only white colors
    mask= cv2.inRange(hsv, lower_white, upper_white)
    
    result = pytesseract.image_to_string(mask,  lang='eng' , config='--psm 9 --oem 1 -c tessedit_char_whitelist=0123456789')
        
    return result

#%%

#Get difference between indices across different regions of a video frame

def index_difference (videoCapture) : 
    
    capt = videoCapture
    frames_to_check = random.sample(range(50, int(capt.get(cv2.CAP_PROP_FRAME_COUNT))), 25)
    differences = []
    for i in frames_to_check:
        capt.set(cv2.CAP_PROP_POS_FRAMES, i)
        res, frame = capt.read()
        try:
            index_bottom = int(get_digits(frame , 51 ,80 , 250 , 400)[0:8])
            index_top = int(get_digits(frame, 0,50, 380,480)[0:10])
            differences.append(abs(index_top-index_bottom))
            
        except:
            pass
      
    capt.release()  
            
            
    return int(np.median(differences))

#%% function to detect on and off LED states

def get_led_states(cap , frame_index_with_shift):
    
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index_with_shift)
        res2, frame2 = cap.read()
     
        led_region = frame2[730:735 , 560:569 , :]
        led_region = cv2.resize(led_region , (75,75))
        #cv2.imshow(str(i) , cropped)
        #cv2.waitKey()
        
        #Thresholding to detect blue light
        th_f = led_region.copy()
        frame_HSV = cv2.cvtColor(th_f, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (98,108,20), (120,255,255))
        
        #Draw contours based on thresholding
        contd_img = led_region.copy()
        contours, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contd_img, contours, -1, (0,255,0), 3)
        
        
        # Store contour properties and corresponding timestamps in the dicts
        if len(contours) >0 :
            
            cnt_area = [cv2.contourArea(contours[c]) for c in range(0,len(contours))]
            cnt = contours[cnt_area.index(max(cnt_area))]
            
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            
            circled_img = led_region.copy()
            cv2.circle(circled_img,(int(x),int(y)),int(radius),(0,255,0),2)
            
            if (max(cnt_area) >= 2500) and (int(x) > 20 and int(x) <= 40) and (int(y) > 30 and int(y) < 40):
                
                return 1
             
                #on_states_dict['on_state_index'].append(i)
                #on_states_dict['area'].append(cv2.contourArea(cnt))
                #dict['perimeter'].append(cv2.arcLength(cnt,True))
                #on_states_dict['center'].append((int(x),int(y)))
                #on_states_dict['radius'].append(int(radius))
                #on_states_dict['on_states_ts'].append(numbers[i])
                
            else:
                return 0
                #off_states_dict['off_state_index'].append(i)
                #off_states_dict['off_states_ts'].append(numbers[i])
                #off_states_dict['area'].append(cv2.contourArea(cnt))
                #dict['perimeter'].append(cv2.arcLength(cnt,True))
                #off_states_dict['center'].append((int(x),int(y)))
                #off_states_dict['radius'].append(int(radius))
            # intensity_img = cropped.copy()
            # mask = np.zeros(cv2.cvtColor(intensity_img, cv2.COLOR_BGR2GRAY).shape,np.uint8)
            # cv2.drawContours(mask,[cnt],0,255,-1)
            # pixelpoints = np.transpose(np.nonzero(mask))
            # dict['mean_val'] = cv2.mean(intensity_img ,mask = mask)
        else:
            return 0
            #off_states_dict['off_state_index'].append(i)
            #off_states_dict['off_states_ts'].append(numbers[i])
            #off_states_dict['area'].append(0)
            #dict['perimeter'].append(cv2.arcLength(cnt,True))
            #off_states_dict['center'].append(("" , ""))
            #off_states_dict['radius'].append(0)
    
    
#%%

#Combining all functions

def video_metadata (fileloc, first_trodes_ts , last_trodes_ts):
    
    count = 0 
    info_dict= []
    #Reading the video file
    cap = cv2.VideoCapture(fileloc)
    led_cap = cv2.VideoCapture(fileloc)
    #Getting frame shift between gpu level and python processor
    shift = index_difference(cap)  
    print(shift)       
    cap = cv2.VideoCapture(fileloc)                             
    #Store first frame ts for use when OCR fails
    first_frame_ts = []
    #Create an array to store video timestamps with corresponding state
    video_index_ts_state = []
    
    #Toggle to get LED states of shifted frame or not
    read_LED_states = 0
    #Declare previous ts variable
    prev_num_ts = 0
    #Loop through the frames to get the numbers 
    #Simultaneous check the whether it is the close to the first timestamp
    while (cap.isOpened()):
        
        count = count +1
        if count > 20:
            break;
    
        #Read the frame 
        res, frame = cap.read()
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)-1)
        
        # Use get_digits function to get the digits as a string
        # Timestamps region - frame[0:50, 162:460, :]
        
        num = get_digits(frame ,0,50,162 , 460 )
        s = num.find('2020')
        try:
            num_ts = pd.Timestamp(datetime(year = int(num[s:s+4]), month = int(num[s+4:s+6]) , day = int(num[s+6:s+8]) , hour = int(num[s+8:s+10]), minute = int(num[s+10:s+12]), second = int(num[s+12:s+14]), microsecond = int( num[s+14:s+20]))).timestamp()
            if frame_index == 0:
                first_frame_ts.append(num_ts)
                print(first_frame_ts)
                # Display the first ts and the OCR recognized ts in the header
                cv2.imshow(str(pd.Timestamp(datetime(year = int(num[s:s+4]), month = int(num[s+4:s+6]) , day = int(num[s+6:s+8]) , hour = int(num[s+8:s+10]), minute = int(num[s+10:s+12]), second = int(num[s+12:s+14]), microsecond = int( num[s+14:s+20])))) , frame[0:50, 162:460 , :])
                cv2.waitKey(5000)
                # Confirm that both are the same.
                first_digits_correct = input('Were the header and frame timestamps the same? Type y or n.')
                if first_digits_correct == 'y':
                    continue
                else:
                    # If it is wrong enter the digits manually.
                    num = input('Write the digits from 2020 without space.')
                    num_ts = pd.Timestamp(datetime(year = int(num[s:s+4]), month = int(num[s+4:s+6]) , day = int(num[s+6:s+8]) , hour = int(num[s+8:s+10]), minute = int(num[s+10:s+12]), second = int(num[s+12:s+14]), microsecond = int( num[s+14:s+20]))).timestamp()
                
        except: 
            num_ts = first_frame_ts[0] + (frame_index* 0.033333)
        
        else:
            #If the ts is much higher than it should be
            if (num_ts > prev_num_ts) and ((num_ts - prev_num_ts) > 0.04):
                num_ts = first_frame_ts[0] + (frame_index* 0.033333)
            # If the ts is lesser than the previous one
            elif (num_ts < prev_num_ts):
                num_ts = first_frame_ts[0] + (frame_index* 0.033333)
            #else:
             #   num_ts = pd.Timestamp(datetime(year = int(num[s:s+4]), month = int(num[s+4:s+6]) , day = int(num[s+6:s+8]) , hour = int(num[s+8:s+10]), minute = int(num[s+10:s+12]), second = int(num[s+12:s+14]), microsecond = int( num[s+14:s+20]))).timestamp()
        prev_num_ts = num_ts 

          
    
        #only for one ts this difference will be <= 0.01 
        if round(abs(num_ts - first_trodes_ts) , 2) <= 0.01:
            read_LED_states = 1  #toggle to read led states
            start_video_ts = num_ts
            first_frame_shifted_index = frame_index+shift
            print(round(abs(num_ts - first_trodes_ts) , 2) , start_video_ts , first_frame_shifted_index)
            a = input("start index reached")
        if read_LED_states:
            
            state = get_led_states(led_cap , frame_index+shift)
            video_index_ts_state.append(frame_index+shift , num_ts , state ))
            
            if round(abs(num_ts - last_trodes_ts) , 2 )< 0.01 :
                end_video_ts = num_ts
                last_frame_shifted_index = frame_index+shift
                print(end_video_ts , last_frame_shifted_index)
                info_dict = {"shift":shift , "first_frame_ts" : first_frame_ts , "start_video_ts" :start_video_ts , 
                  "end_video_ts" : end_video_ts , "first_frame_shifted_index ":first_frame_shifted_index   ,
                  "last_frame_shifted_index": last_frame_shifted_index }     
                break;
                 
                 
            
    
    
    return video_index_ts_state , info_dict
#%%
# Find closest timestamp to the given timestamp in the array

def closest_ts(lst , ts):
     array = np.asarray(lst)
     idx = (np.abs(array - ts)).argmin()
     return idx , array[idx]


#%% Find mismatch between video timestamp/state and trodes timestamp/state

def find_mismatch(video_index_ts_state , trodes_ts_state , state):
    match = []
    mismatch = []
    video_ts_list = [m[1] for m in video_index_ts_state if m[2] == state]
    trodes_ts_list  = [j[0] for j in trodes_ts_state if j[1]==state]
    for i in video_ts_list:
        closest_idx , closest_val = closest_ts(trodes_ts_list , i)
        
        if round(abs(i - closest_val) , 2)  <= 0.01:
            match.append((i , closest_val))
            
        else:
            mismatch.append((i , closest_val))
            
    return match , mismatch

#%%
        

# cap = cv2.VideoCapture(r'D:\EEG data analysis\wireless test maze\rat2_20200905_wirelesstest\eye01_2020-11-09_13-43-12.mp4')
# # while (cap.isOpened()):
# #    res, frame = cap.read()
# #    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
# #    print(frame_index)
# #    if frame_index > 10 :
# #        break;













