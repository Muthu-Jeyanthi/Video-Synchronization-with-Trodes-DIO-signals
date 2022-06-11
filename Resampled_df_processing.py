# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 17:29:39 2021

@author: students
"""


from datetime import datetime , time , timedelta
from readTrodesExtractedDataFile3 import *
import video_sync_functions
import numpy as np
import os
import cv2
import pandas as pd
import sys
import matplotlib.pyplot as plt

#%% Video Files paths

video_loc_eye_1 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye01_2020-11-09_12-34-18.mp4'
video_loc_eye_2 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye02_2020-11-09_12-34-18.mp4'
video_loc_eye_3 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye03_2020-11-09_12-34-18.mp4'
video_loc_eye_4 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye04_2020-11-09_12-34-18.mp4'
video_loc_eye_5 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye05_2020-11-09_12-34-18.mp4'
video_loc_eye_6 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye06_2020-11-09_12-34-18.mp4'
video_loc_eye_7 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye07_2020-11-09_12-34-18.mp4'
video_loc_eye_8 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye08_2020-11-09_12-34-18.mp4'
video_loc_eye_9 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye09_2020-11-09_12-34-18.mp4'
video_loc_eye_10 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye10_2020-11-09_12-34-18.mp4'
video_loc_eye_11 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye11_2020-11-09_12-34-18.mp4'
video_loc_eye_12 = r'C:\Users\students\Documents\Hexmaze\maze_videos\eye12_2020-11-09_12-34-18.mp4'


fnames = [video_loc_eye_1 , video_loc_eye_2 , video_loc_eye_3 , video_loc_eye_4 , video_loc_eye_5 , video_loc_eye_6 , video_loc_eye_7 , video_loc_eye_8 , video_loc_eye_9 , video_loc_eye_10 , video_loc_eye_11 , video_loc_eye_12]


#%%
#Path to maze recording that contains the system time at creation information

dio_time_fname  = r"C:\Users\students\Documents\Hexmaze\Rat4_20201109\Rat4_20201109_maze.DIO\Rat4_20201109_maze.dio_MCU_Din1.dat"

#Path to maze recording with headstage recording data
blue_dio_fname = r'C:\Users\students\Documents\Hexmaze\Rat4_20201109\Rat4_20201109_maze_merged.DIO\Rat4_20201109_maze_merged.dio_MCU_Din1.dat'
blue_dict_dio = readTrodesExtractedDataFile(blue_dio_fname)
blue_DIO = blue_dict_dio['data']

red_dio_fname = r'C:\Users\students\Documents\Hexmaze\Rat4_20201109\Rat4_20201109_maze_merged.DIO\Rat4_20201109_maze_merged.dio_MCU_Din2.dat'
red_dict_dio = readTrodesExtractedDataFile(red_dio_fname)
red_DIO = red_dict_dio['data']

#Extract first and last timestamps from trodes system recording
sys_time = int(readTrodesExtractedDataFile(dio_time_fname)['system_time_at_creation'])/1000
dt_object = datetime.utcfromtimestamp(sys_time)

# Get first and last unix timestamps/ datetime for blue signals
blue_first_state_time = dt_object + timedelta(seconds = blue_DIO[0][0]/ 30000)
blue_first_trodes_ts = pd.Timestamp(blue_first_state_time).timestamp()
blue_last_state_time = dt_object + timedelta(seconds = blue_DIO[-1][0]/ 30000)
blue_last_trodes_ts = pd.Timestamp(blue_last_state_time).timestamp()

# Get first and last unix timestamps/ datetime for red signals
red_first_state_time = dt_object + timedelta(seconds = red_DIO[0][0]/ 30000)
red_first_trodes_ts = pd.Timestamp(red_first_state_time).timestamp()
red_last_state_time = dt_object + timedelta(seconds = red_DIO[-1][0]/ 30000)
red_last_trodes_ts = pd.Timestamp(red_last_state_time).timestamp()

# Get timestamps of Trodes signals
blue_DIO_ts = [ (pd.Timestamp(dt_object + timedelta(seconds = i[0]/ 30000)).timestamp() , i[1]) for i in blue_DIO]
red_DIO_ts = [ (pd.Timestamp(dt_object + timedelta(seconds = i[0]/ 30000)).timestamp() , i[1]) for i in red_DIO]



blue_DIO_df  = pd.DataFrame({"Time" : [datetime.fromtimestamp(i[0])  for i in blue_DIO_ts], "State": [i[1] for i in blue_DIO_ts]} )
blue_DIO_df = blue_DIO_df.set_index('Time' ,  verify_integrity = True) 


red_DIO_df  = pd.DataFrame({"Time" : [datetime.fromtimestamp(i[0])  for i in red_DIO_ts], "State": [i[1] for i in red_DIO_ts]} )
red_DIO_df = red_DIO_df.set_index('Time' ,  verify_integrity = True) 

resampled_blue_DIO_df = blue_DIO_df.resample('33L' , closed = 'right').ffill().reset_index()
resampled_red_DIO_df = red_DIO_df.resample('33L' , closed = 'right').ffill().reset_index()


#%% Get video dataframe
eye = 2
color ='blue'

resampled_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(eye) + color + '.csv' , usecols = [1 , 2 , 3] )
#resampled_video_df.fillna(0 , inplace = True)


#%%Replacing NANs

import video_sync_functions
eye = 12
color = 'blue'
cap = cv2.VideoCapture(fnames[eye-1])

resampled_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(eye) + color + '.csv' , usecols = [1 , 2 , 3] )
trial = resampled_video_df.copy()
nan_index = resampled_video_df['State'].index[resampled_video_df['State'].apply(np.isnan)]
print(len(nan_index))

for i in nan_index:
    frame = resampled_video_df['Frame'].ix[i]
    b_state , r_state = video_sync_functions.get_led_states_mod(cap , frame)
    
    resampled_video_df.loc[i , 'State'] = b_state

resampled_video_df.to_csv(r'C:\Users\students\Documents\Hexmaze\resampled_video_df\eye' + str(eye) + 'blue'+ '.csv')
nan_index = resampled_video_df['State'].index[resampled_video_df['State'].apply(np.isnan)]
print(len(nan_index))

#%% Get one single lag per eye

eye = [ 1, 2 , 3 , 4 , 5 ,6 ,  7, 8, 9, 10 , 11 , 12]
color = 'blue'

single_lag_df = pd.DataFrame(columns = [str(j) for j in eye])
#lags = []

state_df = pd.DataFrame(columns = [str(j) for j in eye])


for i in eye:
    
    if i == 6 : 
        single_lag_df['6'] = single_lag_df['5'].copy()
        state_df['6'] = state_df['5'].copy()
        continue;
        
    resampled_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(i) + color + '.csv' , usecols = [1 , 2 , 3] )

    resampled_video_df.fillna(0 , inplace = True)
    

    resampled_video_df.loc[: , ['State', 'Frame']]  = resampled_video_df[['State' , 'Frame']].shift(lags[i-1])
        
        
    single_lag_df[str(i)] = resampled_video_df['Frame'].copy()
    state_df[str(i)] =   resampled_video_df['State'].copy()

single_lag_df.dropna(inplace = True)
  
    
    lag , corr , _ , _ = plt.xcorr(resampled_blue_DIO_df['State'][0:len(resampled_video_df)].astype('float') , resampled_video_df.astype('float') ,   usevlines=True, maxlags= 20)
    
    best_lag = lag[np.argmax(corr)]
    print(best_lag , max(corr) , corr[np.where(lag == 0)])
    af_lags.append(best_lag)
    state_df[str(i)] = resampled_video_df.shift(best_lag)
    
    single_lag_df[str(i)] = single_lag_df[str(i)].shift(best_lag)
    
    start_ind =  single_lag_df[str(i)].first_valid_index()
    
    if start_ind != 0:
        first_shifted_frame = int(single_lag_df[str(i)].loc[start_ind])
        frame = first_shifted_frame -1
        
        for ind in range(start_ind-1 , -1 , -1  ):
            
            single_lag_df[str(i)].loc[ind ] = frame
            
            frame = frame -1 
print(af_lags)

check_lags = []
for i in eye:
    
    resampled_video_df = state_df[str(i)]
    resampled_video_df.fillna(0 , inplace = True)
    
    
    lag , corr , _ , _ = plt.xcorr(resampled_blue_DIO_df['State'][0:len(resampled_video_df)].astype('float') , resampled_video_df.astype('float') ,   usevlines=True, maxlags= 20)
    
    best_lag = lag[np.argmax(corr)]
    print(best_lag , max(corr) , corr[np.where(lag == 0)])
    check_lags.append(best_lag)
    
#%%Final DF with one lag and no correlation computed between eyes
    
single_lag_df.to_csv('Single_lag_frame_data_no_beteyecorr.csv')
#%%
import matplotlib.mlab as mlab
before_eyes_lag = state_df.fillna(5)
after_lag = before_eyes_lag.copy()
#plt.figure(figsize = (10 , 10))
#plt.imshow(state_df.corr())
#plt.colorbar()

before_corr = []
for i in [str(j) for j in eye]:
    #lag , corr , _ , _ = plt.xcorr(state_df[i][~ np.isnan(state_df[i])], state_df['10'][0:len(state_df[i][~ np.isnan(state_df[i])])] ,    usevlines=True, maxlags= 20)
    lag , corr , _ , _ = plt.xcorr(before_eyes_lag['1'] , before_eyes_lag[i],    usevlines=True, maxlags= 20) #, detrend=mlab.detrend_mean)

    print(min(corr) , max(corr) , corr[np.where(lag == 0)])
    best_lag = lag[np.argmax(corr)]
    after_lag.loc[: , i]  = after_lag[i].shift(best_lag)
    print(best_lag)
    #resampled_video_df.dropna(inplace = True)
    before_corr.append(corr[np.where(lag == 0)])
    
after_lag.fillna(5 , inplace = True)

after_corr = []

for i in [str(j) for j in eye]:
    #lag , corr , _ , _ = plt.xcorr(state_df[i][~ np.isnan(state_df[i])], state_df['10'][0:len(state_df[i][~ np.isnan(state_df[i])])] ,    usevlines=True, maxlags= 20)
    lag , corr , _ , _ = plt.xcorr(after_lag[i] , after_lag['1'],    usevlines=True, maxlags= 20) #, detrend=mlab.detrend_mean)

    print(min(corr) , max(corr) , corr[np.where(lag == 0)])
    print(lag[np.argmax(corr)])
    
    after_corr.append(corr[np.where(lag == 0)])
#%%
#from scipy import signal
#
#corr = signal.correlate(a,b , mode = 'valid') 
##lags = signal.correlation_lags(len(a), len(b))
#corr = corr / max(corr)
    
lag , corr , _ , _ = plt.xcorr(a , c[0:len(a)], usevlines=True, maxlags= 1 ) #, detrend=mlab.detrend_mean)

print(min(corr) , max(corr) , corr[1])
    
    
#%%

plt.figure(figsize = (10 , 10))
plt.plot(before_corr[1:])

plt.figure(figsize = (10 , 10))
plt.plot(after_corr[1:])

#
#before_corr = []
#after_corr = []
#for i in [str(j) for j in eye]:
#    af_coef = np.corrcoef(after_lag[i] , after_lag['1'] , rowvar = False)[0 , 1]
#    after_corr.append(af_coef)
#    bf_coef = np.corrcoef(before_eyes_lag [i] , before_eyes_lag ['1'] , rowvar = False)[0 , 1]
#    before_corr.append(bf_coef)
    
    
#%%  Using just 10 segments for all eyes

import video_sync_functions
eye = [  1, 2 , 3 , 4 , 5 , 6 ,  7, 8, 9, 10 , 11 , 12]
color = 'blue'

acc_list = []

segments_lag_df = pd.DataFrame(columns = [str(j) for j in eye])
list_df = []

for i in eye:
    
    if i == 6 : 
        list_df.append(final_blue_df.copy())
        continue;
        
    resampled_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(i) + color + '.csv' , usecols = [1 , 2 , 3] )
    resampled_video_df.fillna(0 , inplace = True)
    
    blue_wtlcc , blue_index_ranges = video_sync_functions.WTLCC(resampled_blue_DIO_df , resampled_video_df , 10 , plot = 1)

    final_blue_df   = video_sync_functions.get_sync_video_df(resampled_blue_DIO_df, resampled_video_df , blue_wtlcc, blue_index_ranges ,0 , 30 )       
    
    start_ind =  final_blue_df['Frame'].first_valid_index()
    
    if start_ind != 0:
        first_shifted_frame = int(final_blue_df.loc[start_ind , 'Frame'])
        frame = first_shifted_frame -1
        
        for ind in range(start_ind-1 , -1 , -1  ):
            
            final_blue_df.loc[ind , "Frame"] = frame
            
            frame = frame -1 
    
    
    
    
    final_blue_df = final_blue_df.loc[~final_blue_df.index.duplicated()]          
    list_df.append(final_blue_df)
    
    a = np.asarray(final_blue_df['State'])
    b = np.asarray(resampled_blue_DIO_df['State'])
    c = np.where(a == b[0:len(a)])[0]
    acc = (len(c) / len(a)) * 100
    
    acc_list.append((i , acc))
        
# Get frames df for segments

segments_lag_df = pd.concat([i['Frame'] for i in list_df] , axis = 1)

segments_lag_df.columns = [str(i) for i in eye]
segments_lag_df['6'] = segments_lag_df['5'].copy()

segments_lag_df.dropna(inplace = True)

# Get states df for segments

state_segments_df = pd.concat([i['State'] for i in list_df] , axis = 1)
state_segments_df.columns = [str(i) for i in eye]
state_segments_df['6'] = state_segments_df['5'].copy()

#%% Correlation between eyes with segments



#%% Code segment to stitch the frames from different cameras together
video_name = 'stitched_both_RB_blue_led.avi'
sync_df = single_lag_df.copy()
sync_df.dropna(inplace = True)
eyes = list(range(1 , 13))

writer = cv2.VideoWriter(video_name , cv2.VideoWriter_fourcc(*'DIVX'), 30., (2430 , 1418))
frame = cv2.VideoCapture()
# cv2.VideoWriter_fourcc(*'mp4v')
for i in range(0 , len(sync_df)):  # length of frames
    frames = []
    for ind , j in enumerate(eyes):
        
        if ind <= 5:
            cap = cv2.VideoCapture(fnames[ind])
            cap.set(cv2.CAP_PROP_POS_FRAMES , sync_df.iloc[i , ind])
            res , frame = cap.read()
            frames.append( frame[:800-91 , 104: 600-91 , : ])
             
        else:   
            cap = cv2.VideoCapture(fnames[ind])
            cap.set(cv2.CAP_PROP_POS_FRAMES , sync_df.iloc[i , ind])
            res , frame = cap.read()
            frame = frame[:800-91 , 104: 600-91 , : ]
            frame = cv2.flip(frame , -1)
            frames.append(frame)
        
    # concatenating images
    
    top_row = cv2.hconcat(frames[0:6])
    bottom_row = cv2.hconcat(frames[6:])
    complete_img = cv2.vconcat([top_row , bottom_row])
    #print(complete_img.shape)
    writer.write(complete_img)   
        

#%% Using both LED states
    
eye = 1


resampled_blue_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(eye) + color1 + '.csv' , usecols = [1 , 2 , 3] )
#resampled_video_df.fillna(0 , inplace = True)

resampled_red_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(eye) + color2 + '.csv' , usecols = [1 , 2 , 3] )
#resampled_video_df.fillna(0 , inplace = True)


#%% Convert 0 , 1 to 0 , 1, 2, 3 & 4
     
color1 ='blue'
color2  = 'red'
def categorize(df1 , df2):
    DIO_cat = []
    
    merge_DIO_df = pd.merge(df1 , df2 , on = 'Time' , how = 'inner' )[['Time' , 'State_x' , 'State_y']]
    print(len(merge_DIO_df))
    
    # df 1 - blue (state_x)
    #df2 - red  (state_y)

    
    for i in range(len(merge_DIO_df)):
        if merge_DIO_df.loc[i , 'State_x'] == 0 and merge_DIO_df.loc[i , 'State_y'] == 0 :
            DIO_cat.append(200)
        elif merge_DIO_df.loc[i , 'State_x'] == 1 and merge_DIO_df.loc[i , 'State_y'] == 0:  
            DIO_cat.append(500)
        elif merge_DIO_df.loc[i , 'State_x'] == 0 and merge_DIO_df.loc[i , 'State_y'] == 1 :
            DIO_cat.append(2000)
        elif merge_DIO_df.loc[i , 'State_x'] == 1 and merge_DIO_df.loc[i , 'State_y'] == 1 :
            DIO_cat.append(5000)
        else:
            DIO_cat.append(np.nan)
            
    return DIO_cat
    
eye = [  1, 2 , 3 , 4 , 5 , 6 ,  7, 8, 9, 10 , 11 , 12]


merged_DIO_df = categorize(resampled_blue_DIO_df , resampled_red_DIO_df)



merged_eyes = pd.DataFrame(columns = [str(j) for j in eye])


for j in eye:
    
    if j == 6:
        resampled_blue_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(5) + color1 + '.csv' , usecols = [1 , 2 , 3] )
        resampled_red_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(5) + color2 + '.csv' , usecols = [1 , 2 , 3] )
    else:
        
        resampled_blue_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(j) + color1 + '.csv' , usecols = [1 , 2 , 3] )
        resampled_red_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(j) + color2 + '.csv' , usecols = [1 , 2 , 3] )
    resampled_red_video_df.fillna(0 , inplace = True)
    resampled_blue_video_df.fillna(0 , inplace = True)
    merged_eyes[str(j)] = pd.Series(categorize(resampled_blue_video_df , resampled_red_video_df))
    
merged_eyes.dropna(inplace = True)   
    
#%%
merged_lags = []
for i in eye: 

    lag , corr , _ , _ = plt.xcorr(pd.Series(merged_DIO_df).astype('float')[0:len( merged_eyes[str(i)])] , merged_eyes[str(i)].astype('float') ,   usevlines=True, maxlags= 100)
    
    best_lag = lag[np.argmax(corr)]
    print(best_lag , max(corr) , corr[np.where(lag == 0)])   
    
    merged_lags.append((best_lag , max(corr)))
    
  

#%%
    
#merge_DIO_df.to_csv('DIO_states_blue_red.csv')

for j in eye:
    
    if j == 6:
        resampled_blue_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(5) + color1 + '.csv' , usecols = [1 , 2 , 3] )
        resampled_red_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(5) + color2 + '.csv' , usecols = [1 , 2 , 3] )
    else:
        
        resampled_blue_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(j) + color1 + '.csv' , usecols = [1 , 2 , 3] )
        resampled_red_video_df = pd.read_csv(r"C:\Users\students\Documents\Hexmaze\resampled_video_df\eye" + str(j) + color2 + '.csv' , usecols = [1 , 2 , 3] )
    resampled_red_video_df.fillna(0 , inplace = True)
    resampled_blue_video_df.fillna(0 , inplace = True)
    
    df = pd.DataFrame(zip(resampled_blue_video_df['State'] , resampled_red_video_df['State']) , columns = ['blue' , 'red'])

    df.to_csv(r'C:\Users\students\Documents\Hexmaze\REd_blue_states\eye' + str(j) + '.csv'  )
