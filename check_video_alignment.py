# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:17:27 2021

@author: muthu
"""

from datetime import datetime , time , timedelta
from readTrodesExtractedDataFile3 import *
import video_sync_functions
import numpy as np
import os
import pandas as pd
import sys
#%%  Extracting DIO signals

dio_fname  = r"D:\EEG data analysis\wireless test maze\rat2_20200905_wirelesstest\Rat4_20201109_postsleep.dio_MCU_Din1.dat"
dict_dio = readTrodesExtractedDataFile(dio_fname)
DIO = dict_dio['data']


#%%
#Extract first and last timestamps from trodes system recording
sys_time = int(dict_dio['system_time_at_creation'])/1000
dt_object = datetime.utcfromtimestamp(sys_time)
first_state_time = dt_object + timedelta(seconds = DIO[0][0]/ 30000)
first_trodes_ts = pd.Timestamp(first_state_time).timestamp()
last_state_time = dt_object + timedelta(seconds = DIO[-1][0]/ 30000)
last_trodes_ts = pd.Timestamp(last_state_time).timestamp()

# Get timestamps of Trodes signals
DIO_ts = [ (pd.Timestamp(dt_object + timedelta(seconds = i[0]/ 30000)).timestamp() , i[1]) for i in DIO]


#%% Getting the video ts and led states

video_loc = r'D:\EEG data analysis\wireless test maze\rat2_20200905_wirelesstest\eye01_2020-11-09_13-43-12.mp4'

video_index_ts_state , info_dict = video_sync_functions.video_metadata(video_loc, first_trodes_ts, last_trodes_ts)

#%% Comparing Trodes LED states and video LED states to find mismatches

match_on , mismatch_on = video_sync_functions.find_mismatch(video_index_ts_state , DIO_ts , state=1)

match_off , mismatch_off = video_sync_functions.find_mismatch(video_index_ts_state , DIO_ts , state=1)


#%% Test area









