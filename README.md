# Video-Synchronization-with-Trodes-DIO-signals
Python module to find the start and end points of the video recording using the Trodes DIO signals corresponding to the LED on and off blinking states along with a method to check video alignment.   

## Python Packages to be installed
* [pytesseract](https://pypi.org/project/pytesseract/) - Download [Tesseract OCR ](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20201127.exe) , complete the setup and **take note of the complete path to tesseract.exe in Tesseract-OCR folder**.
* [openCV](https://pypi.org/project/opencv-python/) 
* pandas, numpy 

[**readTrodesExtractedDataFile3.py**](https://github.com/Muthu-Jeyanthi/Video-Synchronization-with-Trodes-DIO-signals/blob/main/readTrodesExtractedDataFile3.py) reads the .dat files extracted from the Trodes recording (.rec) files and creates a dictionary with data and session information. 

[**video_sync_functions.py**](https://github.com/Muthu-Jeyanthi/Video-Synchronization-with-Trodes-DIO-signals/blob/main/video_sync_functions.py) contains the following functions:
* Digit recognition from a frame (*get_digits*)
* Find the difference between the gpu-level and python-level frame indices to calculate the shift required to match the timestamps with the appropriate frame (*index_difference*)
* Get LED states from the frame  (*get_led_states*)
* Get video timestamps with corresponding LED states from each frame during the same time interval of Trodes recording. (*video_metadata*)
* Get the closest timestamp to the query timestamp from a list of timestamps. (*closest_ts*)
* Get the instances when there is a match and mismatch between the LED states in video and DIO signals at the closest timestamp. (*find_mismatch*) 

[**check_video_alignment.py**](https://github.com/Muthu-Jeyanthi/Video-Synchronization-with-Trodes-DIO-signals/blob/main/check_video_alignment.py)  computes the first and last timestamps from the Trodes DIO signals and utilizes the video_sync functions to obtain the video timestamps and states along with the match and mismatch. 
