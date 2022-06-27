''' Code to convert uav videos to frames with necesarry preprocessing'''

vid = 'VIDEO NAME'
path = "ENTER DIRECTORY PATH HERE"

import os
try:
    os.mkdir(path)
except OSError as error:
    print(error)    
	
import cv2
vidcap = cv2.VideoCapture(path + ".mp4")
success,image = vidcap.read()
count = 0
while success:
    # uncomment for color inversion
    # image = cv2.bitwise_not(image)
    
    # uncomment for vid smoothing
    # image = cv2.fastNlMeansDenoising(image,None,5,7,21) 
    
    # save frame as JPEG file
    cv2.imwrite(path + f"\\{vid}_frame{count}.jpg",image)     
    print('Read a new frame: ', success)
    success,image = vidcap.read()
    count += 1