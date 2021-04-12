# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:44:37 2020

@author: archi
"""


import scipy.io.wavfile as wav
import pandas as pd
import os
import pyaudio
import wave

os.chdir('E:\\Sem 5\\PROJECTS\\Digital Signal Processing Project')

def lengthfinder(file_path):
    (source_rate, source_sig) = wav.read(file_path)
    duration_seconds = len(source_sig) / float(source_rate)
    print('The length of the recorded Audio in seconds is: ', end='')
    print(round(duration_seconds))

print("Enter your name to register for voice recognition: ")
name = input()

#Recording audio
print("The audio will now be recorded for 5 seconds")

# the file name output you want to record into
path = "E:/Sem 5/PROJECTS/Digital Signal Processing Project/User_Audio/"
filename = path+name+"_recorded.wav"
# set the chunk size of 1024 samples
chunk = 1024
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 44100 samples per second
sample_rate = 44100
record_seconds = 5
# initialize PyAudio object
p = pyaudio.PyAudio()
# open stream object as input & output
stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)
frames = []
print("Recording...")
for i in range(int(44100 / chunk * record_seconds)):
    data = stream.read(chunk)
    # if you want to hear your voice while recording
    # stream.write(data)
    frames.append(data)
print("Finished recording.")
# stop and close stream
stream.stop_stream()
stream.close()
# terminate pyaudio object
p.terminate()
# save audio file
# open the file in 'write bytes' mode
wf = wave.open(filename, "wb")
# set the channels
wf.setnchannels(channels)
# set the sample format
wf.setsampwidth(p.get_sample_size(FORMAT))
# set the sample rate
wf.setframerate(sample_rate)
# write the frames as bytes
wf.writeframes(b"".join(frames))
# close the file
wf.close()

    
# Iterate through each sound file and extract the features 
'''
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
'''
lengthfinder(filename)