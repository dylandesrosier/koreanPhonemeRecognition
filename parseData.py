
import numpy
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fftpack import fft
import os
from enum import Enum
import sys
import math

numpy.set_printoptions(threshold=sys.maxsize)

class labels(Enum):
   cha = 1
   uh = 2
   ga = 3
   ma = 4
   da = 5
   na = 6
   ja = 7
   i = 8
   a = 9
   o = 10
   ya = 11
   yo = 12
   sa = 13
   yu = 14
   pa = 15
   ba = 16
   ha = 18
   ah = 19
   ka = 20
   oo = 21
   yoo = 22
   eu = 23
   la = 24
   ta = 0

def fileToArray(path):

  #Read file  sound object
  _, mySound = wavfile.read(path)
  mySound = mySound / (2.**15)
  mySoundShape = mySound.shape

  #If two channels, then select only one channel
  try:
      mySoundOneChannel = mySound[:,0]
  except IndexError as e:
      mySoundOneChannel = mySound

  return mySoundOneChannel

def iterateOverData():
  path ="/Users/dylandesrosier/Documents/Projects/koreanPhonemeRecognition/korean_wav"
  sounds = []
  labelsList = []

  for subdir, dirs, _ in os.walk(path):
    for dir in dirs:
      for sd, _, files in os.walk(subdir + '/' + dir):
        for file in files:
          if (file == ".DS_Store"):
            continue
          #print(sd + '/' + file)
          a = sd + '/' + file
          b = fileToArray(a)
          orig = len(b)
          c = truncate(b)
          d = truncate(c[::-1])[::-1]
          e = pad(d)
          after = len(e)

          print(orig, after, a)
          if dir == "ba":
            plt.plot([x for x in range(len(e))], e)
            plt.show()

          sounds.append(e)
          labelsList.append(labels[dir].value)
  
  return sounds, labelsList

def truncate(b):
  count = 0
  for i in b:
    if i < 0.05 and i > -0.05:
      count += 1
    else:
      if count > 1500:
        break
  
  return b[count:]

def pad(d):
  amt = 30433
  while len(d) < amt:
    d = numpy.append(d, 0)
  return d

# a, b = iterateOverData()
# print(a, b)


