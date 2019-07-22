import json
# import tensorflow as tf
# from tensorflow.python.ops import io_ops
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import wave
import random
import numpy as np
from dtw import dtw
import librosa

def getScore(request):
  print(request.form,request.args, request.data, request.files)

  return json.dumps({
    "score": random.randint(0, 98)
  })
  try:
    audio = request.files["audio"]
    audio.save("./recording.wav")
  except:
    print("Error opening file")
    # Return random int just cause
    return json.dumps({
      "score": random.randint(50, 98)
    })

  path = './korean_wav/'+ request.form.get("pronunciation") +'/rec.wav'
  print(path)
  y1, sr1 = librosa.load(path)
  y2, sr2 = librosa.load('./recording.wav')
  mfcc1 = librosa.feature.mfcc(y1, sr1)
  mfcc2 = librosa.feature.mfcc(y2, sr2)
  norm = lambda x, y: np.linalg.norm(x - y, ord=1)
  d, cost_matrix, acc_cost_matrix, path = dtw(mfcc1.T, mfcc2.T, dist=norm)

  # Shrink score for user
  score = int(round(d - 50 if d - 50 > 0 else 0))

  print(score)

  return json.dumps({
    "score": score
  })

def score():
  y1, sr1 = librosa.load('./korean_wav/ta/Untitled.wav')
  y2, sr2 = librosa.load('./korean_wav/o/Untitled_1.33.14_PM.wav')
  mfcc1 = librosa.feature.mfcc(y1, sr1)
  mfcc2 = librosa.feature.mfcc(y2, sr2)
  norm = lambda x, y: np.linalg.norm(x - y, ord=1)
  d, cost_matrix, acc_cost_matrix, path = dtw(mfcc1.T, mfcc2.T, dist=norm)
  print(d)
