import json
# import tensorflow as tf
# from tensorflow.python.ops import io_ops
# from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import wave
import random


def getScore(request):
  try:
    print(request.files["audio"])

  except:
    print("Error opening file")

  return json.dumps({
    "score": random.randint(50, 98)
  })

  # 1. loop files, save output to numpy arr of outputs
  # 2. numpy arr of arr  or labels, integers to each
  # index of labels match the outputs

  # toCategorical on labels fo formatting

  # declare a model

  # run examples through it

  # save model

  # load model
