import numpy as np
import parseData as pd
from dtw import dtw

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
# x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
# y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
a = pd.fileToArray("/Users/dylandesrosier/Documents/Projects/koreanPhonemeRecognition/korean_wav/a/Untitled.wav")
b = pd.fileToArray("/Users/dylandesrosier/Documents/Projects/koreanPhonemeRecognition/korean_wav/a/Untitled 4.wav")
print(a[0], b[0])

c = []
for i in a:
  c.append(int(round(i*1000000)))
d = []
for j in b:
  d.append(int(round(j*1000000)))
print(c[0], d[0], len(c), len(d))
x = np.array(c).reshape(-1, 1)
y = np.array(d).reshape(-1, 1)
print(x[0], y[0])

euclidean_norm = lambda x, y: np.abs(x - y)

d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)

print(d)