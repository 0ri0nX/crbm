#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys

#size(200, 200)
size = (10, 10)

f = open(sys.argv[1])

f.next()
m = []
for j in f:
    r = [float(i) for i in j.split(" ")]
    m.append(r)

m = np.array(m)
m = m.transpose()

#mi = np.min(m)
#ma = np.max(m)
#m = np.clip(255.0*(m-mi)/(ma-mi), 0, 255).astype(int)

limit = 100

for i in xrange(min(limit, m.shape[0])):

    r = m[i]

    mi = np.min(r)
    ma = np.max(r)
    r = np.clip(255.0*(r-mi)/(ma-mi), 0, 255).astype(int)

    r = list(r)
    d = zip(r[0::3], r[1::3], r[2::3])
    
    ii = im.new("RGB", size = size)
    
    ii.putdata(d)
    
    ii = ii.resize((100, 100))

    ii.show()

