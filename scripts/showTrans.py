#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys

#size = (200, 200)
#size = (10, 10)
size = (39, 39)


#layers = 3
layers = 39*39
features = 15

f = open(sys.argv[1])

f.next()

limit = 1

for j in f:
    if limit <=0:
        break
    limit -= 1

    r = np.array([float(i) for i in j.split(" ")])

    for ll in xrange(features):
    
        #d = zip(r[0::layers], r[1::layers], r[2::layers])
        #d = zip(r[0:layers], r[layers:layers*2], r[layers*2:layers*3])

        dd = r[ll::features]

        mi = np.min(dd)
        ma = np.max(dd)
        dd = np.clip(255.0*(dd-mi)/(ma-mi), 0, 255)
        dd = list(np.array(dd).astype(int))
        
        d = zip(dd,dd,dd)
        
        ii = im.new("RGB", size = size)
        
        ii.putdata(d)
        
        ii = ii.resize((size[0]*5, size[0]*5))

        ii.show()

