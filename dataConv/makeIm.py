#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys

size = (200, 200)
#size = (10, 10)
#size = (39, 39)

prefix = sys.argv[2]
f = open(sys.argv[1])

f.next()

limit = 10

for idx, j in enumerate(f):
    limit -= 1
    if limit <=0:
        break

    r = [float(i)*255 for i in j.split(" ")]
    r = np.clip(r, 0, 255)
    r = list(np.array(r).astype(int))
    
    d = zip(r[0::3], r[1::3], r[2::3])

    try:
    
        ii = im.new("RGB", size = size)
    
        ii.putdata(d)

        if size[0] < 50:
            ii = ii.resize((size[0]*5, size[0]*5))

        #ii.show()
        ii.save(prefix + "-" + str(idx).zfill(2) + ".jpg")
    except:
        print "bad"

