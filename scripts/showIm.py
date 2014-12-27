#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys
import time

size = (200, 200)
start = 0
stop = 10
#size = (10, 10)
#size = (39, 39)

if len(sys.argv) not in [2,3,4]:
    print sys.argv[0], "<image-matrix> [limit=" + str(limit) + "]"
    print sys.argv[0], "<image-matrix> <start> <stop>"
    print "\texpecting image-size", size
    exit(1)

if len(sys.argv) == 3:
    stop = int(sys.argv[2])

if len(sys.argv) == 4:
    start = int(sys.argv[2])
    stop = int(sys.argv[3])

f = open(sys.argv[1])

f.next()

for jj in xrange(start):
    f.next()

for jj in xrange(stop-start):
    j = f.next()
    r = [float(i)*255 for i in j.split(" ")]
    r = np.clip(r, 0, 255)
    r = list(np.array(r).astype(int))
    
    d = zip(r[0::3], r[1::3], r[2::3])

    try:
    
        ii = im.new("RGB", size = size)
    
        ii.putdata(d)

        if size[0] < 50:
            ii = ii.resize((size[0]*5, size[0]*5))

        ii.show()
        time.sleep(0.3)
    except:
        print "bad"

