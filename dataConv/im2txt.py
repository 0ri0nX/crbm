#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys

size = (200,200)

limit = 5000

data = sys.argv[1:limit+1]

print len(data), size[0]*size[1]*3

for idx, iii in enumerate(data):
    ii = im.open(iii)
    
    ii = ii.resize(size)
    
    ii = ii.convert("RGB")
    
    d = (np.array(ii.getdata())/255.0).flatten().tolist()
    assert len(d) == size[0]*size[1]*3
    d = ["{0:g}".format(i) for i in d]
    sys.stderr.write(str(idx) + "\r")
    sys.stderr.flush()
    
    print " ".join(d)

sys.stderr.write("\n")

