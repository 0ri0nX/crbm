#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys
import struct

size = (200,200)

if len(sys.argv) != 3:
    print sys.argv[0], "<output-file> <limit>"
    print "  Reads image-names from standard input"
    exit(1)


data = [i.strip() for i in sys.stdin]
limit = int(sys.argv[2])
#data = data[:limit]

f=open(sys.argv[1], "wb")

VERSION = 1

f.write("Matrix " + str(VERSION) + "\n")
f.write(str(len(data)) + " " + str(size[0]*size[1]*3) + "\n")
#f.write(struct.pack("<ii", len(data), size[0]*size[1]*3))

fs = "<" + "B"*(size[0]*size[1]*3)

for idx, iii in enumerate(data):
    if idx >= limit:
        break
    ii = im.open(iii)
    
    ii = ii.resize(size)
    
    ii = ii.convert("RGB")
    
    d = (np.array(ii.getdata())).flatten().tolist()

    assert len(d) == size[0]*size[1]*3

    #d = ["{0:g}".format(i) for i in d]
    sys.stderr.write(str(idx+1) + "\r")
    sys.stderr.flush()

    
    #print " ".join(d)
    f.write(struct.pack(fs, *d))

sys.stderr.write("\n")

f.close()
