#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys
import struct

size = (200,200)

if len(sys.argv) != 3:
    print sys.argv[0], "<output-file> <batch-limit>"
    print "  Reads image-names from standard input and creates binary batches"
    exit(1)


data = [i.strip() for i in sys.stdin]
limit = int(sys.argv[2])
#data = data[:limit]

VERSION = 1

fidx = 0

fs = "<" + "B"*(size[0]*size[1]*3)

while len(data) >= limit:

    f=open(sys.argv[1] + "." + str(fidx), "wb")
    fInfo=open(sys.argv[1] + "." + str(fidx) + ".info", "w")
    fidx += 1

    f.write("Matrix " + str(VERSION) + "\n")
    f.write(str(limit) + " " + str(size[0]*size[1]*3) + "\n")
    #f.write(struct.pack("<ii", len(data), size[0]*size[1]*3))
    
    
    idx = 0
    used = 0
    for iii in data:
        used += 1
        if idx >= limit:
            break

        try:
            ii = im.open(iii)
        
            ii = ii.resize(size)
        
            ii = ii.convert("RGB")
        
            d = (np.array(ii.getdata())).flatten().tolist()
        except:
            sys.stderr.write("Skipping [" + iii + "]\n")
            sys.stderr.flush()
            continue
    
        assert len(d) == size[0]*size[1]*3
    
        #d = ["{0:g}".format(i) for i in d]
        sys.stderr.write(str(fidx) + " : " + str(idx+1) + " / " + str(limit) + "\r")
        sys.stderr.flush()
    
        
        #print " ".join(d)
        f.write(struct.pack(fs, *d))
        fInfo.write(iii + "\n")

        idx += 1
    
    sys.stderr.write("\n")
    
    f.close()
    fInfo.close()

    data = data[used:]

