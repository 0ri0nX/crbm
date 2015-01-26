#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys
import time
import struct

size = (200, 200)
start = 1
stop = 1
#size = (10, 10)
#size = (39, 39)

if len(sys.argv) not in [2,3,4]:
    print sys.argv[0], "<image-matrix> [index=" + str(stop) + "]"
    print sys.argv[0], "<image-matrix> <start> <stop>"
    print "\texpecting image-size", size
    exit(1)

if len(sys.argv) == 3:
    stop = int(sys.argv[2])
    start = stop

if len(sys.argv) == 4:
    start = int(sys.argv[2])
    stop = int(sys.argv[3])


f = open(sys.argv[1])

#read header
header = f.readline()
if header.startswith("Matrix"):
    version = int(header.strip().split(" ")[1])
else:
    version = 0

print "Version:", version

if version in [1, 2, 3]:
    sizeInfo = f.readline()
else:
    sizeInfo = header

x, y = map(int, sizeInfo.strip().split(" "))
print "size:", x, "x", y

if version == 3:
    data = f.next()
    data = data.strip().split(" ", 1)[1]
    print "loading referenced file:", data
    f.close()
    f = open(data)
else:
    data = f

assert start > 0 and stop <= x

if version == 0:
    for jj in xrange(start-1):
        f.next()
elif version in [2, 3]:
    fs = "<" + "f"*(y)
    
    dSize = struct.calcsize(fs)

    f.seek((start-1)*dSize, 1)

    #for i in xrange(start-1):
    #    try:
    #        r = struct.unpack(fs, f.read(dSize))
    #    except:
    #        raise
else:
    raise Exception("Version " + str(version) + "not implemented")

for jj in xrange(stop-start+1):
    if version == 0:
        j = f.next()
        r = [float(i)*255 for i in j.split(" ")]
    elif version in [2, 3]:
        try:
            r = struct.unpack(fs, f.read(dSize))
        except:
            raise
        r = [i*255 for i in r]

    r = np.clip(r, 0, 255)
    r = list(np.array(r).astype(int))
    r = np.clip(r, 0, 255)
    
    d = zip(r[0::3], r[1::3], r[2::3])

    try:
    
        ii = im.new("RGB", size = size)
    
        ii.putdata(d)

        if size[0] < 50:
            ii = ii.resize((size[0]*5, size[0]*5))

        ii.show()
        time.sleep(0.3)
    except:
        print "bad", d
        raise

