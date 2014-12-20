#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys
import time
import struct
import math
import colorsys

#size(200, 200)
size = (10, 10)
target = (8, 8)

def makeImage(filename, targetFilename):
    print filename, "->", targetFilename

    f = open(filename, 'rb')
    
    while True:
        z = f.readline()
        if z.startswith("weights"):
            z = f.readline()
            x, y = [int(i) for i in z.strip().split(" ")]
            break
    
    print "window-size x features:", x, y
    
    assert x == size[0]*size[1]*3
    
    fs = "<" + "f"*(y)
    
    dSize = struct.calcsize(fs)
    
    m = []
    for i in xrange(x):
        try:
            r = struct.unpack(fs, f.read(dSize))
            #print r[:10]
            #exit(1)
            #r = [float(i) for i in j.split(" ")]
        except:
            raise
    
        m.append(r)
    
    #print "[", f.readline(), "]"
    #print "[", f.readline(), "]"
    
    m = np.array(m)
    print m.shape
    m = m.transpose()
    
    #mi = np.min(m)
    #ma = np.max(m)
    #m = np.clip(255.0*(m-mi)/(ma-mi), 0, 255).astype(int)
    
    xx=int(math.ceil(math.sqrt(y)))
    print "grid:", xx
    target=(xx, xx)
    
    limit = 1000
    ss = 100
    tar = im.new("RGB", size = (1+(ss+1)*target[0], 1+(ss+1)*target[1]))
    
    mi = 0.0
    ma = 1.0
    #rescale = False
    rescale = True

    def getH(d):
        r = d[0::3]
        g = d[1::3]
        b = d[2::3]
    
        Hdat = []
        Sdat = []
        Vdat = [] 
        for rd,gn,bl in zip(r,g,b) :
            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        
        return np.median(Hdat)

    order = []
    for i in xrange(min(limit, m.shape[0])):
        r = m[i]
    
        if rescale:
            mi = np.min(r)
            ma = np.max(r)
            #print ma, mi

        r = np.clip(255.0*(r-mi)/(ma-mi), 0, 255).astype(int)

        order.append((getH(r), i))

    order = list(sorted(order, key = lambda x: x[0]))

    for i in xrange(min(limit, m.shape[0])):
   
        if False:
            r = m[order[i][1]]
        else:
            r = m[i]
    
        if rescale:
            mi = np.min(r)
            ma = np.max(r)
            #print ma, mi
    
        r = np.clip(255.0*(r-mi)/(ma-mi), 0, 255).astype(int)
    
        #print r[:6]
    
        r = list(r)
        d = zip(r[0::3], r[1::3], r[2::3])
        
        ii = im.new("RGB", size = size)
        
        ii.putdata(d)
        
        ii = ii.resize((100, 100))
    
        x = i / target[0]
        y = i % target[0]
    
        tar.paste(ii, ((1+x*(ss+1), 1+y*(ss+1), (x+1)*(ss+1), (y+1)*(ss+1))))
    
    tar.save(targetFilename)

if len(sys.argv) < 3:
    print sys.argv[0], "<prefix> <rbm-file1 ... n>"
    print "  Expecting size", size
    
for i in sys.argv[2:]:
    try:
        n = int(i.strip().rsplit(".", 1)[1])
    except:
        n = 0

    fn = sys.argv[1] + "." + str(n).zfill(5) + ".jpg"

    try:
        makeImage(i.strip(), fn)
    except:
        print "error"
        raise


