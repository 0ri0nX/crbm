#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys
import crbmComputer as c
import time


def reconstruct(data):#list of data
    #for idx, i in enumerate(data):
    #    zz = instance.reconstruct(i)
    #
    #    print idx, zz[:10] 
    
    print "shape:", np.array(data).transpose().shape

    batchX = np.array(data).transpose().flatten().tolist()
    
    #print batchX
    
    bb=instance.reconstructBatch(batchX)
    bb = list(bb)

    batchY = np.array(bb).reshape(instance.inputNum, len(bb)/instance.inputNum).transpose()
    
    for i in range(len(data)):
        print i, batchY[i, :10]

    return batchY

if len(sys.argv) != 3:
    print sys.argv[0], "<transformed-images> <rbm-file1> ... <rbm-fileN>"
    exit(1)

instance = c.CRBMComputer(["../dataConv/data5-5k-200x200x3.txt.rbm"
    , "../dataConv/data5-5k-200x200x3.txt.transformed.rbm"
    , "../dataConv/data5-5k-200x200x3.txt.transformed.transformed.rbm"
    , "../dataConv/data5-5k-200x200x3.txt.transformed.transformed.transformed.rbm"
    , "../dataConv/data5-5k-200x200x3.txt.transformed.transformed.transformed.transformed.rbm"
    , "../dataConv/data5-5k-200x200x3.txt.transformed.transformed.transformed.transformed.transformed.rbm"
    ], 1)

#instance = c.CRBMComputer(sys.argv[2:], 1)

d = []

for i in range(min(1024, instance.outputNum)):
    z = [0]*instance.outputNum
    z[i] = 1.0

    d.append(z)

m = reconstruct(d)


size =(200, 200)
limit2 = 10
ss = 100
idx = 0
prefix = 'test'

for i in xrange(m.shape[0]):
    if i % (limit2*limit2) == 0:
        if i != 0:
            tar.save(prefix + "-" + str(idx) + ".jpg")
            idx += 1
        tar = im.new("RGB", size = (1+(ss+1)*limit2, 1+(ss+1)*limit2))

    r = m[i]

    mi = np.min(r)
    ma = np.max(r)
    r = np.clip(255.0*(r-mi)/(ma-mi), 0, 255).astype(int)

    r = list(r)
    d = zip(r[0::3], r[1::3], r[2::3])
    
    ii = im.new("RGB", size = size)
    
    ii.putdata(d)
    
    ii = ii.resize((ss, ss))

    x = (i / limit2) % limit2
    y = i % limit2

    tar.paste(ii, ((1+x*(ss+1), 1+y*(ss+1), (x+1)*(ss+1), (y+1)*(ss+1))))

    #ii.save(prefix + "-" + str(i).zfill(2) + ".jpg")
    #ii.show()
    #time.sleep(0.1)

tar.save(prefix + "-" + str(idx) + ".jpg")

