#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys
import time

#size(200, 200)
size = (10, 10)
target = (8, 8)

if len(sys.argv) != 3:
    print sys.argv[0], "<transformed-images> <rbm-files>"
    print "  Expecting size", size
    exit(1)

f = open(sys.argv[1])
prefix = sys.argv[2]

while True:
    z = f.next()
    if z.startswith("weights"):
        break

m = []
for j in f:
    try:
        r = [float(i) for i in j.split(" ")]
    except:
        break

    m.append(r)

m = np.array(m)
m = m.transpose()

#mi = np.min(m)
#ma = np.max(m)
#m = np.clip(255.0*(m-mi)/(ma-mi), 0, 255).astype(int)

limit = 100
ss = 100
tar = im.new("RGB", size = (1+(ss+1)*target[0], 1+(ss+1)*target[1]))

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

    x = i / target[0]
    y = i % target[0]

    tar.paste(ii, ((1+x*(ss+1), 1+y*(ss+1), (x+1)*(ss+1), (y+1)*(ss+1))))

    #ii.save(prefix + "-" + str(i).zfill(2) + ".jpg")
    #ii.show()
    #time.sleep(0.1)

tar.save(prefix + "-all.jpg")

import crbmComputer as c
from PIL import Image
import numpy as n

import time

instance = c.CRBMComputer(["../dataConv/data5-5k-200x200x3.txt.rbm", "../dataConv/data5-5k-200x200x3.txt.transformed.rbm", "../dataConv/data5-5k-200x200x3.txt.transformed.transformed.rbm", "../dataConv/data5-5k-200x200x3.txt.transformed.transformed.transformed.rbm", "../dataConv/data5-5k-200x200x3.txt.transformed.transformed.transformed.transformed.rbm", "../dataConv/data5-5k-200x200x3.txt.transformed.transformed.transformed.transformed.transformed.rbm"], 1)

batch = []
num = 4
for i in range(1, num+1):
    ss = time.time()
    try:
        x=Image.open('/home/orionx/data/imageHash/obrazkyZeSerpu/data2/' + str(i))
    except:
        continue

    x=x.resize((200, 200))

    d=n.array(x.getdata())
    d=(d/255.0).flatten()
    d = d.tolist()
    batch.append(d)

    print "Preparation:", time.time() - ss

    z = instance.transform(d)

    print i, z[:10] 


batchX = n.array(batch).transpose().flatten().tolist()

print len(batchX)
#print batchX

bb=instance.transformBatch(batchX)
bb = list(bb)

batchY = n.array(bb).reshape(instance.outputNum, len(bb)/instance.outputNum).transpose()

for i in range(0, num):
    print i, batchY[i, :10]
