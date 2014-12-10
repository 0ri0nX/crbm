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
