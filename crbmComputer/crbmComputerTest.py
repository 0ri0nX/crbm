#!/usr/bin/env python
import sys
import crbmComputer as c
from PIL import Image
import numpy as n
import pkg_resources
import time

if len(sys.argv) < 2:
    print "Syntax:", sys.argv[0], "[image 1] ... [image N]"
    exit(1)

instance = c.CRBMComputer([
        pkg_resources.resource_filename(__name__, "testWeights/img-batch2.0.rbm"),
        pkg_resources.resource_filename(__name__, "testWeights/img-batch2.0.transformed.rbm"),
        #"testWeights/img-batch2.0.transformed.transformed.rbm",
        #"testWeights/img-batch2.0.transformed.transformed.transformed.rbm",
        #"testWeights/img-batch2.0.transformed.transformed.transformed.transformed.rbm",
        #"testWeights/img-batch2.0.transformed.transformed.transformed.transformed.transformed.rbm",
        #"testWeights/img-batch2.0.transformed.transformed.transformed.transformed.transformed.transformed.rbm",
        #"testWeights/img-batch2.0.transformed.transformed.transformed.transformed.transformed.transformed.transformed.rbm"
        ], 1)

batch = []
names = []
for i in sys.argv[1:]:
    ss = time.time()
    try:
        x = Image.open(i.strip())
        x = x.resize((200, 200))
        x = x.convert("RGB")
    except:
        continue

    d = n.array(x.getdata())
    d = (d/255.0).flatten().tolist() #normalize data

    batch.append(d)
    names.append(i)

    print "Preparation:", time.time() - ss

    #example of transformation of only one image
    if False:
        z = instance.transform(d)

        print i, z

if len(batch) > 0:
    batchX = n.array(batch).transpose().flatten().tolist()

    #example of batch transformation 
    bb = list(instance.transformBatch(batchX))

    batchY = n.array(bb).reshape(instance.outputNum, len(bb)/instance.outputNum).transpose()
    
    for i, name in enumerate(names):
        print batchY[i], "#", name


