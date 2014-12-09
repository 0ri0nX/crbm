import crbmComputer as c
from PIL import Image
import numpy as n

import time

instance = c.CRBMComputer(["../dataConv/data5-5k-200x200x3.txt.rbm", "../dataConv/data5-5k-200x200x3.txt.transformed.rbm", "../dataConv/data5-5k-200x200x3.txt.transformed.transformed.rbm.zal"], 1)

for i in range(1,10):
    ss = time.time()
    try:
        x=Image.open('/home/orionx/data/imageHash/obrazkyZeSerpu/data2/' + str(i))
    except:
        continue

    x=x.resize((200, 200))

    d=n.array(x.getdata())
    d=(d/255.0).flatten().tolist()

    print "Preparation:", time.time() - ss

    z=instance.transform(d)


