#!/usr/bin/env python

from PIL import Image as im
import numpy as np
import sys
import struct

size = (200,200)

if len(sys.argv) not in [3, 4]:
    print sys.argv[0], "<output-file> <images-per-batch> [batch-limit]"
    print "  Reads image-names from standard input and creates binary batches"
    exit(1)


data = [i.strip() for i in sys.stdin]
limit = int(sys.argv[2])
batchLimit = int(sys.argv[3])
#data = data[:limit]

VERSION = 2

fidx = 0

if VERSION == 1:#bytes 0-255
    fs = "<" + "B"*(size[0]*size[1]*3)

if VERSION == 2:#floats (0-255)/255.0
    fs = "<" + "f"*(size[0]*size[1]*3)

#if VERSION == 3:#floats 0-255
#    fs = "<" + "f"*(size[0]*size[1]*3)

idx = 0


def outputImage(f, fInfo, ii, name):
    if ii.size != size:
        #print "si:", ii.size, "ok"
        ii = ii.resize(size, im.ANTIALIAS)

    outputImage1(f, fInfo, ii, name)

#flip
def outputImage1(f, fInfo, ii, name):
    outputImage2(f, fInfo, ii, name)
    outputImage2(f, fInfo, ii.transpose(im.FLIP_LEFT_RIGHT), "LRFlip-"+name)

#noise
def outputImage2(f, fInfo, iiOrig, name):
    #outputImageFinal(f, fInfo, iiOrig, name)

    ii = im.new("RGB", (iiOrig.size))

    for i in range(1):
        loc, scale, r, g, b = np.random.normal(size=(5), loc = 0.0, scale = 32.0)

        #noise = np.random.normal(size=(ii.size[0]*ii.size[1], 3), loc = 64.0*i, scale = 32.0).astype(int)
        noise = np.random.normal(size=(ii.size[0]*ii.size[1], 3), loc = loc, scale = abs(scale))
        noise[:,0] += r
        noise[:,1] += g
        noise[:,2] += b

        noise = noise.astype(int)

        noise += iiOrig.getdata()

        noise[noise < 0] = 0
        noise[noise > 255] = 255

        data = zip(noise[:,0], noise[:,1], noise[:,2])

        ii.putdata(data)

        outputImageFinal(f, fInfo, ii, "noise" + str(i+1) + "-"+name)

def outputImageFinal(f, fInfo, ii, name):
    global idx
    global limit

    if idx >= limit:
        return

    if False: #speed test
        idx += 1
        return

    if ii.size != size:
        print "sf:", ii.size, "bad"
        ii = ii.resize(size, im.ANTIALIAS)

    d = (np.array(ii.getdata())).flatten()
    if VERSION == 2:
        d = d.astype(float)
    d = d.tolist()

    assert len(d) == size[0]*size[1]*3

    f.write(struct.pack(fs, *d))
    fInfo.write(name + "\n")

    idx += 1


while len(data)*10 >= limit and batchLimit > 0:
    batchLimit -= 1

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
        
            ii = ii.convert("RGB")

        except:
            sys.stderr.write("Skipping [" + iii + "]\n")
            sys.stderr.flush()
            continue


        #resized
        if True:
            outputImage(f, fInfo, ii, "resized\t" + iii)

        #9 subimages - deterministic
        if False:
            edge = int(min(ii.size)*0.9)

            for i, iName in zip([0, (ii.size[0]-edge)/2, ii.size[0]-edge], ["L", "C", "R"]):
                for j, jName in zip([0, (ii.size[1]-edge)/2, ii.size[1]-edge], ["T", "C", "B"]):
                    outputImage(f, fInfo, ii.crop((i, j, i+edge, j+edge)), "patch" + iName + jName +"\t" + iii)

        #9 subimages - random
        if True:
            emi = min(ii.size)
            ema = max(ii.size)

            #rescale image to obtaing size which has smaller edge equal to 223 which means that 223*0.9 = 200
            #ema/emi = x/223
            minEdge = 223
            maxEdge = int(float(ema)*223/float(emi))

            if ii.size[0] < ii.size[1]:
                newSize = (minEdge, maxEdge)
            else:
                newSize = (maxEdge, minEdge)
            
            if ii.size != newSize:
                ii = ii.resize(newSize, im.ANTIALIAS)

            edge = 200 #int(min(ii.size)*0.9) because if min edge is 223 then 0.9*223 = 200
            rnd = (ii.size[0] - edge, ii.size[1] - edge)

            for x in range(4):
                i = np.random.randint(rnd[0])
                j = np.random.randint(rnd[1])

                outputImage(f, fInfo, ii.crop((i, j, i+edge, j+edge)), "random-patch" +"\t" + iii)

            #for i, iName in zip([0, (ii.size[0]-edge)/2, ii.size[0]-edge], ["L", "C", "R"]):
            #    for j, jName in zip([0, (ii.size[1]-edge)/2, ii.size[1]-edge], ["T", "C", "B"]):
            #        outputImage(f, fInfo, ii.crop((i, j, i+edge, j+edge)), "patch" + iName + jName +"\t" + iii)
    
        sys.stderr.write(str(fidx) + " : " + str(idx) + " / " + str(limit) + "\r")
        sys.stderr.flush()
    
    sys.stderr.write("\n")
    
    f.close()
    fInfo.close()

    data = data[used:]

