#!/usr/bin/env python

import sys
import struct

#size(200, 200)
size = (10, 10)
target = (8, 8)

def bin2sparse(filename, filenameInfo, targetFilename):
    "version 2 to version 0"

    #Matrix 2
    #40000 1500

    print filename, "+", filename + ".info ->", targetFilename

    with open(filename, 'rb') as f, open(filenameInfo, 'r') as fInfo, open(targetFilename, 'w') as fOut:
        header = f.readline().strip()
        assert header == "Matrix 2"
        x, y = map(int, f.readline().split(" "))
        
        print "window-size x features:", x, y
        
        fs = "<" + "f"*(y)
        
        dSize = struct.calcsize(fs)

        #fOut.write("Matrix 0\n" + str(x) + " " + str(y) + "\n")
        sigs = map(str, range(1, y+1))

        for i in xrange(x):
            try:
                r = struct.unpack(fs, f.read(dSize))
            except:
                raise
            imgName = fInfo.next().strip()

            sigVals = map(str, r)

            ss = [":".join([a,b]) for a,b in zip(sigs, sigVals)]


            fOut.write(" ".join(ss + ["#", imgName]) + "\n")

            sys.stdout.write(str(i+1) + "\r")
            sys.stdout.flush()

    sys.stdout.write(str(i+1) + "\n")

if len(sys.argv) != 4:
    print sys.argv[0], "<binary-in-file> <binary-in-file-info> <text-out-file>"
    print "  reads version 2 matrix (binary) and writes sparse representation (signal:value)*"

    exit(1)

bin2sparse(sys.argv[1], sys.argv[2], sys.argv[3])
    
