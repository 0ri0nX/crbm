#!/usr/bin/env python

import sys
import struct

#size(200, 200)
size = (10, 10)
target = (8, 8)

def bin2txt(filename, targetFilename):
    "version 2 to version 0"

    #Matrix 2
    #40000 1500

    print filename, "->", targetFilename

    with open(filename, 'rb') as f, open(targetFilename, 'w') as fOut:
        header = f.readline().strip()
        assert header == "Matrix 2"
        x, y = map(int, f.readline().split(" "))
        
        print "window-size x features:", x, y
        
        fs = "<" + "f"*(y)
        
        dSize = struct.calcsize(fs)

        fOut.write("Matrix 0\n" + str(x) + " " + str(y) + "\n")

        for i in xrange(x):
            try:
                r = struct.unpack(fs, f.read(dSize))
            except:
                raise
            fOut.write(" ".join(map(str, r)) + "\n")

            sys.stdout.write(str(i) + "\r")
            sys.stdout.flush()
    sys.stdout.write(str(i) + "\n")

if len(sys.argv) != 3:
    print sys.argv[0], "<binary-in-file> <text-out-file>"
    print "  reads version 2 matrix (binary) and writes Version 0 matrix (textual)"

    exit(1)

bin2txt(sys.argv[1], sys.argv[2])
    
