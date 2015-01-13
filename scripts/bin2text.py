#!/usr/bin/env python

import sys
import struct

#size(200, 200)
size = (10, 10)
target = (8, 8)

def bin2txtVer2(filename, targetFilename):
    "version 2 to version 0"

    print "version 2:", filename, "->", targetFilename

    with open(filename, 'rb') as f:
        header = f.readline().strip()

        assert header == "Matrix 2"

        with open(targetFilename, 'w') as fOut:
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

                sys.stdout.write(str(i+1) + "\r")
                sys.stdout.flush()
            sys.stdout.write(str(i+1) + "\n")

def bin2txtVer1(filename, targetFilename):
    "version 1 to version 0"

    print "version 1:", filename, "->", targetFilename

    with open(filename, 'rb') as f:
        header = f.readline().strip()

        assert header == "Matrix 1"

        with open(targetFilename, 'w') as fOut:
            x, y = map(int, f.readline().split(" "))
            
            print "window-size x features:", x, y
            
            fs = "<" + "B"*(y)
            
            dSize = struct.calcsize(fs)

            fOut.write("Matrix 0\n" + str(x) + " " + str(y) + "\n")

            for i in xrange(x):
                try:
                    r = struct.unpack(fs, f.read(dSize))
                except:
                    raise

                r = map(lambda x:x/255.0, r)
                fOut.write(" ".join(map(str, r)) + "\n")

                sys.stdout.write(str(i+1) + "\r")
                sys.stdout.flush()
            sys.stdout.write(str(i+1) + "\n")

if len(sys.argv) != 3:
    print sys.argv[0], "<binary-in-file> <text-out-file>"
    print "  reads version 2 or 1 matrix (binary) and writes Version 0 matrix (textual)"

    exit(1)

filename = sys.argv[1]

with open(filename, 'rb') as f:
    header = f.readline().strip()

if header == "Matrix 1":
    bin2txtVer1(filename, sys.argv[2])

if header == "Matrix 2":
    bin2txtVer2(filename, sys.argv[2])


