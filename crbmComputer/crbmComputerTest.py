print "Initializing ..."
import rbmComputerInstance as r
print "... done"

for i in xrange(20):
    d = [0]*4096
    d[i] = 1

    z = r.transform(d)

    z = list(z)

    print z[:10]
