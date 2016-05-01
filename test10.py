import time
import ripple as rp
import test8_4_numpy as pref

file1 = "c125.txt"
fileName = file1

def test10(filename, iterations, RR = False):
    start = time.time()
    ordering = pref.main(filename, iterations)
    #ordering, weights = pref.main(filename, iterations)
    
    base = []
    OG = rp.graphConv(filename)
    OGSize = len(OG.keys())
    if RR:
        print("Reverse Ripple")
        base.append(rp.reverseRippling(ordering, iterations, OGSize)[0][0])
        #base.append(rp.reverseRippling(ordering, iterations, OGSize, weights)[0][0])
    else:
        print("Ripple")
        base.append(rp.rippling(ordering, iterations, OGSize)[0][0])
        #base.append(rp.rippling(ordering, iterations, OGSize, weights)[0][0])

    for iter in range(len(OG.keys())):
        print(base)
        tempG = rp.makeGraphCopy(OG)
        tempG = pref.closedCliqueNeighborhood(base, tempG)
        if len(tempG.keys()) == 0:
            break
        ordering = pref.testSample(tempG, iterations, True, OGSize)
        #ordering, weights = pref.testSample(tempG, iterations, True, OGSize)
        if RR:
            base.append(rp.reverseRippling(ordering, iterations, OGSize)[0][0])
            #base.append(rp.reverseRippling(ordering, iterations, OGSize, weights)[0][0])
        else:
            base.append(rp.rippling(ordering, iterations, OGSize)[0][0])
            #base.append(rp.rippling(ordering, iterations, OGSize, weights)[0][0])

    if pref.checkClique(base, OG):
        print("Size of Return_Set: {0}".format(len(base)))
        print("Run-Time: {0}".format(time.time() - start))
        return base
    else:
        print("Something went wrong")
        print("Run-Time: {0}".format(time.time() - start))
        return base

test10(fileName, 100)
#clique = [114, 104, 45, 54, 80, 7, 29, 60, 40, 110, 99, 34, 1, 117, 44, 68, 31, 9, 123, 52, 122, 49, 11, 96, 103, 125, 50, 70, 5, 77, 55, 25]
#clique = [114, 104, 45, 54, 80, 7, 29, 60, 40, 110, 99, 34, 1, 117, 44, 68, 31, 9, 123, 52, 122, 49, 11, 96, 103, 125, 5, 70, 77, 50, 55, 65] --> 25 or 35
#graph = rp.graphConv(fileName)
#print(pref.closedCliqueNeighborhood(clique, graph))
#print(pref.testSample(pref.closedCliqueNeighborhood(clique, graph), 100, True, 125))
        
