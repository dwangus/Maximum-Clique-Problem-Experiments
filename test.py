from random import shuffle

def makeWorstCaseGraph(k,c):
    kcopy = k
    ksec = kcopy - 1
    n = (ksec)*c + kcopy
    nodes = [None]*6
    nodes[0] = {}   # Unique connections table                                                  -- dict of (node-name-keys to dict-values of (neighbor-name-keys to binary-connection-values))
    nodes[1] = []   # Names of all nodes (keys to access their corresponding dicts)             -- array of all node-names
    nodes[2] = {}   # For each node, a list of its actual neighbors                             -- dict of node's arrays of neighbor-names
    nodes[3] = []   # Deleted-edges list                                                        -- array of tuples of edges of node-endpoints
    nodes[4] = {}   # Unique counter for every node -- after all, in order for a node to be     -- dict of node's ints
                    #   considered part of a solution, it needs to have a quantifiable
                    #   identifier
    nodes[5] = []   # Duplicate of [1], to preserve original order, if [1] shuffled or changed  -- array of all node-names in original order
    table = nodes[0]
    neighbors = nodes[2]
    countneigh = 0
    counter = 0
    for i in range(n):
        if i < (n-kcopy):
            group = 1+(int)(i/(ksec))
            modi = i%(ksec) + 1
            name = str(group)+"."+str(modi)
            neighbors[name] = []
        else:
            countneigh += 1
            name = "k."+str(countneigh)
            neighbors[name] = []
    for i in range(n):
        if i < (n - kcopy):
            group = 1+(int)(i/(ksec))
            modi = i%(ksec) + 1
            name = str(group)+"."+str(modi)
            table[name] = {}
            cEnv = table[name]
            for j in range(i):
                groupj = 1+(int)(j/(ksec))
                modj = j%(ksec) + 1
                namej = str(groupj)+"."+str(modj)
                if modj != modi:
                    cEnv[namej] = 1
                    neighbors[name].append(namej)
                    neighbors[namej].append(name)
                else:
                    cEnv[namej] = 0
                    nodes[3].append((name,namej))
        else:
            counter += 1
            name = "k."+str(counter)
            table[name] = {}
            kEnv = table[name]
            counterk = 0
            for k in range(counter-1):
                counterk += 1
                namek = "k."+str(counterk)
                kEnv[namek] = 1
                neighbors[name].append(namek)
                neighbors[namek].append(name)
            for c in range(n-kcopy):
                group = 1+(int)(c/(ksec))
                modc = c%(ksec) + 1
                namec = str(group)+"."+str(modc)
                kEnv[namec] = 0
                nodes[3].append((name,namec))
    counter2 = 0
    for i in range(n):
        if i < (n-kcopy):
            group = 1+(int)(i/(ksec))
            mod = i%(ksec) + 1
            name = str(group)+"."+str(mod)
            nodes[1].append(name)
            nodes[5].append(name)
            nodes[4][name] = 0
        else:
            counter2 += 1
            name = "k."+str(counter2)
            nodes[1].append(name)
            nodes[5].append(name)
            nodes[4][name] = 0

    return nodes

def allIntersections(arbitraryListOrderArray):
    g = arbitraryListOrderArray
    inter = []
    for first in range(len(g) - 3):
        fir = g[first]
        for second in range(first + 1, len(g)- 2):
            sec = g[second]
            for third in range(second + 1, len(g) - 1):
                thir = g[third]
                for fourth in range(third + 1, len(g)):
                    four = g[fourth]
                    inter.append([fir,sec,thir,four,0,((fir,thir),(sec,four))])
    return inter            # -- intersection is an array of 4-element arrays

#for this one function, I would have to pay a price of n^4 most likely,
#to create a symmetricizing algorithm
def symmetricize(worstCaseList,k,c):
    g = worstCaseList
    kcopy = k
    ksec = kcopy - 1
    n = (ksec)*c + kcopy
    new = float(n)/float(kcopy)
    counter = 0
    for i in range((ksec)*c,len(g)):
        newindex = int(round((counter*new),0))
        counter += 1
        g.insert(newindex, g.pop(i))
    return g

#Alg1 = Backwards edge deletion + named intersections increment + summed to nodes
def testAlg1(k,c):
    #(6,5) takes about 15 seconds
    #(7,5) takes like 40 seconds
    #(8,5) takes almost like 2.5 minutes shit
    print("Full Intersections Only")
    kcopy = k
    ksec = kcopy - 1
    n = (ksec)*c + kcopy
    nodes = makeWorstCaseGraph(k,c)
    uniqueEdges = nodes[0]
    allNodes = nodes[1]
    #shuffle(allNodes)
    #print("Shuffled Order: ")
    #print(allNodes)
    symmetricize(allNodes,k,c)
    print("k: %d // c: %d // ksec: %d") %(k,c,ksec)
    print("Symmetricized Order: ")
    print(allNodes)
    neighbors = nodes[2]
    deleted = nodes[3]
    nodeCounter = nodes[4]
    ordered = nodes[5]
    #for i in range(len(nodes)):
        #print("%d is printing.") % (i)
        #print(nodes[i])
    intersect = allIntersections(allNodes)
    #print(intersect)
    '''
    #testing one node of c against one node of k -- first and last of array allNodes
    cMem = allNodes[0]
    kMem = allNodes[-1]
    cVals = [None]*4
    cVals[0] = 0
    cVals[1] = []
    cVals[2] = 0
    cVals[3] = 0
    kVals = [None]*4
    kVals[0] = 0
    kVals[1] = []
    kVals[2] = 0
    kVals[3] = 0'''
    for i in range(len(deleted)):
        edge = deleted[i]
        e1 = edge[0]
        e2 = edge[1]
        for y in range(len(intersect)):
            x = intersect[y]
            if (e1 not in x) and (e2 not in x):
                value = x[4]
                x[4] = value + 1
        '''
        #throw-away code:
        if (e1 != cMem) and (e2 != cMem):
            cVals[0] += 1
            last = nodeCounter[0]
            neighborNames = neighbors[cMem]
            for nn in range(len(neighborNames)):
                c2 = neighborNames[nn]
                for y in range(len(intersect)):
                    x = intersect[y]
                    if (cMem in x) and (c2 in x):
                        nodeCounter[0] += 1
            value = nodeCounter[0] - last
            if (e1 in allNodes[(ksec)*c:]) or (e2 in allNodes[(ksec)*c:]):
                cVals[1].append(("k-c",value))
                cVals[2] += 1
            else:
                cVals[1].append(("c-c",value))
                cVals[3] += 1
        if (e1 != kMem) and (e2 != kMem):
            kVals[0] += 1
            last = nodeCounter[-1]
            neighborNames = neighbors[kMem]
            for nn in range(len(neighborNames)):
                k2 = neighborNames[nn]
                for y in range(len(intersect)):
                    x = intersect[y]
                    if (kMem in x) and (k2 in x):
                        nodeCounter[-1] += 1
            value = nodeCounter[-1] - last
            if (e1 in allNodes[(ksec)*c:]) or (e2 in allNodes[(ksec)*c:]):
                kVals[1].append(("k-c",value))
                kVals[2] += 1
            else:
                kVals[1].append(("c-c",value))
                kVals[3] += 1
    print(cVals)
    print("\n")
    print(kVals)#'''
    #'''    
    #print(intersect)
    #if len(deleted) == (ksec)*(k)*(c) + (c*(c-1)*(ksec)/2):
    #    print(True)
    for i in range(len(allNodes)):
        e1 = allNodes[i]
        neighborNames = neighbors[e1]
        for nn in range(len(neighborNames)):
            e2 = neighborNames[nn]
            for y in range(len(intersect)):
                x = intersect[y]
                pairs = x[5]
                if (e1 in pairs[0] and e2 in pairs[0]):
                    #commented-out code is to check for ONLY full intersections;
                    #w/o comments, checks for both full and half intersections
                    e3 = pairs[1][0]
                    e4 = pairs[1][1]
                    if (e4 in neighbors[e3]):
                        current = nodeCounter[e1]
                        nodeCounter[e1] = current + x[4]
                    #current = nodeCounter[i]
                    #nodeCounter[i] = current + x[4]
                elif (e1 in pairs[1] and e2 in pairs[1]):
                    e3 = pairs[0][0]
                    e4 = pairs[0][1]
                    if (e4 in neighbors[e3]):
                        current = nodeCounter[e1]
                        nodeCounter[e1] = current + x[4]
                    #current = nodeCounter[i]
                    #nodeCounter[i] = current + x[4]
    for i in range(len(allNodes)):
        leftovers = n - kcopy
        numcount = nodeCounter[ordered[i]]
        edgecount = len(neighbors[ordered[i]])
        if i < leftovers:
            print("C-Member %s --> %d, Ratio: %d") %(ordered[i],numcount,int(numcount/edgecount))
        else:
            print("K-Member %s --> %d, Ratio: %d") %(ordered[i],numcount,int(numcount/edgecount))#'''
'''
#Alg2 = basically just Alg1 but with counting half-connections
def testAlg2(k,c):
    print("Half + Full")
    kcopy = k
    ksec = kcopy - 1
    n = (ksec)*c + kcopy
    nodes = makeWorstCaseGraph(k,c)
    uniqueEdges = nodes[0]
    allNodes = nodes[1]
    #shuffle(allNodes)
    symmetricize(allNodes,k,c)
    print("k: %d // c: %d // ksec: %d") %(k,c,ksec)
    print("Symmetricized Order: ")
    print(allNodes)
    neighbors = nodes[2]
    deleted = nodes[3]
    nodeCounter = nodes[4]
    ordered = nodes[5]
    intersect = allIntersections(nodes[1])
    for i in range(len(deleted)):
        edge = deleted[i]
        e1 = edge[0]
        e2 = edge[1]
        for y in range(len(intersect)):
            x = intersect[y]
            if (e1 not in x) and (e2 not in x):
                value = x[4]
                x[4] = value + 1
    for i in range(len(allNodes)):
        e1 = allNodes[i]
        neighborNames = neighbors[e1]
        for nn in range(len(neighborNames)):
            e2 = neighborNames[nn]
            for y in range(len(intersect)):
                x = intersect[y]
                pairs = x[5]
                if (e1 in pairs[0] and e2 in pairs[0]):
                    current = nodeCounter[e1]
                    nodeCounter[e1] = current + x[4]
                elif (e1 in pairs[1] and e2 in pairs[1]):
                    current = nodeCounter[e1]
                    nodeCounter[e1] = current + x[4]
    for i in range(len(allNodes)):
        leftovers = n - kcopy
        numcount = nodeCounter[ordered[i]]
        edgecount = len(neighbors[ordered[i]])
        if i < leftovers:
            print("C-Member %s --> %d, Ratio: %d") %(ordered[i],numcount,int(numcount/edgecount))
        else:
            print("K-Member %s --> %d, Ratio: %d") %(ordered[i],numcount,int(numcount/edgecount))
#'''
testAlg1(6,4)
#testAlg2(6,4)

'''
Notes:
    - First, I need to analyze how big this gets -- AKA, how many times each are added to a node of k vs. a node of c
    - ...Second, if my hunch is right, I can recursively screen out the lowest integers from the bottom up by
            proportionalizing out worst-case scenarios for a given k by their # of neighbors
    - Third, I need to think up and implement an actual symmetricizing algorithm...
    - Fourth: consider, too, what if the k clique has many members each, that each take away from itself when proportionalizing?
            (as well, the difficulty of singling out a single node rather than an entire clique's members as a whole)
    - Fifth: It's also worthwhile looking into optimizations of my approach, if it works -- if the maximum number of edges any node has
            is some number significantly less than n, say 39 vs. 4000, then looking at subgraphs of nodes' neighbors and doing the algorithm would be worthwhile --> 4000*39^11 = 10^21 vs. 4000^11 = 10^39

    - # Intersections in graph of n-clique = Summation from 1 to n-4 of T(n) (tetrahedral number) = C(n,4)
    - If this indeed does work... wow, what a chimera of approaches. -->
            Proportionality of symmetrical, non-existent abstract edge intersections in circular relative ordering, while playing a numbers game comparing between integers... between successive reductionist rounds on the graph.
    - I seriously need to figure out this ratio thing... if it can ever be concluded that ratio of edge to edge intersections based off deleted edges can determine SOMETHING, no matter the case
        - Some cases to explore:
            - How a member of the isolated k-clique, if it is more connected with irrelevant nodes, what that does to its proportion
            - How multiple members of the isolated k-clique behave if above (but never more than k-1 members of the k-clique)
            - How multiple k-cliques interconnected affects their proportion
            - How isolated members of lower-order cliques affect the k-clique's proportion
            - How isolated members of lower-order cliques (less than k-1), once connected to members of the k-clique, affect its proportion

Other Approaches to Try:
    - ***I need to figure out ways of gaining concrete information about a graph, that can apply to all graphs
    - Analyze growth
    - Analyze a node's composition of cliques
        - 7 --> 1111111, 111112, 11113, 1114, 115, 16, 7 // 11122, 1123, 124, 25 // 1222, 223, 133, 34 --> 15 permutations
        - Ehhh...   https://www.physicsforums.com/threads/in-how-many-ways-can-i-write-n-as-a-sum-of-integers.437047/
                    https://en.wikipedia.org/wiki/Stars_and_bars_%28combinatorics%29
                    http://oeis.org/A000041
                    https://www.physicsforums.com/attachments/integer-partition-table-png.29493/
    - Analyze ratio of edge to intersections
    - 3-box shuffling
    - Degree of connection consumptions?
    - 2 k-2 vs. 1 k -->
        - So as k gets bigger, even with 3 k-3, it still outpaces k. However, the proportion of edge to edge intersections remains tight nonetheless.
'''





























