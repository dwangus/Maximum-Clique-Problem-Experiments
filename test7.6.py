from operator import itemgetter
import random

########################################################################################################################################################################################
def graphConv(filename):
    graph = {}
    size = 0
    with open(filename, 'r') as file:
        for line in file:
            contents = line.split()
            if contents[0] == 'p':
                size = int(contents[2])
                break
    for i in range(1, size + 1):
        graph[i] = []
    with open(filename, 'r') as file:
        for line in file:
            edge = line.split()
            if edge[0] == 'e':
                e1 = int(edge[1])
                e2 = int(edge[2])
                graph[e1].append(e2)
                graph[e2].append(e1)
    return graph
########################################################################################################################################################################################
def merge(bunch, graph):
    root = bunch[0]
    for i in range(1, len(bunch)):
        tomerge = bunch[i]
        for friend in graph[tomerge]:
            graph[friend].remove(tomerge)
            if friend not in graph[root]:
                graph[root].append(friend)
                graph[friend].append(root)
        graph.pop(tomerge, None)
    return graph
########################################################################################################################################################################################
def nodeweb(neighbors, graph):
    temp = []
    size = len(graph.keys()) - 1
    for n in neighbors:
        temp.append(n)
    i = 0
    while i < len(temp):
        if len(graph[temp[i]]) == size:
            temp.pop(i)
        else:
            i += 1
    return temp
########################################################################################################################################################################################
def MIS2(root, neighbors, graph):
    miset = []
        
    allnodes = []
    for key in graph.keys():
        allnodes.append(key)
    allnodes.remove(root)
    
    for i in range(len(neighbors)):
        first = neighbors[i]
        if first in allnodes:
            miset.append(first)
            adjacency = graph[first]
            allnodes.remove(first)
            for friend in adjacency:
                if friend in allnodes:
                    allnodes.remove(friend)
                        
    while len(allnodes) > 0:
        first = allnodes.pop(0)
        miset.append(first)
        adjacency = graph[first]
        for friend in adjacency:
            if friend in allnodes:
                allnodes.remove(friend)
    return miset
########################################################################################################################################################################################
def MISroot(root, graph):
    miset = [root]
    disconnected = []
    left = []
    adjroot = graph[root]
    for key in graph.keys():
        if key not in adjroot:
            if key != root:
                disconnected.append(key)
                left.append(key)
    for i in range(len(disconnected)):
        add = disconnected[i]
        if add in left:
            miset.append(add)
            adjadd = graph[add]
            for mutual in adjadd:
                if mutual in left:
                    left.remove(mutual)
    return miset
########################################################################################################################################################################################
def MISleftovers(graph):
    miset = []
    leftovers = []
    size = len(graph.keys()) - 1
    for key in graph.keys():
        if len(graph[key]) < size:
            leftovers.append(key)
    while len(leftovers) > 0:
        first = leftovers.pop(0)
        miset.append(first)
        adjacency = graph[first]
        for friend in adjacency:
            if friend in leftovers:
                leftovers.remove(friend)
    return miset
########################################################################################################################################################################################
def reduction2(root, G):
    lower = 0
    while len(nodeweb(G[root], G)) > 0:
        MISAM = MIS2(root, nodeweb(G[root], G), G)
        if len(MISAM) > 1:
            lower += 1
        G = merge(MISAM, G)
    return lower
def boundsnorm(root, H):
    lower = reduction2(root, H)
    while len(MISroot(root, H)) > 1:
        lower += 1
        H = merge(MISroot(root, H), H)#doesn't run until completion, just until the root's un-connected neighbors are exhausted
    while len(MISleftovers(H)) > 0:
        lower += 1
        H = merge(MISleftovers(H), H)
    upper = len(H.keys())
    lower = upper - lower
    #print("Inserted all edges for root {0}:".format(root))
    #print("Range[{0}, {1}]".format(upper, lower))
    return((upper, lower))
########################################################################################################################################################################################
def upps(filename):
    G = graphConv(filename)
    norm = []
    for vertex in G.keys():
        H = {}
        for key in G.keys():
            temp = []
            for item in G[key]:
                temp.append(item)
            H[key] = temp
        norm.append(boundsnorm(vertex, H))
    #print(norm)
    avg_upps = 0.0
    for item in norm:
        avg_upps += item[0]
    avg_upps = avg_upps/float(len(norm))
    lowest_upper = sorted(norm, key=itemgetter(0))[0][0]
    return (avg_upps, lowest_upper)
def avgMIS(graph):
    keys = graph.keys()
    avg_mis = 0.0
    for item in keys:
        avg_mis += len(MISroot(item, graph))
    avg_mis = avg_mis / float(len(keys))
    return avg_mis
def leftovers(root, filename, test_k):
    G = graphConv(filename)
    i = 0
    x = 0
    while i < test_k:
        if len(nodeweb(G[root], G)) > 0:
            MISAM = MIS2(root, nodeweb(G[root], G), G)
            G = merge(MISAM, G)
        elif len(MISroot(root, G)) > 1:
            MISAM = MISroot(root, G)
            G = merge(MISAM, G)
        elif len(MISleftovers(G)) > 0:
            MISAM = MISleftovers(G)
            G = merge(MISAM, G)
        else:
            x = test_k - i
            break
        i += 1
    left = abs(len(G.keys()) - test_k) + x
    if (len(G.keys()) - test_k) == 0:
        left += len(G.keys())
    return left
def approx(filename):
    tup = upps(filename)
    avg_upps = tup[0]
    bound = tup[1]
    avg_mis = avgMIS(graphConv(filename))
    G = graphConv(filename)
    All_Errs = []
    allnodes = []
    for key in G.keys():
        allnodes.append(key)
    print(bound)
    for test_k in range(1, bound + 1):
        print(test_k)
        random.shuffle(allnodes)
        left_size = 0.0
        for node in allnodes:
            left_size += leftovers(node, filename, test_k)
        left_size /= float(len(G.keys()))
        Expected_Extras = left_size/avg_mis
        Expected_Actual = avg_upps - Expected_Extras
        Rel_Err = abs(Expected_Actual - test_k)/float(test_k)
        All_Errs.append((test_k, Rel_Err))
    print(All_Errs)
    All_Errs = sorted(All_Errs, key=itemgetter(1))
    print(All_Errs[0])
    print(All_Errs[1:int(bound/10)])

approx("c125.txt")










    
