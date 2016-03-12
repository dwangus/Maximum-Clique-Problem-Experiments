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
def MIS(root, neighbors, graph):
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
    return miset
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
def reduction(root, G):
    lower = 0
    while len(nodeweb(G[root], G)) > 0:
        MISAM = MIS(root, nodeweb(G[root], G), G)
        if len(MISAM) > 1:
            lower += 1
        G = merge(MISAM, G)
    return lower
def boundsreduction(G, root):
    lower = reduction(root, G)
    upper = len(G.keys())
    lower = upper - lower
    #print("Inserted all edges for root {0}:".format(root))
    #print("Range[{0}, {1}]".format(upper, lower))
    return((upper, lower))
########################################################################################################################################################################################
def MCP(filename):
    G = graphConv(filename)
    bounds = []
    for v in G.keys():
        H = {}
        for u in G.keys():
            temp = []
            for item in G[u]:
                temp.append(item)
            if u == v:
                for item in G.keys():
                    if item not in temp:
                        if item != v:
                            temp.append(item)
            else:
                if v not in temp:
                    temp.append(v)
            H[u] = temp
        tup = boundsreduction(H, v)
        bounds.append(tup)
    #print(bounds)
    lows = sorted(bounds, key=itemgetter(1), reverse=True)[0][1]
    upps = sorted(bounds, key=itemgetter(0))[0][0]
    print((upps, lows))
    print(float((upps - lows)/2.0 + lows))
    return bounds
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
def MCPnorm(filename):
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
    lows = sorted(norm, key=itemgetter(1), reverse=True)[0][1]
    upps = sorted(norm, key=itemgetter(0))[0][0]
    print((upps, lows))
    print(float((upps - lows)/2.0 + lows))
    return norm
########################################################################################################################################################################################
first = MCP("c125.txt")
first1 = MCP("c250.txt")
'''
second = MCPnorm("c125.txt")
flag = True
for i in range(len(first)):
    if first[i][0] != second[i][0] + 1:
        flag = False
        break
print(flag)
'''































