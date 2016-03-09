from operator import itemgetter
import random

def complement(filename):
    def conv(filename):
        size = 0
        with open(filename, 'r') as file:
            for line in file:
                contents = line.split()
                if contents[0] == 'p':
                    size = int(contents[2])
                    break
        adjmat = [[0 for x in range(size)] for x in range(size)]
        with open(filename, 'r') as file:
            for line in file:
                edge = line.split()
                if edge[0] == 'e':
                    e1 = int(edge[1]) - 1
                    e2 = int(edge[2]) - 1
                    adjmat[e1][e2] = 1
                    adjmat[e2][e1] = 1
        return adjmat
    G = conv(filename)
    for i in range(len(G[0])):
        for j in range(len(G[0])):
            if i != j:
                if G[i][j] == 0:
                    G[i][j] = 1
                else:
                    G[i][j] = 0
    return G

def complementConv(filename):
    G = complement(filename)
    graph = {}
    for i in range(1, len(G) + 1):
        graph[i] = []
    for i in range(1, len(G)):
        for j in range(0, i):
            if G[i][j] == 1:
                graph[i+1].append(j + 1)
                graph[j+1].append(i + 1)
    #print(graph)
    return graph

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

def MCP(G, shuff = False):
    H = []
    for key in G.keys():
        H.append(key)
    if shuff:
        random.shuffle(H)
    i = 0
    while i != len(H) - 1:
        node1 = H[i]
        outerflag = False
        for y in range(len(H)):
            if y != i:
                node2 = H[y]
                if node2 not in G[node1]:
                    flag = False
                    for shared in G[node1]:
                        if shared in G[node2]:
                            flag = True
                            break
                    if flag:
                        for item in G[node2]:
                            G[item].remove(node2)
                            if item not in G[node1]:
                                G[node1].append(item)
                                G[item].append(node1)
                        H.remove(node2)
                        G.pop(node2, None)
                        outerflag = True
                        break
        if outerflag:
            i = 0 
        else:
            i += 1
    #print(len(H))
    return H

def a(filename):
    return MCP(complementConv(filename))

def neighborhoodCom(filename):
    upbounds = []
    G = complementConv(filename)
    for key in G.keys():
        newgraph = {}
        neighbors = G[key]
        for i in range(len(neighbors)):
            newgraph[neighbors[i]] = []
        for i in range(len(neighbors) - 1):
            A = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                B = neighbors[j]
                if B in G[A]:
                    newgraph[A].append(B)
                    newgraph[B].append(A)
        upbounds.append(newgraph)
    return upbounds

def c(filename):
    graphs = neighborhoodCom(filename)
    upbounds = []
    for graph in graphs:
        upbounds.append(len(MCP(graph)))
    print(max(upbounds))
    print(upbounds)
    return upbounds

#print(graphConv("c125.txt"))
#a("c125.txt")   #which would be the calculated max-independent set of c125.txt
                 #...but how accurate would that be... how much does it overshoot?

c("c125.txt")

















