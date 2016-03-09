import random
from operator import itemgetter

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
#"I will only merge with you if
    #a) Original graph is not disconnected
    #   (which is assumed)
    #b) One of my neighbors is connected to you
'''
def MCP2(G):
    neigh = {}
    for l in range(1, len(G.keys()) + 1):
        neigh[l] = [l]
    H = []
    for key in G.keys():
        H.append(key)
    #i = 0
    i = len(H) - 1
    #while i != len(H) - 1:
    while i != 0:
        #print(i)
        node1 = H[i]
        #print(G[node1])
        outerflag = False #if a "fold" has occurred
        #for y in range(len(H)):
        for y in range(len(H)-1, -1, -1):
            if y != i:
                node2 = H[y]
                #print("--{0}".format(node2))
                if node2 not in G[node1]: #if these two nodes don't share an edge
                    flag = False
                    for shared in G[node1]:
                        if shared in G[node2]:#do they both have a mutual neighbor?
                            flag = True
                            break
                    if flag:#if yes,
                        #print("Removing {1}, merged with {0}".format(node1, node2))
                        #print(G[node2])
                        neigh[node1].append(node2)
                        neigh[node2].append(node1)
                        for item in G[node2]: #fold/merge operation
                            G[item].remove(node2)#remove node 2 from all neighbors' lists
                            if item not in G[node1]:
                                G[node1].append(item)#update all of node 2's exclusive neighbors
                                G[item].append(node1)#with node 1 and vice-versa
                        H.remove(node2)#remove all references of node 2
                        G.pop(node2, None)
                        outerflag = True #graph fundamentally changed
                        break
        if outerflag:
            #i = 0 #start all over
            i = len(H) - 1
        else:
            #i += 1
            i -= 1
    print(len(H))
    neighbor = []
    for key in neigh:
        if key in H:
            #print("{0}, {1}".format(key, neigh[key]))
            neighbor.append(neigh[key])
    return neighbor

def MCP(G):
    neigh = {}
    for l in range(1, len(G.keys()) + 1):
        neigh[l] = [l]
    H = []
    for key in G.keys():
        H.append(key)
    i = 0
    #i = len(H) - 1
    while i != len(H) - 1:
    #while i != 0:
        #print(i)
        node1 = H[i]
        #print(G[node1])
        outerflag = False #if a "fold" has occurred
        for y in range(len(H)):
        #for y in range(len(H)-1, -1, -1):
            if y != i:
                node2 = H[y]
                #print("--{0}".format(node2))
                if node2 not in G[node1]: #if these two nodes don't share an edge
                    flag = False
                    for shared in G[node1]:
                        if shared in G[node2]:#do they both have a mutual neighbor?
                            flag = True
                            break
                    if flag:#if yes,
                        #print("Removing {1}, merged with {0}".format(node1, node2))
                        #print(G[node2])
                        neigh[node1].append(node2)
                        neigh[node2].append(node1)
                        for item in G[node2]: #fold/merge operation
                            G[item].remove(node2)#remove node 2 from all neighbors' lists
                            if item not in G[node1]:
                                G[node1].append(item)#update all of node 2's exclusive neighbors
                                G[item].append(node1)#with node 1 and vice-versa
                        H.remove(node2)#remove all references of node 2
                        G.pop(node2, None)
                        outerflag = True #graph fundamentally changed
                        break
        if outerflag:
            i = 0 #start all over
            #i = len(H) - 1
        else:
            i += 1
            #i -= 1
    #for key in G:
    #    print("{0}, {1}".format(key, G[key]))
    print(len(H))
    neighbor = []
    for key in neigh:
        if key in H:
            #print("{0}, {1}".format(key, neigh[key]))
            neighbor.append(neigh[key])
    return neighbor
'''
'''
#print(MCP(graphConv("c125.txt")))
def group(filename):
    grouping = [1]
    neigh1 = MCP(graphConv(filename))
    neigh2 = MCP2(graphConv(filename))
    recurs(1, neigh1, neigh2, grouping, True)
    print(sorted(grouping))
    print(len(grouping))
    return grouping
    
def recurs(search, neigh1, neigh2, bunch, switch = True):
    if switch:
        for ele in neigh1:
            if search in ele:
                for num in ele:
                    if num not in bunch:
                        bunch.append(num)
                        recurs(num, neigh1, neigh2, bunch, switch = False)
                return
    else:
        for ele in neigh2:
            if search in ele:
                for num in ele:
                    if num not in bunch:
                        bunch.append(num)
                        recurs(num, neigh1, neigh2, bunch, switch = True)
                return
'''
#MCP(graphConv("c125.txt"))
#MCP2(graphConv("c125.txt"))
#group("c125.txt")
#graph = graphConv("c125.txt")
#for key in graph.keys():
#    print("{0}, {1}".format(key, graph[key]))

#MCP(graphConv("frb30-15-1.txt"))
#MCP2(graphConv("frb30-15-1.txt"))
'''
MCP(graphConv("c250.txt"))
MCP2(graphConv("c250.txt"))
MCP(graphConv("c500.txt"))
MCP2(graphConv("c500.txt"))
'''
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

def neighborhood(filename):
    upbounds = []
    G = graphConv(filename)
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
    
def a(filename):
    return MCP(graphConv(filename))
def b(filename):
    return MCP(graphConv(filename), True)
def c(filename):
    graphs = neighborhood(filename)
    upbounds = []
    for graph in graphs:
        upbounds.append(len(MCP(graph)))
    print(min(upbounds))
    print(upbounds)
    return upbounds

#a("frb30-15-1.txt")    #30 vs. 30 // 450 nodes, 83198 edges
#a("frb30-17-1.txt")    #35 vs. 35 // 595 nodes, 148859 edges
#a("c125.txt")          #57 vs. 34 // 125 nodes, 6963 edges
#a("c250.txt")          #98 vs. 44 // 250 nodes, 27984 edges
#a("c500.txt")          #184 vs. >= 57 // 500 nodes, 112332 edges
#a("mann_a27.txt")      #135 vs. 126 // 378 nodes, 70551 edges
#a("brock200_2.txt")    #36 vs. 12 // 200 nodes, 9876 edges
#a("hamming8-4.txt")    #32 vs. 16 // 256 nodes, 20864 edges
#a("gen200_p9_44.txt")  #76 vs. 44 // 200 nodes, 17910 edges

#b("frb30-15-1.txt")    #87 vs. 30
#b("frb30-17-1.txt")    #114 vs. 35
#b("c125.txt")          #57 vs. 34
#b("c250.txt")          #100 vs. 44
#b("c500.txt")          #177 vs. >= 57
#b("mann_a27.txt")      #144 vs. 126
#b("brock200_2.txt")    #34 vs. 12
#b("hamming8-4.txt")    #31 vs. 16
#b("gen200_p9_44.txt")  #75 vs. 44

#c("frb30-15-1.txt")    #29 vs. 30 -- that's wr-...no, that's right haha
#c("frb30-17-1.txt")    # vs. 35
#c("c125.txt")          #47 vs. 34
#c("c250.txt")          #81 vs. 44
#c("c500.txt")          # vs. >= 57
#c("mann_a27.txt")      # vs. 126
#c("brock200_2.txt")    #17 vs. 12
#c("hamming8-4.txt")    #26 vs. 16
#c("gen200_p9_44.txt")  #61 vs. 44
