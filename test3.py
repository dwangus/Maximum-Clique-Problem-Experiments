from operator import itemgetter

def graphConv(filename):
    graph = {}
    #'''
    size = 0
    with open(filename, 'r') as file:
        for line in file:
            contents = line.split()
            if contents[0] == 'p':
                size = int(contents[2])
                break
    #'''
    for i in range(1, size + 1):
        RT = {}
        for j in range(1, size + 1):
            if j != i:
                RT[j] = 0
        graph[i] = [[], RT]
    with open(filename, 'r') as file:
        for line in file:
            edge = line.split()
            if edge[0] == 'e':
                graph[int(edge[1])][0].append(int(edge[2]))
                graph[int(edge[2])][0].append(int(edge[1]))
    return graph #as a... dict of node-name : [[adj-list], RT-dict {all nodes initialized to 0}]
'''
def unpack(filename):
    POHNS = []
    size = 0
    with open(filename, 'r') as file:
        for line in file:
            contents = line.split()
            if contents[0] == 'p':
                size = int(contents[2])
                break
            
    for i in range(1, size + 1):
        POHNS.append([i,0])
        
    graph = graphConv(filename, size)

    for key in graph.keys():
        newgraph = {}
        adj = graph[key][0]
        for i in range(len(adj)):
            RT = {}
            for j in range(len(adj)):
                if j != i:
                    RT[adj[j]] = 0
            newgraph[adj[i]] = [[], RT]
        with open(filename, 'r') as file:
            for line in file:
                edge = line.split()
                if edge[0] == 'e':
                    e1 = int(edge[1])
                    e2 = int(edge[2])
                    if e1 in adj and e2 in adj:
                        newgraph[e1][0].append(e2)
                        newgraph[e2][0].append(e1)
        pohn = MCP(newgraph) + 1
        for x in range(len(POHNS)):
            if POHNS[x][0] == key:
                POHNS[x][1] = pohn
                break
        print("{0}, {1}".format(key, pohn))
    POHNS = sorted(POHNS, key=itemgetter(1), reverse=True)
    toplen = POHNS[0][1]
    topportion = POHNS[:toplen]
    portion = []
    truth = []
    for y in range(len(topportion)):
        portion.append(topportion[y][0])
    for z in portion:
        flag = True
        for a in portion:
            if a != z:
                if a not in graph[z][0]:
                    flag = False
                    truth.append([z, a, False])
                    break
        if flag:
            truth.append([z, True])
    print(truth)
'''
def MCP(G):
    H = []
    for key in G.keys():
        H.append([key, len(G[key][0])]) #Name, Degree
    H = sorted(H, key=itemgetter(1), reverse=True) #sort in descending order of degree
    i = 0
    while H[i] != H[-1]:
        if H[i] == "Null":
            i += 1
            continue
        else:
            A = H[i][0] #key into G
            for j in range(i + 1, len(H)):
                if H[j] == "Null":
                    continue
                else:
                    B = H[j][0]
                    if (H[i][1] >= H[j][1]) and (G[A][1][B] == 0) and (B not in G[A][0]):
                        #degreeA is >=, nodeB's name isn't in nodeA's RT-dict, nodeB's has no edge in nodeA's adj. list
                        for X in G[B][0]:
                            if X not in G[A][0]:
                                G[X][1][A] = 1 #update X's RT-list w/ A
                                G[A][1][X] = 1 #update A's RT-list w/ X
                        for Y in G[B][0]:
                            G[Y][0].remove(B)
                            for z in range(i + 1, len(H)):
                                if H[z][0] == Y:
                                    H[z][1] -= 1
                                    break
                        H[j] = "Null"
                    elif (H[i][1] < H[j][1]) and (G[A][1][B] == 0) and (B not in G[A][0]):
                        for X in G[A][0]:
                            if X not in G[B][0]:
                                G[X][1][B] = 1 #update X's RT-list w/ A
                                G[B][1][X] = 1 #update A's RT-list w/ X
                        for Y in G[A][0]:
                            G[Y][0].remove(A)
                            for z in range(i + 1, len(H)):
                                if H[z][0] == Y:
                                    H[z][1] -= 1
                                    break
                        temp = [B, H[j][1]]
                        H[i] = temp
                        H[j] = "Null"
                        A = H[i][0]
            i += 1
    maxclique = 0
    for i in range(len(H)):
        if H[i] != "Null":
            maxclique += 1
    return maxclique

#print(unpack("frb30-15-1.txt"))
#print(MCP(graphConv("frb30-15-1.txt")))
print(MCP(graphConv("c125.txt")))
