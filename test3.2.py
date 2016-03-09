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

def MCP(G):
    H = []
    for key in G.keys():
        H.append([key, len(G[key][0])]) #Name, Degree
    H = sorted(H, key=itemgetter(1), reverse=True) #sort in descending order of degree
    M = []
    while len(H) != 1:
        A = H[0][0] #key into G
        toremove = []
        for j in range(1, len(H)):
            B = H[j][0]
            if (G[A][1][B] == 0) and (B not in G[A][0]):
                for X in G[B][0]:
                    if X not in G[A][0]:
                        G[A][1][X] = 1 #update A's RT-list w/ X
                for Y in G[B][0]:
                    G[Y][0].remove(B)
                    for z in range(1, len(H)):
                        if H[z][0] == Y:
                            H[z][1] -= 1
                            break
                toremove.append(B)
        if len(toremove) == 0:
            M.append(H.pop(0))
            continue
        else:
            for item in toremove:
                h = 1
                while h != len(H):
                    if H[h][0] == item:
                        H.pop(h)
                    else:
                        h += 1
            M.append(H.pop(0))
            H = sorted(H, key=itemgetter(1), reverse = True)
    return(len(M) + 1)

#print(MCP(graphConv("c125.txt")))
#print(MCP(graphConv("c125.txt")))
print(MCP(graphConv("frb30-15-1.txt")))
#doesn't work with, let's say, the first node -- when it can't be merged
#with anymore nodes in the graph, you don't know for sure whether or not
#it's either connected to one of the highest clique or has merged with/
#is one of the highest clique
