def createMeta4CG(graph):
    metaG = {}

    keys = graph.keys()

    counter = 1
    for i in range(len(keys)-1):
        neighbors = graph[keys[i]]
        for j in range(i+1, len(keys)):
            if keys[j] in neighbors:
                metaG[counter] = {"name": [keys[i], keys[j]], "adj": [], "parents": (keys[i],keys[j])}
                counter += 1

    metakeys = metaG.keys()
    for i in range(len(metakeys)-1):
        curMetaNode = metakeys[i]
        parents = metaG[curMetaNode]["parents"]
        neighbors1 = graph[parents[0]]
        neighbors2 = graph[parents[1]]
        for j in range(i+1, len(metakeys)):
            parents2 = metaG[metakeys[j]]["parents"]
            neigh3 = parents2[0]
            neigh4 = parents2[1]
            if neigh3 in neighbors1 and neigh3 in neighbors2 and neigh4 in neighbors1 and neigh4 in neighbors2:
                metaG[metakeys[j]]["adj"].append(curMetaNode)
                metaG[curMetaNode]["adj"].append(metakeys[j])
    for key in metakeys:
        if len(metaG[key]["adj"]) == 0:
            metaG.pop(key, None)
    return metaG

def createMetaMeta(metaG, testprint = False):
    mmeta = {}

    hashing = {}
    keys = metaG.keys()
    counter = 1
    for i in range(len(keys)-1):
        neighbors = metaG[keys[i]]["adj"]
        for j in range(i+1, len(keys)):
            if keys[j] in neighbors:
                name = sorted(metaG[keys[i]]["name"] + metaG[keys[j]]["name"])
                strName = ", ".join(str(e) for e in name)
                if hashing.get(strName) == None:
                    mmeta[counter] = {"name": name, "adj": [], "parents": (keys[i],keys[j])}
                    hashing[strName] = 1
                    counter += 1
    if testprint:
        print(hashing)
    metakeys = mmeta.keys()
    for i in range(len(metakeys)-1):
        curMetaNode = metakeys[i]
        parents = mmeta[curMetaNode]["parents"]
        neighbors1 = metaG[parents[0]]["adj"]
        neighbors2 = metaG[parents[1]]["adj"]
        for j in range(i+1, len(metakeys)):
            parents2 = mmeta[metakeys[j]]["parents"]
            neigh3 = parents2[0]
            neigh4 = parents2[1]
            if neigh3 in neighbors1 and neigh3 in neighbors2 and neigh4 in neighbors1 and neigh4 in neighbors2:
                mmeta[metakeys[j]]["adj"].append(curMetaNode)
                mmeta[curMetaNode]["adj"].append(metakeys[j])
    flag = False
    for key in metakeys:
        if len(mmeta[key]["adj"]) > 0:
            flag = True
    if flag:
        for key in metakeys:
            if len(mmeta[key]["adj"]) == 0:
                mmeta.pop(key, None)
    return mmeta

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

def createComplete(n):
    graph = {}
    for i in range(1, n+1):
        graph[i] = []
        for j in range(1, n+1):
            if j != i:
                graph[i].append(j)
    return graph
def removeFromGraph(node, graph):
    neighbors = graph[node]
    for key in graph.keys():
        if key in neighbors:
            graph[key].remove(node)
    graph.pop(node, None)

#sampleGraph = createComplete(8)
#removeFromGraph(7, sampleGraph)
#graph = graphConv("c125.txt")
#meta1 = createMeta4CG(sampleGraph)
#for key in meta1:
#    print("Key {0}, {1}".format(key, meta1[key]))
#meta2 = createMetaMeta(meta1)
#for key in meta2:
#    print("Key {0}, {1}".format(key, meta2[key]))
#meta3 = createMetaMeta(meta2)
#print(meta3)


