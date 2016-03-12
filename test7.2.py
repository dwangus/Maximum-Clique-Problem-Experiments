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
def reduction(root, graph):
    H = graph.copy()
    while len(nodeweb(H[root], H)) > 0:
        MISAM = MIS(root, nodeweb(H[root], H), H)
        H = merge(MISAM, H)
    return H
########################################################################################################################################################################################
def bounds(root, graph):
    Hbefore = graph.copy()
    Hafter = {}
    for key in graph.keys():
        temp = []
        for item in graph[key]:
            temp.append(item)
        Hafter[key] = temp
    Hafter.pop(root, None)
    
    while len(MISroot(root, Hbefore)) > 1:
        Hbefore = merge(MISroot(root, Hbefore), Hbefore)#doesn't run until completion, just until the root's un-connected neighbors are exhausted
    while len(MISleftovers(Hbefore)) > 0:
        Hbefore = merge(MISleftovers(Hbefore), Hbefore)
    boundbef = len(Hbefore.keys())

    while len(MISleftovers(Hafter)) > 0:
        Hafter = merge(MISleftovers(Hafter), Hafter)
    boundaft = len(Hafter.keys())

    if boundaft != boundbef:
        return True
    else:
        return False
########################################################################################################################################################################################
def MCP(filename):
    G = graphConv(filename)
    #counter = 0
    for vertex in G.keys():
        H = {}
        for key in G.keys():
            temp = []
            for item in G[key]:
                temp.append(item)
            H[key] = temp
        H = reduction(vertex, H)
        if not bounds(vertex, H):
            #counter += 1
            vadj = G[vertex]
            for item in vadj:
                G[item].remove(vertex)
            G.pop(vertex, None)
    #print(counter)
    print(G.keys())
    print(len(G.keys()))
    
    running = []
    for key in G.keys():
        running.append(key)
    for i in range(len(running) - 1):
        first = running[i]
        for j in range(i + 1, len(running)):
            second = running[j]
            if second not in G[first]:
                print(False)
    
    return G
########################################################################################################################################################################################

#18, 21, 30-something... and the ones picked aren't even cliques together...
#MCP("c125.txt")
#MCP("c250.txt")
#MCP("c500.txt")

########################################################################################################################################################################################        
def MCP2(filename):
    G = graphConv(filename)
    mcn = []
    for vertex in G.keys():
        H = {}
        for key in G.keys():
            temp = []
            for item in G[key]:
                temp.append(item)
            H[key] = temp
        H = reduction(vertex, H)
        if bounds(vertex, H):
            mcn.append(vertex)

    print(mcn)
    print(len(mcn))
    
    for i in range(len(mcn) - 1):
        first = running[i]
        for j in range(i + 1, len(running)):
            second = running[j]
            if second not in G[first]:
                print(False)
    
    return G
########################################################################################################################################################################################    

#They all end up being empty lists with the len(MCN) = 0...
#MCP2("c125.txt")
#MCP2("c250.txt")
#MCP2("c500.txt")

########################################################################################################################################################################################    
def MCP3(filename):
    G = graphConv(filename)
    x = random.randint(1, len(G.keys()))
    keys = G.keys()
    root = keys[x-1]
    lower = reduction2(root, G)
    while len(MISroot(root, G)) > 1:
        lower += 1
        G = merge(MISroot(root, G), G)
    while len(MISleftovers(G)) > 0:
        lower += 1
        G = merge(MISleftovers(G), G)
    upper = len(G.keys())
    lower = upper - lower
    print("Range[{0}, {1}]".format(upper, lower))
########################################################################################################################################################################################       
def reduction2(root, G):
    lower = 0
    while len(nodeweb(G[root], G)) > 0:
        MISAM = MIS(root, nodeweb(G[root], G), G)
        if len(MISAM) > 1:
            lower += 1
        G = merge(MISAM, G)
    return lower
########################################################################################################################################################################################

#MCP3("c125.txt")
#MCP3("c250.txt")
#MCP3("c500.txt")

########################################################################################################################################################################################    
def MCP4(filename, x):
    G = graphConv(filename)
    #x = random.randint(1, len(G.keys())-1)
    keys = G.keys()
    root = keys[x]
    merged = {}
    for key in keys:
        merged[key] = []
    #print("Root: {0}".format(root))
    #print("Original: \n{0}\n".format(G.keys()))
    
    reduction3(root, G, merged)
    unneighbors = []
    for key in G.keys():
        if key not in G[root]:
            if key != root:
                unneighbors.append(key)
    #unneighborclique = (1, True)
    unneighborclique = True
    for i in range(len(unneighbors) - 1):
        first = unneighbors[i]
        for j in range(i + 1, len(unneighbors)):
            second = unneighbors[j]
            if second not in G[first]:
                #unneighborclique = (1, False)
                unneighborclique = False
    #if len(unneighbors) == 0:
        #unneighborclique = (0, False)
    #print("Neighbor-Reduction (length: {1}): \n{0}".format(G.keys(), len(G.keys())))
    #print("Root {2} Neighbors (length: {1}): \n{0}".format(G[root], len(G[root]), root))
    #print("Root {2} Un-neighbors (length: {1}): \n{0}".format(unneighbors, len(unneighbors), root))
    #print("Unneighbor-Clique? {0}".format(unneighborclique))
    #for key in merged.keys():
        #if len(merged[key]) > 0:
            #print("{0}, {1}".format(key, merged[key]))
            
    while len(MISroot(root, G)) > 1:
        MISAM = MISroot(root, G)
        if len(MISAM) > 0:
            for i in range(1, len(MISAM)):
                merged[MISAM[0]].append(MISAM[i])
        G = merge(MISAM, G)
    unneighbors = []
    for key in G.keys():
        if key not in G[root]:
            if key != root:
                unneighbors.append(key)
    #print("Root-Reduction (length: {1}): \n{0}".format(G.keys(), len(G.keys())))
    #print("Root {2} Neighbors (length: {1}): \n{0}".format(G[root], len(G[root]), root))
    #print("Root {2} Un-neighbors (length: {1}): \n{0}".format(unneighbors, len(unneighbors), root))
    #for key in merged.keys():
        #if len(merged[key]) > 0:
            #print("{0}, {1}".format(key, merged[key]))
            
    while len(MISleftovers(G)) > 0:
        MISAM = MISleftovers(G)
        if len(MISAM) > 0:
            for i in range(1, len(MISAM)):
                merged[MISAM[0]].append(MISAM[i])
        G = merge(MISAM, G)
    unneighbors = []
    for key in G.keys():
        if key not in G[root]:
            if key != root:
                unneighbors.append(key)
    #print("Any Leftovers (length: {1}): \n{0}".format(G.keys(), len(G.keys())))
    #print("Root {2} Neighbors (length: {1}): \n{0}".format(G[root], len(G[root]), root))
    #print("Root {2} Un-neighbors (length: {1}): \n{0}".format(unneighbors, len(unneighbors), root))
    #for key in merged.keys():
        #if len(merged[key]) > 0:
            #print("{0}, {1}".format(key, merged[key]))
    
    #print("\nUnneighbor-Clique? {0}".format(unneighborclique))
    return unneighborclique
########################################################################################################################################################################################       
def reduction3(root, G, merged):
    while len(nodeweb(G[root], G)) > 0:
        MISAM = MIS(root, nodeweb(G[root], G), G)
        if len(MISAM) > 1:
            for i in range(1, len(MISAM)):
                merged[MISAM[0]].append(MISAM[i])
        G = merge(MISAM, G)
    return G
########################################################################################################################################################################################
def innercycle(filename):
    graph = graphConv(filename)
    size = len(graph)
    redundancy = {}
    for i in range(size):
        truth = MCP4(filename, i)
        redundancy[i + 1] = truth
        #print("Root {0}, Redundant: {1}".format(i + 1, truth))
    #Tcounter = 0
    #Fcounter = 0
    remove = []
    for key in redundancy:
        if redundancy[key]:
            #Tcounter += 1
            remove.append(key)
        #else:
            #Fcounter += 1
    #print("Truths: {0}".format(Tcounter))
    #print("Falses: {0}".format(Fcounter))
    return remove
########################################################################################################################################################################################
def boundsreduction(G):
    keys = G.keys()
    root = keys[0]
    lower = reduction2(root, G)
    while len(MISroot(root, G)) > 1:
        lower += 1
        G = merge(MISroot(root, G), G)
    while len(MISleftovers(G)) > 0:
        lower += 1
        G = merge(MISleftovers(G), G)
    upper = len(G.keys())
    lower = upper - lower
    print("Range[{0}, {1}]".format(upper, lower))
    return(G)
########################################################################################################################################################################################
def MCP6(G, root):
    merged = {}
    for key in G.keys():
        merged[key] = []
    
    reduction3(root, G, merged)
    unneighbors = []
    for key in G.keys():
        if key not in G[root]:
            if key != root:
                unneighbors.append(key)
    unneighborclique = True
    for i in range(len(unneighbors) - 1):
        first = unneighbors[i]
        for j in range(i + 1, len(unneighbors)):
            second = unneighbors[j]
            if second not in G[first]:
                unneighborclique = False
    return unneighborclique
######################################################################################################################################################################################## 
def innercycle2(graph):
    size = len(graph)
    redundancy = {}
    for key in graph.keys():
        H = {}
        for key in graph.keys():
            temp = []
            for item in graph[key]:
                temp.append(item)
            H[key] = temp
        truth = MCP6(H, key)
        redundancy[key] = truth
    remove = []
    for key in redundancy:
        if redundancy[key]:
            remove.append(key)
    return remove
######################################################################################################################################################################################## 
def outercycle(filename):
    remove = innercycle(filename)
    outergraph = graphConv(filename)
    for item in remove:
        friends = outergraph[item]
        for friend in friends:
            outergraph[friend].remove(item)
        outergraph.pop(item, None)
    remove2 = innercycle2(outergraph)
    remove += remove2
    newgraph = graphConv(filename)
    for item in remove:
        friends = newgraph[item]
        for friend in friends:
            newgraph[friend].remove(item)
    newgraph = boundsreduction(newgraph)
    return newgraph
########################################################################################################################################################################################    

#outercycle("c125.txt")
#print(MCP4("c125.txt", 1))

        
