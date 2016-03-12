from operator import itemgetter
import random

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

def MIS(root, neigh, graph):
    for i in range(len(neigh)):
        neighbors = []
        for n in neigh:
            neighbors.append(n)
        
        allnodes = []
        for key in graph.keys():
            allnodes.append(key)
        allnodes.remove(root)
        
        miset = []
        guy = neighbors.pop(i)
        miset.append(guy)
        adj = graph[guy]
        allnodes.remove(guy) #remove v
        #print(allnodes)
        #i = 0
        while len(neighbors) != 0:
            #print(miset)
            #print(i)
            #print("hello2")
            #print(len(neighbors))
            for friend in adj:
                if friend in neighbors:
                    neighbors.remove(friend)
                if friend in allnodes: #remove all its neighbors
                    allnodes.remove(friend)
            #print(allnodes)
            if len(neighbors) != 0:
                guy = neighbors.pop(0)
                allnodes.remove(guy)
                miset.append(guy)
                adj = graph[guy]
            #i += 1
        while len(allnodes) != 0:
            #print("hello3")
            for friend in adj:
                if friend in allnodes:
                    allnodes.remove(friend)
            #print(allnodes)
            if len(allnodes) != 0:
                guy = allnodes.pop(0)
                #print(guy)
                miset.append(guy)
                adj = graph[guy]
        if len(miset) > 1:
            return miset
        else:
            continue
    return ["hello"]

def connected(root, graph):
    adj = graph[root]
    size = len(graph.keys()) - 1
    result = True
    for neighbor in adj:
        if len(graph[neighbor]) != size:
            result = False
            break
    return result

def reduction(root, graph):
    G = graph.copy()
    neighbors = G[root]
    #print(neighbors)
    misam = MIS(root, neighbors, G)
    while not connected(root, G):
        #print("hello1")
        merged = misam[0]
        for i in range(1, len(misam)):
            node = misam[i]
            for item in G[node]:
                #if len(misam) == 5:
                #    print(G[node])
                #    print(item)
                G[item].remove(node)
                if item not in G[merged]:
                    G[merged].append(item)
                    G[item].append(merged)
            G.pop(node, None)
        neighbors = G[root]
        misam = MIS(root, neighbors, G)
    return G

def complete(graph):
    result = True
    size = len(graph.keys()) - 1
    for node in graph.keys():
        if len(graph[node]) != size:
            result = False
            break
    return result

def iterate(filename, vertex):
    graph = graphConv(filename)
    graph = reduction(vertex, graph)
    return complete(graph)
'''
def MCP(filename):
    G = graphConv(filename)
    for vertex in G.keys():
        graph = {}
        for item in G.keys():
            graph[item] = G[item]
        print(G[1])
        graph = reduction(vertex, graph)
        print(G[1])
        if not complete(graph):
            vadj = G[vertex]
            for item in vadj:
                G[item].remove(vertex)
            G.pop(vertex, None)
    print(G.keys())
    print(len(G.keys()))
    return G
'''
def MCP(filename):
    G = graphConv(filename)
    counter = 0
    for vertex in G.keys():
        if iterate(filename, vertex):
            counter += 1
    print(counter)
    return counter

MCP("c125.txt")
        
            
        











        
