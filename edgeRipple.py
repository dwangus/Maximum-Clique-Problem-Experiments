import math
import time
import numpy as np
from operator import itemgetter
import ast

def graphConvE(filename):
    graph = {}
    edges = []
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
                edges.append((e2,e1))
    return graph, edges

def mutual(graph, n1, n2):
    N1 = graph[n1]
    N2 = graph[n2]
    shared = []
    for n in N1:
        if n in N2:
            shared.append(n)
    return sorted(shared)

def closedNeighborhood(graph, clique):
    for key in graph.keys():
        inClique = True
        if key not in clique:
            for c in clique:
                if key not in graph[c]:
                    inClique = False
                    break
        if not inClique:
            for m in graph[key]:
                graph[m].remove(key)
            graph.pop(key, None)
    for c in clique:
        for m in graph[c]:
            graph[m].remove(c)
        graph.pop(c, None)
    return graph

def closedEdgeNeighborhood(graph, clique, eDict, edges):
    for key in graph.keys():
        inClique = True
        if key not in clique:
            for c in clique:
                if key not in graph[c]:
                    inClique = False
                    break
        if not inClique:
            for m in graph[key]:
                graph[m].remove(key)
                if key < m:
                    e = (key, m)
                else:
                    e = (m, key)
                eDict.pop(e, None)
                edges.remove(e)
            graph.pop(key, None)
    for c in clique:
        for m in graph[c]:
            graph[m].remove(c)
            if c < m:
                e = (c, m)
            else:
                e = (m, c)
            eDict.pop(e, None)
            edges.remove(e)
        graph.pop(c, None)
    return graph, eDict, edges

def disconnected(graph):
    disc = []
    for key in graph.keys():
        if len(graph[key]) == 0:
            disc.append(key)
    return disc

def edgeList(graph):
    list = []
    for key in graph.keys():
        neighbors = graph[key]
        for n in neighbors:
            if key < n:
                list.append((key,n))
    return list

def mutFirst(graph, mutu, edges):
    for e in edges:
        mut = sorted(mutual(graph, e[0], e[1]))
        temp = []
        for i in range(len(mut)-1):
            neigh1 = mut[i]
            for j in range(i+1, len(mut)):
                if mut[j] in graph[neigh1]:
                    temp.append((neigh1, mut[j]))
        mutu[e] = temp
    return mutu

def edgeRank(eDict, edgeList, graph, mutu, descending = False):
    for e in edgeList:
        mut = mutu[e]
        for m in mut:
            eDict[m] += 1
    rankings = []
    for key in eDict.keys():
        rankings.append((key, eDict[key]))
    rankings = sorted(rankings, key=itemgetter(1), reverse=descending)
    return rankings

def edgeRank2(eDict, edgeList, graph, descending = False):
    for e in edgeList:
        mut = mutual(graph, e[0], e[1])
        for i in range(len(mut)-1):
            neigh1 = mut[i]
            for j in range(i+1, len(mut)):
                tupn = (neigh1,mut[j])
                if eDict.get(tupn) != None:
                    eDict[tupn] += 1
    rankings = []
    for key in eDict.keys():
        rankings.append((key, eDict[key]))
    rankings = sorted(rankings, key=itemgetter(1), reverse=descending)
    return rankings

def main3(filename, onResume = []):
    start = time.time()
    graph, edges = graphConvE(filename)
    if len(onResume) > 0:
        for edge in onResume:
            #print(edge)
            edges.remove(edge)
            graph[edge[0]].remove(edge[1])
            graph[edge[1]].remove(edge[0])
            disc = disconnected(graph)
            for d in disc:
                print("Deleting: {0}".format(d))
                graph.pop(d, None)                
    eDict = {}
    for e in edges:
        eDict[e] = 0

    mutu = mutFirst(graph, {}, edges)
    size = len(graph.keys())
    #addTo = []
    r = edgeRank(eDict, edges, graph, mutu)
    while len(edges) < ((size*(size-1))/2):
        item = r.pop(0)
        edge = item[0]
        for mE in mutu[edge]:
            eDict[mE] -= 1
        '''addTo.append(edge)
        if len(addTo)%15 == 0:
            print(addTo)'''
        print("{0}, {1}, {2} vs. {3}'s {4}".format(len(edges), edge, item[1], r[-1][0], r[-1][1]))
        edges.remove(edge)
        eDict.pop(edge, None)
        graph[edge[0]].remove(edge[1])
        graph[edge[1]].remove(edge[0])
        for m in mutu.keys():
            adjList = mutu[m]
            if edge in adjList:
                adjList.remove(edge)
        disc = disconnected(graph)
        for d in disc:
            print("Deleting: {0}".format(d))
            for e in edges:
                if d in mutu[e]:
                    mutu[e].remove(d)
            graph.pop(d, None)
        rankings = []
        for key in eDict.keys():
            rankings.append((key, eDict[key]))
        r = sorted(rankings, key=itemgetter(1))
        size = len(graph.keys())
        
    print(len(graph.keys()))
    print(graph.keys())
    #clique = True
    #for key in graph.keys():
    #    if len(graph[key]) != (size-1):
    #        clique = False
    #        break
    #print(clique)
    print(time.time() - start)
    return graph

def main2(filename):
    start = time.time()
    graph, edges = graphConvE(filename)
    eDict = {}
    for e in edges:
        eDict[e] = 0
    clique = []
    while True:
        print(len(clique)/2)
        if len(graph.keys()) <= 4:
            break
        r = edgeRank(eDict, edges, graph)
        edge = r[-1][0]
        e = [edge[0], edge[1]]
        clique += e
        graph, eDict, edges = closedEdgeNeighborhood(graph, e, eDict, edges)
        ###Optional
        for key in eDict.keys():
            eDict[key] = 0
    print(graph)
    clique = sorted(clique)
    print(len(clique))
    return clique, graph

def main(filename):
    start = time.time()
    graph, edges = graphConvE(filename)
    eDict = {}
    for e in edges:
        eDict[e] = 0
    '''for i in range(iterations):
        r = edgeRank2(eDict, edges, graph)
        edge = r[0][0]
        edges.remove(edge)
        eDict.pop(edge, None)
        graph[edge[0]].remove(edge[1])
        graph[edge[1]].remove(edge[0])
        ###Optional
        for key in eDict.keys():
            eDict[key] = 0'''
    rankings = edgeRank2(eDict, edges, graph, True)
    for e in rankings[len(rankings)-50:]:
        print(e)
    '''nodes = {}
    for key in graph.keys():
        nodes[key] = 0
    for e in rankings:
        nodes[e[0][0]] += e[1]
        nodes[e[0][1]] += e[1]
    nrankings = []
    for n in nodes.keys():
        nrankings.append((n, nodes[n]))
    nrankings = sorted(nrankings, key=itemgetter(1), reverse=True)
    for n in nrankings:
        print(n)'''
    for e in rankings[:50]:
        print(e)
    #print("Split!")
    #for e in rankings[len(edges)-500:]:
    #    print(e)
    print(time.time() - start)
    return rankings

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
def isClique(graph, clique):
    flag = True
    for c1 in range(len(clique)-1):
        adjC1 = graph[clique[c1]]
        for c2 in range(c1+1, len(clique)):
            if clique[c2] not in adjC1:
                flag = False
    print(flag)

def main4(filename, clique):
    start = time.time()
    graph, edges = graphConvE(filename)
    OG = graphConv(filename)
    graph = closedNeighborhood(graph, clique)
    edges = edgeList(graph)
    eDict = {}
    for e in edges:
        eDict[e] = 0
    mut = mutFirst(graph, {}, edges)
    rankings = edgeRank(eDict, edges, graph, mut, False)
    nodes = [0]*(len(OG.keys())+1)
    for e in rankings:
        nodes[e[0][0]] += e[1]
        nodes[e[0][1]] += e[1]
    for e in rankings[len(rankings)-20:]:
        print(e)
    print(nodes.index(max(nodes)))
    print(time.time() - start)
    print(clique)
    print(len(clique))
    print(graph)
    #isClique(OG, clique)

#main4("c125.txt", [114,104,45,54,80,7,29,60,110,40,99,34,1,117,68,44,31,96,66,9,11,49,122,52,103,55,125,25,70,5])
main("c125.txt")
#r = main("c125.txt", 5)
#c, graph = main2("c125.txt")
#graph = main3("c125.txt")
'''
onResume = [(83, 108),(83, 90),(15, 83), (36, 83), (73, 83), (16, 83), (83, 88), (83, 102), (83, 112), (83, 94), (64, 83), (32, 83), (12, 83), (83, 107), (81, 83), (83, 97), (83, 95), (21, 83), (83, 124), (83, 89), (28, 83), (27, 83), (83, 105)]
onResume += [(83, 109), (83, 87), (37, 83), (74, 83), (53, 83), (75, 83), (13, 83), (39, 83), (58, 83), (30, 83), (83, 119), (63, 83)]
onResume += [(83,116),(38,83),(23,83),(35,83),(56,83),(83,92),(65,83),(78,83),(83,118),(62,83),(8,83),(83,120),]
onResume += [(57,83),(59,83,),(83,115),(83,101),(52,83),(5,83),(70,83),(83,93),(83,111),(71,83),(83,123),(83,122),]
onResume += [(2,83),(51,83),(22,83),(50,83),(14,83),(40,83),(18,83),(83,99),(83,98),(67,83),(33,83),(43,83),(41,83),(83,96),(48,83),(25,83),]
onResume += [(45,83),(68,83),(54,83),(80,83),(83,104),(34,83),(31,83),(11,83),(20,83),(44,83),(83,110),(66,83),(9,83),(83,85),(83,84),(1,83),]
onResume += [(83,114),(29,83),(83,117),(60,83),(49,83),(7,83),(83,121),(72,83),(77,83),(82,83),(79,83),]
onResume += [(36,76),(36,108),(16,36),(36,94),(36,95),(36,64),(36,102),(36,42),(27,36),(15,36),(3,36),(36,121),]
onResume += [(36, 87), (36, 113), (36, 38), (12, 36), (36, 107), (36, 124), (14, 36), (36, 62), (36, 116), (32, 36), (36, 43), (36, 55), (36, 68), (36, 37), (4, 36), (36, 112), (20, 36), (21, 36), (36, 120), (36, 41), (28, 36), (30, 36), (36, 57), (36, 84), (36, 74), (23, 36), (36, 63), (9, 36), (36, 70), (36, 58), (6, 36), (36, 72), (36, 106), (36, 119), (36, 53)]
'''
#graph = main3("c125.txt", onResume)#took like about an hour
#graph, edges = graphConvE("c125.txt")

#clique = [70,117,79,1,7,77,125,115,121,54,99,44,29,11,98,31,49,45,17,2,110,40,122,80,101,34,71,48,19,18,114,9]
#clique += [5, 25]
#print(len(clique))
#isClique(graph, clique)
#print(sorted(clique))
#print(closedNeighborhood(graph, clique))

def checkSolutions(sol, filename):
    sols = []
    with open(filename, 'r') as file:
        for line in file:
            sols.append(ast.literal_eval(line))
    flag = False
    for i in range(len(sols)):
        s = sols[i]
        for n in range(len(s)):
            if s[n] not in sol:
                break
            elif n == (len(s)-1):
                flag = True
                print(i)
    print(flag)
    return flag

#checkSolutions(, "c125_Solutions.txt")




























