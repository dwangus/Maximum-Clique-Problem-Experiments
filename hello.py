import random
import math
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
def makeGraphCopy(graph):
    copy = {}
    for key in graph.keys():
        temp = []
        for n in graph[key]:
            temp.append(n)
        copy[key] = temp
    return copy
def closedCliqueNeighborhood(base, graph):
    toDel = []
    for key in graph.keys():
        for member in base:
            if key not in graph[member]:
                toDel.append(key)
                break
    return delete(toDel, graph)
def delete(toDel, graph):
    for node in toDel:
        neighbors = graph[node]
        for n in neighbors:
            graph[n].remove(node)
        graph.pop(node, None)
    return graph
def updateDelete(node, graph):
    neighbors = graph[node]
    for n in neighbors:
        graph[n].remove(node)
    graph.pop(node, None)
def nonNeighbors(node, graph):
    antiNeighbors = []
    neighbors = graph[node]
    for key in graph.keys():
        if key not in neighbors:
            antiNeighbors.append(key)
    return antiNeighbors

def findMaximal(graph):
    base = []
    temp = makeGraphCopy(graph)
    while True:
        nextNode = temp.keys()[random.randint(0, len(temp.keys())-1)]
        base.append(nextNode)
        toDel = nonNeighbors(nextNode, temp)
        for item in toDel:
            updateDelete(item, temp)
        if len(temp.keys()) == 0:
            break
    return base
def findMutual(one, two, set, graph):
    count = 0
    memberset = []
    for member in set:
        if member in graph[one] and member in graph[two]:
            count += 1
            memberset.append(member)
    return count, memberset

def hello(OG, first, second):
    temp = makeGraphCopy(OG)
    #estik = 46.0899
    #first = random.randint(1, len(OG.keys()))
    #second = OG[first][random.randint(0, len(OG[first])-1)]
    #print((first, second))
    edge = [first, second]
    temp = delete(edge, temp)
    clique = findMaximal(temp)

    cliquecount, cliqueshared = findMutual(first, second, clique, OG)
    newprob = (1.0/125) ** (1.0/(cliquecount-1))
    temp = makeGraphCopy(OG)
    temp = delete(cliqueshared, temp)
    others = []
    for key in temp.keys():
        if key not in others and key not in edge:
            others.append(key)

    othercount, othershared = findMutual(first, second, others, OG)
    #print(cliquecount)
    #print(othercount)
    #temp = closedCliqueNeighborhood(edge + cliqueshared, OG)
    #hell = metaHello(temp)
    hell = math.log(1.0/othercount, newprob)
    estim = cliquecount + hell
    return estim

def firsthello(OG):
    #OG = graphConv(filename)
    temp = makeGraphCopy(OG)
    #estik = 46.0899
    first = random.randint(1, len(OG.keys()))
    second = OG[first][random.randint(0, len(OG[first])-1)]
    #print((first, second))
    edge = [first, second]
    temp = delete(edge, temp)
    clique = findMaximal(temp)

    cliquecount, cliqueshared = findMutual(first, second, clique, OG)
    newprob = (1.0/125) ** (1.0/(cliquecount-1))
    temp = makeGraphCopy(OG)
    temp = delete(cliqueshared, temp)
    others = []
    for key in temp.keys():
        if key not in others and key not in edge:
            others.append(key)

    othercount, othershared = findMutual(first, second, others, OG)
    #print(cliquecount)
    #print(othercount)
    #temp = closedCliqueNeighborhood(edge + cliqueshared, OG)
    #hell = metaHello(temp)
    hell = math.log(1.0/othercount, newprob)
    estim = cliquecount + hell
    return estim

'''
def metaHello(graph):
    OG = graph
    temp = makeGraphCopy(OG)
    #first = random.randint(1, len(OG.keys()))
    first = OG.keys()[random.randint(1, len(OG.keys()))]
    second = OG[first][random.randint(0, len(OG[first])-1)]
    edge = [first, second]
    temp = delete(edge, temp)
    clique = findMaximal(temp)

    cliquecount, cliqueshared = findMutual(first, second, clique, OG)
    newprob = (1.0/125) ** (1.0/(cliquecount-1))
    temp = makeGraphCopy(OG)
    print(cliqueshared)
    print(edge)
    temp = delete(cliqueshared, temp)
    print(temp)
    others = []
    for key in temp.keys():
        if key not in others and key not in edge:
            others.append(key)

    #print(cliqueshared)
    othercount, othershared = findMutual(first, second, others, OG)
    #print(cliquecount)
    #print(othercount)
    #print(othershared)
    hell = math.log(1.0/othercount, newprob)
    estim = cliquecount + hell
    return estim'''
def edgeList(filename):
    list = []
    with open(filename, 'r') as file:
        for line in file:
            edge = line.split()
            if edge[0] == 'e':
                e1 = int(edge[1])
                e2 = int(edge[2])
                list.append((e1, e2))
    return list
def firstavg(graph, k):
    sumAvg = 0.0
    for i in range(k):
        sumAvg += firsthello(graph)
    sumAvg /= k
    #print(sumAvg)
    return sumAvg

def avg(graph, first, second, k):
    sumAvg = 0.0
    for i in range(k):
        sumAvg += hello(graph, first, second)
    sumAvg /= k
    #print(sumAvg)
    return sumAvg

def iteration(filename, k):
    OG = graphConv(filename)
    eList = edgeList(filename)
    average = firstavg(OG, k)
    below = []
    for edge in eList:
        predix = avg(OG, edge[0], edge[1], k)
        if predix < average:
            below.append((predix, edge))
            print("Appending {0}".format(edge))
    below = sorted(below, key=itemgetter(0))
    for edge in below:
        OG[edge[0]].remove(edge[1][1])
        OG[edge[1]].remove(edge[1][0])
    average = firstavg(OG, k)
    print(below)
    print(average)
    return average, below

def iterationResume(filename, k, prevList):
    OG = graphConv(filename)
    eList = edgeList(filename)
    average = firstavg(OG, k)
    prevList = sorted(prevList, key=itemgetter(0))
    below = []
    for edge in prevList:
        predix = avg(OG, edge[0], edge[1], k)
        below.append((predix, edge))
        print(below[-1])
    lastEdge = prevList[-1]
    for edge in eList:
        if edge[0] >= lastEdge[0] and edge[1] > lastEdge[1]:
            predix = avg(OG, edge[0], edge[1], k)
            if predix < average:
                below.append((predix, edge))
                print("Appending {0}".format(edge))
    below = sorted(below, key=itemgetter(0))
    print(below)
    for edge in below:
        OG[edge[0]].remove(edge[1][1])
        OG[edge[1]].remove(edge[1][0])
    average = firstavg(OG, k)
    print(average)
    return average, below

def wholeIteration(filename, k):
    OG = graphConv(filename)
    eList = edgeList(filename)
    average = firstavg(OG, k*10)
    print(average)
    below = []
    for edge in eList:
        print(edge)
        predix = avg(OG, edge[0], edge[1], k)
        below.append((predix,edge))
    below = sorted(below, key=itemgetter(0))
    print(below)
    return average, below

    
#iteration("c125.txt", 10)
previous = [(3, 1), (5, 3), (6, 1), (9, 6), (10, 6), (11, 3), (12, 3), (12, 4), (12, 5), (13, 1), (13, 3), (13, 4), (13, 12), (14, 3), (14, 4), (14, 12), (15, 1), (15, 3), (15, 4), (15, 5), (15, 7), (15, 8), (15, 10), (15, 13), (16, 1), (16, 2), (16, 6), (16, 8), (16, 12), (16, 13), (16, 14), (16, 15), (17, 3), (17, 4), (17, 6), (17, 7), (17, 10), (17, 12), (17, 14), (17, 15), (17, 16), (18, 3), (18, 15), (18, 16), (19, 16), (20, 3), (20, 12), (20, 14), (20, 15), (20, 16), (20, 17), (21, 3), (21, 6), (21, 14), (21, 15), (21, 17), (21, 20), (22, 6), (22, 15), (22, 16), (22, 21), (23, 3), (23, 12), (23, 15), (23, 20), (24, 3), (24, 14), (24, 20), (25, 6), (25, 12), (26, 2), (26, 4), (26, 6), (26, 12), (26, 14), (26, 15), (26, 16), (26, 18), (26, 20), (26, 25), (27, 3), (27, 4), (27, 10), (27, 12), (27, 15), (27, 16), (27, 23), (28, 3), (28, 12), (28, 16), (28, 17), (29, 15), (30, 3), (30, 6), (30, 12), (30, 23), (31, 12), (31, 16), (31, 21), (31, 26), (31, 30), (32, 14), (32, 21), (32, 27), (33, 3), (33, 6), (33, 7), (33, 8), (33, 9), (33, 14), (33, 15), (33, 16), (33, 21), (33, 27), (33, 32), (34, 12), (34, 28), (35, 12), (35, 15), (35, 16), (35, 26), (36, 1), (36, 2), (36, 3), (36, 4), (36, 6), (36, 10), (36, 11), (36, 12), (36, 14), (36, 15), (36, 16), (36, 18), (36, 21), (36, 22), (36, 23), (36, 24), (36, 25), (36, 26), (36, 27), (36, 28), (36, 29), (36, 30), (36, 31), (36, 32), (36, 35), (37, 3), (37, 6), (37, 12), (37, 13), (37, 14), (37, 15), (37, 16), (37, 20), (37, 21), (37, 25), (37, 27), (37, 28), (37, 30), (37, 33), (37, 36), (38, 3), (38, 6), (38, 10), (38, 12), (38, 13), (38, 18), (38, 20), (38, 22), (38, 32), (38, 33), (38, 36), (38, 37), (39, 3), (39, 6), (39, 16), (39, 21), (39, 27), (39, 33), (39, 37), (40, 3), (40, 4), (41, 4), (41, 12), (41, 15), (41, 16), (41, 21), (41, 27), (41, 30), (41, 32), (41, 36), (42, 2), (42, 5), (42, 9), (42, 12), (42, 13), (42, 15), (42, 16), (42, 19), (42, 20), (42, 21), (42, 24), (42, 26), (42, 27), (42, 28), (42, 30), (42, 32), (42, 36), (42, 37), (42, 41), (43, 3), (43, 15), (43, 16), (43, 23), (43, 25), (43, 26), (43, 28), (43, 30), (43, 32), (43, 38), (43, 41), (43, 42), (44, 6), (44, 12), (44, 15), (44, 26), (44, 33), (46, 5), (46, 12), (46, 14), (46, 21), (46, 27), (46, 32), (46, 43), (47, 2), (47, 3), (47, 15), (47, 16), (47, 20), (47, 36), (48, 6), (48, 16), (48, 17), (48, 36), (48, 38), (48, 42), (49, 10), (49, 12), (49, 14), (50, 2), (50, 3), (50, 4), (50, 5), (50, 10), (50, 14), (50, 15), (50, 16), (50, 20), (50, 21), (50, 24), (50, 26), (50, 28), (50, 32), (50, 33), (50, 35), (50, 37), (50, 38), (50, 40), (50, 41), (50, 42), (50, 43), (50, 47), (51, 2), (51, 3), (51, 5), (51, 6), (51, 10), (51, 12), (51, 14), (51, 16), (51, 17), (51, 21), (51, 22), (51, 23), (51, 24), (51, 26), (51, 27), (51, 30), (51, 31), (51, 35), (51, 37), (51, 39), (51, 41), (51, 44), (51, 46), (52, 1), (52, 12), (52, 20), (52, 28), (52, 36), (52, 37), (52, 38), (52, 41), (52, 42), (52, 43), (52, 50), (53, 3), (53, 6), (53, 12), (53, 14), (53, 15), (53, 21), (53, 23), (53, 26), (53, 32), (53, 36), (53, 37), (53, 38), (53, 41), (53, 42), (53, 50), (53, 51), (54, 12), (54, 26), (54, 37), (54, 51), (55, 3), (55, 10), (55, 12), (55, 13), (55, 14), (55, 15), (55, 16), (55, 17), (55, 18), (55, 21), (55, 24), (55, 26), (55, 27), (55, 32), (55, 36), (55, 38), (55, 42), (55, 43), (55, 50), (55, 51), (55, 52), (56, 4), (56, 6), (56, 10), (56, 13), (56, 15), (56, 16), (56, 20), (56, 21), (56, 23), (56, 27), (56, 28), (56, 30), (56, 31), (56, 33), (56, 34), (56, 35), (56, 38), (56, 42), (56, 50), (56, 51), (57, 6), (57, 12), (57, 16), (57, 17), (57, 28), (57, 32), (57, 36), (57, 42), (57, 43), (57, 51), (57, 52), (58, 3), (58, 6), (58, 15), (58, 16), (58, 20), (58, 21), (58, 25), (58, 36), (58, 37), (58, 38), (58, 42), (58, 43), (58, 51), (58, 55), (58, 56), (58, 57), (59, 2), (59, 12), (59, 15), (59, 16), (59, 20), (59, 21), (59, 33), (59, 36), (59, 42), (59, 50), (59, 56), (61, 4), (61, 6), (61, 12), (61, 15), (61, 16), (61, 17), (61, 20), (61, 21), (61, 27), (61, 30), (61, 33), (61, 37), (61, 38), (61, 39), (61, 48), (61, 50), (61, 51), (61, 53), (61, 57), (61, 59), (62, 3), (62, 12), (62, 15), (62, 16), (62, 17), (62, 20), (62, 22), (62, 23), (62, 33), (62, 36), (62, 37), (62, 38), (62, 42), (62, 48), (62, 50), (62, 51), (62, 55), (62, 56), (63, 1), (63, 3), (63, 4), (63, 5), (63, 12), (63, 15), (63, 16), (63, 20), (63, 21), (63, 23), (63, 30), (63, 33), (63, 34), (63, 36), (63, 37), (63, 38), (63, 41), (63, 49), (63, 50), (63, 52), (63, 56), (63, 59), (64, 1), (64, 2), (64, 3), (64, 4), (64, 5), (64, 6), (64, 8), (64, 10), (64, 11), (64, 12), (64, 13), (64, 15), (64, 18), (64, 20), (64, 21), (64, 22), (64, 23), (64, 24), (64, 26), (64, 27), (64, 29), (64, 30), (64, 31), (64, 32), (64, 33), (64, 36), (64, 37), (64, 38), (64, 39), (64, 41), (64, 42), (64, 43), (64, 44), (64, 46), (64, 47), (64, 48), (64, 49), (64, 50), (64, 51), (64, 52), (64, 53), (64, 54), (64, 55), (64, 56), (64, 57), (64, 58), (64, 59), (64, 60), (64, 61), (64, 62), (64, 63), (65, 1), (65, 12), (65, 15), (65, 24), (65, 36), (65, 50), (65, 51), (65, 61), (65, 63), (65, 64), (66, 6), (66, 12), (66, 14), (66, 16), (66, 30), (66, 32), (66, 33), (66, 34), (66, 36), (66, 37), (66, 50), (66, 53), (66, 57), (66, 58), (66, 64), (67, 16), (67, 50), (67, 61), (68, 15), (68, 20), (68, 26), (68, 27), (68, 33), (68, 36), (68, 37), (68, 42), (68, 46), (68, 50), (68, 61), (68, 64), (68, 66), (69, 3), (69, 12), (69, 16), (69, 17), (69, 33), (69, 36), (69, 44), (69, 56), (69, 58), (69, 63), (69, 64), (69, 66), (70, 3), (70, 16), (70, 21), (70, 27), (70, 36), (70, 64), (71, 12), (71, 16), (71, 27), (71, 38), (71, 55), (71, 58), (71, 63), (71, 66), (72, 3), (72, 12), (72, 21), (72, 37), (72, 42), (72, 51), (72, 55), (72, 56), (72, 61), (72, 62), (72, 63), (72, 64), (73, 1), (73, 3), (73, 4), (73, 6), (73, 9), (73, 10), (73, 12), (73, 13), (73, 15), (73, 16), (73, 17), (73, 18), (73, 20), (73, 21), (73, 22), (73, 23), (73, 24), (73, 26), (73, 27), (73, 28), (73, 30), (73, 32), (73, 33), (73, 35), (73, 37), (73, 38), (73, 41), (73, 42), (73, 44), (73, 46), (73, 47), (73, 50), (73, 53), (73, 55), (73, 56), (73, 57), (73, 58), (73, 60), (73, 61), (73, 62), (73, 63), (73, 64), (73, 65), (73, 66), (73, 67), (73, 69), (73, 70), (73, 71), (73, 72), (74, 3), (74, 12), (74, 15), (74, 17), (74, 21), (74, 36), (74, 41), (74, 42), (74, 43), (74, 56), (74, 64), (74, 68), (74, 71), (75, 3), (75, 4), (75, 5), (75, 6), (75, 10), (75, 12), (75, 14), (75, 16), (75, 17), (75, 18), (75, 20), (75, 21), (75, 23), (75, 25), (75, 26), (75, 27), (75, 28), (75, 30), (75, 31), (75, 33), (75, 38), (75, 41), (75, 42), (75, 43), (75, 44), (75, 46), (75, 47), (75, 48), (75, 50), (75, 51), (75, 52), (75, 56), (75, 58), (75, 60), (75, 61), (75, 62), (75, 63), (75, 64), (75, 68), (75, 69), (75, 74), (76, 1), (76, 2), (76, 3), (76, 5), (76, 6), (76, 7), (76, 8), (76, 9), (76, 10), (76, 11), (76, 13), (76, 14), (76, 15), (76, 17), (76, 18), (76, 19), (76, 20), (76, 21), (76, 22), (76, 23), (76, 24), (76, 25), (76, 26), (76, 27), (76, 30), (76, 32), (76, 33), (76, 34), (76, 36), (76, 37), (76, 40), (76, 41), (76, 42), (76, 43), (76, 44), (76, 46), (76, 47), (76, 49), (76, 51), (76, 55), (76, 56), (76, 57), (76, 59), (76, 60), (76, 61), (76, 62), (76, 63), (76, 64), (76, 65), (76, 66), (76, 67), (76, 68), (76, 69), (76, 70), (76, 71), (76, 73), (77, 3), (77, 36), (77, 42), (77, 43), (77, 55), (77, 56), (77, 75), (77, 76), (78, 2), (78, 3), (78, 14), (78, 15), (78, 16), (78, 17), (78, 20), (78, 27), (78, 28), (78, 32), (78, 33), (78, 34), (78, 38), (78, 42), (78, 44), (78, 47), (78, 51), (78, 52), (78, 56), (78, 58), (78, 59), (78, 61), (78, 64), (78, 66), (78, 71), (78, 72), (78, 73), (78, 74), (78, 75), (78, 76), (79, 12), (79, 16), (79, 33), (79, 55), (79, 64), (79, 68), (79, 76), (80, 15), (80, 75), (81, 3), (81, 4), (81, 5), (81, 6), (81, 8), (81, 9), (81, 10), (81, 12), (81, 13), (81, 14), (81, 15), (81, 16), (81, 17), (81, 19), (81, 20), (81, 21), (81, 22), (81, 23), (81, 24), (81, 26), (81, 27), (81, 28), (81, 29), (81, 37), (81, 41), (81, 43), (81, 46), (81, 47), (81, 49), (81, 50), (81, 52), (81, 58), (81, 61), (81, 62), (81, 63), (81, 64), (81, 66), (81, 71), (81, 72), (81, 73), (81, 76), (81, 78), (81, 80), (82, 3), (82, 4), (82, 15), (82, 16), (82, 17), (82, 21), (82, 27), (82, 33), (82, 36), (82, 43), (82, 55), (82, 56), (82, 64), (82, 71), (82, 76), (83, 1), (83, 2), (83, 5), (83, 7), (83, 8), (83, 9), (83, 12), (83, 13), (83, 14), (83, 15), (83, 16), (83, 18), (83, 20), (83, 21), (83, 23), (83, 25), (83, 27), (83, 28), (83, 30), (83, 31), (83, 32), (83, 33), (83, 36), (83, 37), (83, 38), (83, 39), (83, 40), (83, 43), (83, 44), (83, 45), (83, 50), (83, 51), (83, 52), (83, 53), (83, 56), (83, 57), (83, 58), (83, 59), (83, 62), (83, 63), (83, 64), (83, 65), (83, 66), (83, 68), (83, 70), (83, 71), (83, 72), (83, 73), (83, 74), (83, 75), (83, 77), (83, 78), (83, 79), (83, 80), (83, 81), (83, 82), (84, 4), (84, 6), (84, 15), (84, 16), (84, 17), (84, 21), (84, 26), (84, 27), (84, 28), (84, 31), (84, 32), (84, 61), (84, 62), (84, 63), (84, 73), (84, 75), (84, 76), (84, 83), (85, 3), (85, 14), (85, 15), (85, 16), (85, 17), (85, 26), (85, 32), (85, 42), (85, 53), (85, 58), (85, 59), (85, 61), (85, 63), (85, 65), (85, 73), (85, 76), (85, 78), (85, 81), (85, 83), (86, 3), (86, 6), (86, 12), (86, 15), (86, 16), (86, 31), (86, 42)]
#iterationResume("c125.txt", 10, previous)
#graph = graphConv("c125.txt")
#print(findMaximal(graph))

wholeIteration("c125.txt", 10)
    




















