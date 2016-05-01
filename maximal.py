import time
import random
def makeGraphCopy(graph):
    copy = {}
    for key in graph.keys():
        temp = []
        for n in graph[key]:
            temp.append(n)
        copy[key] = temp
    return copy
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
def checkClique(clique, graph, printing = False):
    flag = True
    for member in range(len(clique)-1):
        neighbors = graph[clique[member]]
        for other in range(member+1, len(clique)):
            if clique[other] not in neighbors:
                flag = False
                break
    if printing:
        print("Return_Set: {0}".format(clique))
        print("Size of Return_Set: {0}".format(len(clique)))
        print("Return_Set is a Clique? -- {0}".format(flag))
    return flag
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
def findMaximal(graph):
    #start = time.time()
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
    #print("Time taken: {0}".format(time.time() - start))
    return base

#graph = graphConv("c125.txt")
#maximalClique = findMaximal(graph)
#print(maximalClique)
#checkClique(maximalClique, graph, True)

def setToZeros(list):
    for i in range(len(list)):
        list[i] = 0
    return list
def RandomFind(filename, origSize, k):
    base = []
    OG = graphConv(filename)
    graph = makeGraphCopy(OG)
    freq = [0]*origSize
    while True:
        freq = setToZeros(freq)
        for i in range(k):
            clique = findMaximal(graph)
            freq[clique[-1]-1] += 1
        nextToAdd = freq.index(max(freq))+1
        base.append(nextToAdd)
        print(base)
        toDel = nonNeighbors(nextToAdd, graph)
        for item in toDel:
            updateDelete(item, graph)
        if len(graph.keys()) == 0:
            break
    base = sorted(base)
    print("Size: {0}".format(len(base)))
    #checkClique(base, OG, True)
    return base
def RandomGraphFind(OG, origSize, k):
    base = []
    graph = makeGraphCopy(OG)
    freq = [0]*origSize
    while True:
        freq = setToZeros(freq)
        for i in range(k):
            clique = findMaximal(graph)
            freq[clique[-1]-1] += 1
        nextToAdd = freq.index(max(freq))+1
        base.append(nextToAdd)
        print(base)
        toDel = nonNeighbors(nextToAdd, graph)
        for item in toDel:
            updateDelete(item, graph)
        if len(graph.keys()) == 0:
            break
    base = sorted(base)
    print("Size: {0}".format(len(base)))
    #checkClique(base, OG, True)
    return base

#RandomFind("c125.txt", 50)
#solution found: [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 70, 71, 77, 79, 92, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
#solution2 found: [5, 7, 9, 11, 19, 24, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 117, 122, 125]
#solution3 found: [7, 9, 11, 29, 40, 45, 54, 80, 104, 110, 114, 25, 52, 1, 5, 19, 31, 34, 44, 49, 55, 65, 66, 68, 70, 77, 79, 82, 96, 98, 103, 117, 122, 125]


#RandomFind("c250.txt", 50)
def delete(toDel, graph):
    for node in toDel:
        neighbors = graph[node]
        for n in neighbors:
            graph[n].remove(node)
        graph.pop(node, None)
    return graph
def closedCliqueNeighborhood(base, graph):
    toDel = []
    for key in graph.keys():
        for member in base:
            if key not in graph[member]:
                toDel.append(key)
                break
    return delete(toDel, graph)
def intersection(listOfLists):
    intersecting = []
    for i in range(len(listOfLists)-1):
        curList = listOfLists[i]
        for member in curList:
            if member not in intersecting:
                flag = True
                for j in range(i+1, len(listOfLists)):
                    if member not in listOfLists[j]:
                        flag = False
                        break
                if flag:
                    intersecting.append(member)
    return intersecting

def MetaRandomFind(graph, origSize, kTries):
    tryCliques = []
    #graph = graphConv(filename)
    #origSize = len(graph.keys())
    iterations = origSize #needs to scale to size of graph... presumably, the more iterations, the better it generally performs
    for i in range(kTries):
        tryCliques.append(RandomGraphFind(graph, origSize, iterations))
    intersect = intersection(tryCliques)
    #graph = closedCliqueNeighborhood(intersect, graph)

    #intersect += RandomGraphFind(graph, origSize, iterations)
    #checkClique(intersect, graphConv(filename), True)
    return intersect

def MetaMetaRandomFind(filename, kTries, seed = []):
    graph = graphConv(filename)
    origSize = len(graph.keys())
    clique = seed
    if len(clique) != 0:
        for item in clique:
            toDel = nonNeighbors(item, graph)
            for deleting in toDel:
                updateDelete(deleting, graph)
    while True:
        intersect = MetaRandomFind(graph, origSize, kTries)
        clique += intersect
        print(clique)
        for item in intersect:
            toDel = nonNeighbors(item, graph)
            for deleting in toDel:
                updateDelete(deleting, graph)
        if len(graph.keys()) == 0:
            break
    clique = sorted(clique)
    checkClique(clique, graphConv(filename), True)
    return clique
    

#MetaRandomFind("c250.txt", 3)#you probably have to scale the number of iterations as the number of nodes increases...
#RandomFind("c250.txt", 250, 250)
#MetaMetaRandomFind("c250.txt", 3)
    
#Known Solution:
#[3, 8, 21, 26, 30, 31, 34, 37, 41, 45, 58, 63, 70, 72, 84, 87, 90, 92, 96, 97, 99, 122, 129, 131, 136, 138, 147, 152, 161, 162, 163, 165, 177, 183, 186, 191, 197, 203, 207, 212, 214, 227, 235, 241]
#Clique Size 43 Found:
#[5, 6, 41, 44, 58, 61, 62, 63, 64, 76, 84, 86, 93, 95, 97, 105, 106, 111, 113, 117, 120, 121, 125, 127, 129, 136, 138, 144, 150, 159, 177, 183, 185, 197, 199, 200, 202, 204, 210, 226, 227, 230, 249]
#Clique Size 42 Found:
#[6, 15, 32, 41, 44, 47, 55, 57, 58, 61, 62, 63, 64, 76, 84, 86, 93, 95, 97, 104, 106, 111, 113, 117, 120, 121, 125, 127, 129, 159, 174, 177, 183, 185, 189, 191, 197, 204, 224, 226, 230, 235]
#Shared Among All:
#Seed1 = [41, 58, 63, 84, 97, 129, 183, 197]
#Shared Among Latter 2:
#Seed2 = [6, 41, 44, 61, 62, 63, 64, 76, 84, 86, 93, 95, 97, 106, 111, 113, 117, 120, 121, 125, 127, 129, 159, 177, 183, 185, 197, 204, 230]

#MetaMetaRandomFind("c250.txt", 3, Seed1)
#w/ Seed1 of [41, 58, 63, 84, 97, 129, 183, 197]:
#Found new solutions: (clique size 44)
#[3, 6, 8, 26, 30, 31, 35, 37, 41, 58, 63, 70, 72, 73, 84, 87, 90, 96, 97, 99, 102, 105, 122, 129, 131, 136, 138, 147, 152, 157, 161, 162, 163, 174, 177, 183, 191, 197, 203, 207, 212, 214, 235, 241]
#[3, 6, 8, 26, 30, 31, 35, 37, 41, 55, 58, 63, 70, 72, 84, 87, 90, 92, 96, 97, 99, 102, 105, 111, 122, 129, 131, 136, 138, 147, 152, 157, 161, 163, 165, 177, 183, 191, 197, 203, 212, 214, 235, 241]

#--> From first new solution, new seed (from this solution and clique size 43):
#Seed3 = [6, 41, 58, 84, 105, 136, 138, 177]
#MetaMetaRandomFind("c250.txt", 3, Seed3)
#Seed3 rarely finds ultimate cliques of size 44, but intermediately, found them, + a new solution:
#[3, 6, 8, 26, 30, 35, 37, 41, 58, 63, 70, 72, 84, 87, 90, 92, 96, 97, 99, 102, 105, 122, 129, 131, 136, 138, 147, 152, 157, 161, 162, 163, 174, 177, 183, 191, 197, 198, 203, 207, 212, 214, 235, 241]

#MetaMetaRandomFind("c250.txt", 3, Seed2)
#w/ Seed2:
#Clique of Size 43 Found:
#[5, 6, 41, 44, 57, 58, 61, 62, 63, 64, 76, 84, 86, 93, 95, 97, 105, 106, 111, 113, 117, 120, 121, 125, 127, 129, 144, 150, 159, 177, 183, 185, 189, 197, 199, 200, 202, 204, 210, 226, 227, 230, 249]
#Ran for 10 iterations -- could not find anything above size 43

'''
#
#
#'''

#MetaMetaRandomFind("c500.txt", 3)

#Best known solution (size 57):
#[21, 22, 33, 40, 46, 61, 63, 87, 97, 110, 121, 122, 132, 137, 155, 179, 181, 182, 186, 189, 193, 194, 203, 212, 223, 244, 248, 249, 253, 266, 280, 290, 294, 310, 316, 319, 327, 329, 340, 350, 351, 357, 373, 374, 375, 381, 390, 395,
# 404, 405, 411, 415, 463, 478, 490, 491, 497]

#####Found clique of size 50:
#[6, 490, 204, 110, 52, 97, 488, 82, 194, 118, 374, 411, 230, 48, 484, 290, 5, 137, 108, 195, 452, 371, 15, 177, 322, 295, 493, 364, 404, 385, 305, 132, 280, 133, 336, 481, 203, 80, 343, 248, 272, 53, 85, 368, 217, 226, 386, 72, 258, 21]
#Shared Seed w/ best-known-solution:
#####[490, 110, 97, 194, 374, 411, 290, 137, 404, 132, 280, 203, 248, 21]

#####Found clique of size 52:
#[97, 110, 5, 186, 295, 28, 404, 411, 493, 478, 132, 195, 82, 194, 40, 61, 108, 248, 319, 484, 405, 454, 374, 113, 329, 294, 425, 153, 47, 435, 415, 407, 68, 368, 323, 84, 208, 166, 63, 327, 217, 488, 386, 410, 43, 300, 204, 332, 149, 439, 346, 471]
#Shared Seed w/ best-known-solution:
#####[97, 110, 186, 404, 411, 478, 132, 194, 40, 61, 248, 319, 405, 374, 294, 415, 63, 327]

#####Found clique of size 49:
#[52, 310, 164, 266, 132, 5, 97, 411, 248, 108, 493, 21, 478, 194, 119, 137, 340, 260, 253, 376, 305, 336, 383, 385, 177, 295, 98, 342, 22, 33, 319, 120, 46, 378, 291, 244, 494, 280, 118, 16, 343, 374, 203, 80, 480, 61, 427, 405, 450]
#Shared Seed w/ best-known-solution:
#####[310, 266, 132, 97, 411, 248, 21, 478, 194, 137, 340, 253, 22, 33, 319, 46, 244, 280, 374, 203, 61, 405]

#####Intersection of all seeds: [97, 110, 132, 194, 248, 374, 404, 411]

#Nodes shared among found cliques: [5, 97, 108, 132, 194, 248, 295, 374, 411, 493, 61, 319, 405, 478]
#Ultimately Returned Set: (Size 51)
#[5, 16, 46, 47, 52, 61, 68, 84, 97, 108, 113, 119, 120, 132, 147, 153, 194, 195, 215, 226, 248, 266, 284, 287, 294, 295, 319, 323, 336, 350, 359, 368, 374, 378, 381, 383, 405, 407, 411, 415, 425, 427, 435, 450, 454, 471, 478, 490, 492, 493, 494]

#MetaMetaRandomFind("c500.txt", 3, [97, 110, 132, 194, 248, 374, 404, 411])
#Ultimately Returned Set: [5, 19, 21, 28, 52, 56, 61, 82, 97, 98, 103, 108, 110, 113, 120, 132, 137, 141, 159, 194, 203, 215, 226, 248, 256, 260, 269, 294, 316, 319, 323, 328, 332, 336, 340, 357, 359, 368, 374, 385, 404, 405, 411, 415, 425, 427, 439, 442, 454, 481, 484, 493]

#a = [310, 266, 182, 61, 181, 202, 290, 108, 385, 244, 87, 21, 147, 316, 222, 478, 433, 493, 253, 351, 40, 94, 193, 378, 212, 186, 405, 319, 68, 497, 22, 77, 350, 294, 137, 145, 361, 189, 463, 190, 323, 1, 357, 33, 390]
#b = [493, 5, 266, 340, 137, 425, 141, 108, 385, 52, 455, 427, 98, 484, 113, 260, 28, 294, 306, 159, 454, 103, 19, 316, 61, 269, 415, 203, 335, 120, 226, 357, 82, 332, 336, 399, 478, 215, 74, 328, 319, 37, 56, 81]
#c = [295, 493, 490, 5, 294, 118, 52, 108, 425, 137, 203, 310, 21, 415, 340, 212, 53, 215, 181, 193, 290, 375, 364, 284, 450, 494, 145, 484, 405, 319, 73, 341, 442, 72, 447, 209, 481, 210, 291, 80, 98]

#d = [21, 22, 33, 40, 46, 61, 63, 87, 97, 110, 121, 122, 132, 137, 155, 179, 181, 182, 186, 189, 193, 194, 203, 212, 223, 244, 248, 249, 253, 266, 280, 290, 294, 310, 316, 319, 327, 329, 340, 350, 351, 357, 373, 374, 375, 381, 390, 395, 404, 405, 411, 415, 463, 478, 490, 491, 497]

#oldSeed = [97, 110, 132, 194, 248, 374, 404, 411]
#newSeed = [319, 294, 137, 266, 61, 316, 478, 357]
#combSeed = oldSeed + newSeed
#MetaMetaRandomFind("c500.txt", 3, combSeed)
#Clique Size 54 = combSeed + [340, 212, 182, 405, 310, 381, 415, 87, 21, 19, 121, 155, 395, 290, 40, 329, 186, 390, 327, 351, 181, 493, 149, 463, 375, 63, 350, 193, 33, 190, 223, 22, 249, 203, 433, 253, 425, 332]
#Clique Size 57 = combSeed + [310, 340, 182, 212, 415, 405, 22, 390, 186, 87, 121, 249, 63, 327, 395, 223, 203, 122, 244, 181, 491, 280, 189, 375, 351, 329, 40, 463, 290, 21, 490, 253, 46, 381, 497, 179, 33, 155, 373, 193, 350]
#Clique Size 54 = combSeed + [310, 340, 19, 212, 182, 405, 381, 415, 21, 332, 425, 121, 155, 190, 493, 433, 149, 375, 40, 390, 327, 181, 186, 87, 350, 463, 290, 351, 193, 395, 33, 329, 22, 253, 361, 77, 68, 1]
#Returned Size 56 = [21, 22, 33, 40, 46, 61, 63, 87, 97, 110, 121, 122, 132, 137, 149, 155, 181, 182, 186, 189, 193, 194, 203, 212, 223, 244, 248, 249, 253, 266, 280, 290, 294, 310, 316, 319, 327, 329, 340, 350, 351, 357, 374, 375, 381, 390, 395, 404, 405, 411, 415, 463, 478, 490, 491, 493]






































