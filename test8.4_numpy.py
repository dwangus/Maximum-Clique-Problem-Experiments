import math
import time
from operator import itemgetter
import numpy as np

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

def nonNeigh(originNode, graph):
    unneighbors = []
    neighborhood = graph[originNode]
    for key in graph.keys():
        if key != originNode:
            if key not in neighborhood:
                unneighbors.append(key)
    return unneighbors

def vectorFormation(startpoint, endpoint):
    return np.asarray(endpoint) - np.asarray(startpoint)
def vectorAddition(vector_list):
    vector = np.asarray(vector_list[0])
    for vec in range(1, len(vector_list)):
        vector += np.asarray(vector_list[vec])
    return vector
def vectorNegation(vector_list):
    result = []
    vector = np.asarray(vector_list[0])*(-1)
    for vec in range(1, len(vector_list)):
        vector -= np.asarray(vector_list[vec])
    return vector
def vectorSubtraction(subXFromThis, X):
    return np.asarray(subXFromThis) - np.asarray(X)
def unitVector(vector):
    return (vector / np.linalg.norm(vector))
def dot(A, B):
    return np.dot(A, B)
def vectorProj(Aon, toB):
    toUnitB = unitVector(toB)
    dotScalar = dot(Aon, toUnitB)
    return toUnitB*dotScalar

def equidistant_vectors(N, spacing):
    dims = N-1
    coords = {}
    first = [0.0]*dims
    second = [float(spacing)] + [0.0]*(dims-1)
    third = [float(spacing/2.0), float(spacing/2.0)*math.sqrt(3)] + [0.0]*(dims-2)
    coords[1] = first
    coords[2] = second
    coords[3] = third
    #initialize N equidistant vectors
    for i in range(4, N+1):
        node_coord = [0.0]*dims
        for dimension in range(0, i-2):
            temp = 0.0
            for prevNode in range(1, i):
                temp += coords[prevNode][dimension]
            temp /= float(i-1)
            node_coord[dimension] = temp
        temp = 0.0
        for prevDims in range(0, i-2):
            temp += (node_coord[prevDims]**2)
        last_dim_solution = math.sqrt(float(spacing)**2 - temp)
        node_coord[i-2] = last_dim_solution
        coords[i] = node_coord
    return coords

def dirSim(coords, memoized, graph, pastWeights):
    #newMemoize = []
    newMemoize = {}
    #for i in range(len(memoized)):
    for i in memoized.keys():
        initDir = memoized[i]
        if sum(initDir) == 0.0 and np.linalg.norm(initDir) == 0.0:
            #newMemoize.append(initDir)
            newMemoize[i] = initDir
            continue
        #node = i+1
        node = i
        neighbors = graph[node]
        weights = {}
        sumAvg = 0.0
        for connection in neighbors:
            neighCoord = np.copy(coords[connection])
            #neighCoord += unitVector(memoized[connection-1])
            neighCoord += unitVector(memoized[connection])
            ### Should try and figure out what works best in how to create this new dirUpdate; as of now, I'm preserving the cumulative weight from past iterations
            dirUpdate = np.copy(memoized[i]) + pastWeights[node][connection]*(unitVector(vectorFormation(coords[node], neighCoord)) - unitVector(vectorFormation(coords[node], coords[connection])))
            ###
            dotP = dot(dirUpdate, initDir)
            #dotP = np.linalg.norm(vectorProj(dirUpdate, initDir))
            weights[connection] = dotP
            sumAvg += dotP
        vecList = []
        for key in weights.keys():
            weights[key] /= sumAvg
            pastWeights[node][key] = weights[key]
            vecList.append(weights[key]*unitVector(vectorFormation(coords[node], coords[key])))
        #newMemoize.append(vectorAddition(vecList))
        newMemoize[i] = vectorAddition(vecList)
    return newMemoize

def iterate(graphOrig, numIter, redux = False, origSize = 0):
    start_time = time.time()
    if disconnected(graphOrig):
        return []
    graph = {}
    for key in graphOrig.keys():
        copy = []
        for neighbor in graphOrig[key]:
            copy.append(neighbor)
        graph[key] = copy

    if redux:
        coords = equidistant_vectors(origSize, 50000)
    else:
        coords = equidistant_vectors(len(graph.keys()), 50000)
    #toDel = [54, 114, 7, 125, 29, 80, 104, 11, 67, 70, 49]
    #toDel += [1, 3, 6, 10, 12, 14, 15, 16, 18, 20, 23, 24, 26, 27, 30, 32, 38, 41, 43, 46, 51, 53, 57, 58, 59, 61, 64, 69, 73, 76, 78, 81, 83, 84, 86, 87, 91, 92, 94, 95, 97, 100, 101, 105, 106, 107, 108, 109, 112, 115, 119, 124]

    toDel = []
    for key in graph.keys():
        if len(graph[key]) == 0:
            toDel.append(key)
    #'''
    for node in toDel:
        neighbors = graph[node]
        for n in neighbors:
            graph[n].remove(node)
        graph.pop(node, None)#'''

    weights = {}
    dims = len(coords[1])

    memoized = {}
    for point1 in graph.keys():
        weights[point1] = {}
        weight_p1 = weights[point1]
        cur_point = coords[point1]
        vecList = []
        neighbors = graph[point1]
        numNeighbors = len(neighbors)
        equalWeighting = 1.0/float(numNeighbors)
        if numNeighbors == 0:
            continue
        for point2 in neighbors:
            vecList.append(unitVector(vectorFormation(cur_point, coords[point2])))
            weight_p1[point2] = equalWeighting
        memoized[point1] = vectorAddition(vecList)

    i = 0
    while i < numIter:
        print(i)
        memoized = dirSim(coords, memoized, graph, weights)
        i += 1
    similarity = {}
    #for key in graphOrig.keys():
    finalOrdering = {}
    for key in graph.keys():
        similarity[key] = []
        #myDir = unitVector(memoized[key-1])
        myDir = unitVector(memoized[key])
        #for neighbor in graphOrig[key]:
        for neighbor in graph[key]:
            similarity[key].append((neighbor, dot(myDir, unitVector(vectorFormation(coords[key], coords[neighbor])))))
        similarity[key] = sorted(similarity[key], key=itemgetter(1), reverse=True)
        ordering = []
        for neighb in similarity[key]:
            ordering.append(neighb[0])
        #print("{0}: {1}, ".format(key, ordering))
        finalOrdering[key] = ordering
    return finalOrdering

def main(filename, iterations):
    graph = graphConv(filename)
    return iterate(graph, iterations)
def testSample(graph, iterations, redux = False, origSize = 0):
    return iterate(graph, iterations, redux, origSize)


def findAdditional(graph, clique):
    additional = []
    for key in graph.keys():
        if key not in clique:
            neighbors = graph[key]
            flag = True
            for member in clique:
                if member not in neighbors:
                    flag = False
                    break
            if flag:
                additional.append(key)
    return additional
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
def constructNeighborhood(graph, clique):
    new = {}
    keys = {}
    for member in range(len(clique)):
        keys[member+1] = clique[member]
        neighbors = []
        for other in range(len(clique)):
            if other != member:
                if clique[other] in graph[clique[member]]:
                    neighbors.append(other+1)
        new[member+1] = neighbors
    return new, keys
def analyzeNeighborhood(graph, clique):
    newGraph, keys = constructNeighborhood(graph, clique)
    temp = iterate(newGraph)
    newClique = []
    for node in temp:
        newClique.append(keys[node])
    if checkClique(newClique, graph):
        return newClique
    else:
        return []
def disconnected(graph):    
    for key in graph.keys():
        if len(graph[key]) > 0:
            return False
    return True
def optimizeSub(graph, closedClique):
    takeOut = []
    for i in range(len(closedClique)):
        new = []
        for j in range(len(closedClique)):
            if j != i:
                new.append(clique[j])
        additional = findAdditional(graph, new)
        cliqueOf = analyzeNeighborhood(graph, additional)
        #takeOut.append((closedClique[i], len(cliqueOf), len(additional)))
        takeOut.append((closedClique[i], len(cliqueOf), len(additional), cliqueOf))
    takeOut = sorted(takeOut, key=itemgetter(1), reverse=True)
    print(takeOut)
    print("Previous Size: {0}".format(len(closedClique)))
    closedClique.remove(takeOut[0][0])
    print("Resultant Size: {0}".format(len(closedClique) + takeOut[0][1]))
    closedClique += takeOut[0][3]
    return sorted(closedClique)
def optimizeAdd(graph, openClique):
    print("Previous Size: {0}".format(len(openClique)))
    addition = analyzeNeighborhood(graph, findAdditional(graph, openClique))
    openClique += addition
    print("Resultant Size: {0}".format(len(openClique)))
    return sorted(openClique)
    

file3 = "c500.txt"
file2 = "c250.txt"
#file1 = "C:\Users\dwangus\Desktop\c125.txt"
file1 = "c125.txt"
fileName = file1
sampleGraph = {1: [3, 4], 2: [3, 4, 5], 3: [1, 2], 4: [1, 2, 5], 5: [2, 4]}
#clique = main(fileName, 100)
#clique = testSample(sampleGraph, 100)
#testSample(sampleGraph, 100)
#size = len(clique)
#print("Returned Clique Size: {0}".format(size))


orderings1 = {}
antiOrder = {}

'''
def checkEquiv(order1, order2):
    #flag = True
    for key in order1.keys():
        ordering1 = order1[key]
        ordering2 = order2[key]
        for i in range(len(ordering1)):
            if ordering1[i] != ordering2[i]:
                #flag = False
                print("Key {0}".format(key))
                print("i-th: {0}".format(i))
                #print(ordering1)
                print(ordering1[i])
                #print(ordering2)
                print(ordering2[i])
                break
        #if not flag:
        #    break
    #return flag

#print(checkEquiv(orderings10, orderings100))
checkEquiv(orderings10, orderings100)
'''
graph = graphConv(fileName)
def antiGraph(graph):
    antiG = {}
    for key in graph.keys():
        neighbors = graph[key]
        nonNeigh = []
        for key2 in graph.keys():
            if key2 != key:
                if key2 not in neighbors:
                    nonNeigh.append(key2)
        antiG[key] = nonNeigh
    return antiG
antiG = antiGraph(graph)

def freqGet(orderings):
    freq = {}
    freqarr = []

    for key in orderings.keys():
        if freq.get(orderings[key][0]) == None:
            freq[orderings[key][0]] = 1
        else:
            freq[orderings[key][0]] += 1
    for key in freq.keys():
        freqarr.append((key,freq[key]))
    freqarr = sorted(freqarr, key=itemgetter(1), reverse=True)
    return freqarr

def freqToClique(freqarr):
    clique = []
    for item in freqarr:
        clique.append(item[0])
    return clique
def topExclusive(clique, graph):
    for i in range(len(clique)):
        nonNeigh = []
        neighbors = graph[clique[i]]
        for j in range(len(clique)):
            if j != i:
                if clique[j] not in neighbors:
                    nonNeigh.append(clique[j])
        print("{0}: {1}".format(clique[i], nonNeigh))
def nonNeighClique(clique, antigraph):
    exile = []
    for member in clique:
        antiN = antigraph[member]
        for n in antiN:
            if n not in exile:
                exile.append(n)
    exile = sorted(exile)
    #print(exile)
    return exile
def oddOneOut(cliqueExcluding, clique, graph):
    freqEx = []
    for member in cliqueExcluding:
        count = 0
        for c in clique:
            if member in graph[c]:
                count += 1
        freqEx.append((member,count))
    freqEx = sorted(freqEx, key=itemgetter(1))
    #print(freqEx)
    return freqEx
def revOOO(cliqueExcluding, clique, graph):
    freqEx = []
    for member in clique:
        count = 0
        for member2 in cliqueExcluding:
            if member2 not in graph[member]:
                count += 1
        freqEx.append((member, count))
    freqEx = sorted(freqEx, key=itemgetter(1), reverse=True)
    return freqEx
def intersection(list1, list2):
    ulist = []
    for member in list1:
        if member in list2:
            ulist.append(member)
    for member in list2:
        if member in list1 and member not in ulist:
            ulist.append(member)
    return ulist
def union(list1, list2):
    temp = []
    for item in list2:
        temp.append(item)
    ulist = []
    for member in list1:
        ulist.append(member)
        if member in temp:
            temp.remove(member)
    for member in temp:
        ulist.append(member)
    return ulist
def difList(bigList, littleList):
    temp = []
    for item in bigList:
        if item not in littleList:
            temp.append(item)
    return temp
def delete(toDel, graph):
    for node in toDel:
        neighbors = graph[node]
        for n in neighbors:
            graph[n].remove(node)
        graph.pop(node, None)
    return graph

#main(fileName, 100)
def mapping(graph):
    size = len(graph.keys())
    mapNodes = {}
    i = 1
    for item in graph.keys():
        mapNodes[item] = i
        i += 1
    newMap = {}
    for key in graph.keys():
        temp = []
        neighbors = graph[key]
        for n in neighbors:
            temp.append(mapNodes[n])
        newMap[mapNodes[key]] = temp
    reverseMap = {}
    for key in mapNodes.keys():
        reverseMap[mapNodes[key]] = key
    return newMap, reverseMap
def reverseCliqueMap(mapkey, clique):
    temp = []
    for member in clique:
        temp.append(mapkey[member])
    return temp
def antiNeighbors(clique, graph):
    for i in range(len(clique)-1):
        node1 = clique[i]
        for j in range(i+1, len(clique)):
            node2 = clique[j]
            if node2 not in graph[node1]:
                print(node1)
                print(node2)
        


#topExclusive(freqToClique(freqGet(orderings1)), graph)
#topExclusive(freqToClique(freqGet(antiOrder)), antiG2)
#print(freqGet(antiOrder))
#nonNeighClique([54, 114, 7, 125, 29, 80], antiG)
#nonNeighClique([54, 114, 7, 125, 29, 80, 104, 11, 67, 70, 49], antiG)
#base = [7,9,11,19,25,29,34,40,44,49,54,70,79,80,99,110,114,117,122,125]
#antiNeighbors = [3, 4, 6, 8, 10, 12, 14, 15, 16, 20, 21, 23, 24, 26, 27, 28, 30, 32, 35, 36, 37, 38, 39, 41, 42, 43, 46, 47, 50, 51, 53, 56, 57, 58, 59, 60, 61, 62, 63, 64, 69, 72, 73, 74, 75, 76, 78, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 97, 100, 102, 105, 106, 107, 108, 109, 112, 113, 116, 118, 119, 124]
#toDel = base + antiNeighbors
#graphRedux = {1: [2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 121, 123], 2: [1, 5, 17, 18, 22, 31, 33, 45, 48, 52, 65, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 5: [1, 2, 13, 17, 18, 31, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 121, 123], 13: [1, 5, 17, 18, 22, 31, 33, 45, 52, 55, 66, 67, 68, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 123], 17: [1, 2, 5, 13, 18, 22, 31, 45, 48, 52, 55, 65, 66, 67, 71, 77, 89, 93, 96, 98, 101, 103, 104, 115, 120, 121, 123], 18: [1, 2, 5, 13, 17, 22, 31, 33, 45, 48, 55, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 111, 115, 120, 121, 123], 22: [1, 2, 13, 17, 18, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 123], 31: [1, 2, 5, 13, 17, 18, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 93, 96, 98, 101, 103, 104, 115, 120, 121, 123], 33: [1, 2, 13, 18, 22, 31, 45, 48, 52, 55, 65, 66, 67, 68, 93, 96, 98, 101, 103, 104, 111, 120, 121, 123], 45: [1, 2, 5, 13, 17, 18, 22, 31, 33, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 96, 98, 101, 103, 104, 115, 120, 121, 123], 48: [1, 2, 5, 17, 18, 22, 31, 33, 45, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 104, 111, 115, 120, 121, 123], 52: [1, 2, 5, 13, 17, 22, 31, 33, 45, 55, 65, 66, 67, 68, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 120, 121, 123], 55: [1, 5, 13, 17, 18, 22, 31, 33, 45, 52, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 65: [1, 2, 5, 17, 22, 31, 33, 45, 48, 52, 55, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 115, 120, 123], 66: [1, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 120, 121], 67: [2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 111, 120, 121, 123], 68: [1, 2, 5, 13, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 71, 77, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 71: [1, 2, 5, 17, 18, 22, 31, 45, 48, 55, 65, 66, 67, 68, 77, 89, 93, 96, 98, 101, 104, 111, 115, 120, 121, 123], 77: [1, 2, 5, 13, 17, 18, 22, 31, 45, 48, 52, 55, 65, 66, 67, 68, 71, 89, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 89: [1, 2, 5, 13, 17, 18, 22, 45, 48, 52, 55, 65, 66, 67, 71, 77, 93, 98, 101, 103, 104, 111, 115, 120, 123], 93: [1, 2, 5, 13, 17, 18, 22, 31, 33, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 96, 98, 101, 103, 104, 111, 115, 120, 121, 123], 96: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 93, 98, 103, 104, 111, 120, 121, 123], 98: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 96, 101, 103, 104, 111, 115, 120, 121], 101: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 98, 103, 111, 115, 120, 121, 123], 103: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 52, 55, 65, 66, 67, 68, 77, 89, 93, 96, 98, 101, 104, 111, 115, 120, 121, 123], 104: [1, 2, 5, 13, 17, 22, 31, 33, 45, 48, 52, 55, 65, 66, 67, 68, 71, 77, 89, 93, 96, 98, 103, 111, 115, 120, 121, 123], 111: [1, 2, 5, 13, 18, 22, 33, 48, 52, 55, 66, 67, 68, 71, 77, 89, 93, 96, 98, 101, 103, 104, 115, 120, 121, 123], 115: [1, 2, 5, 13, 17, 18, 22, 31, 45, 48, 52, 55, 65, 68, 71, 77, 89, 93, 98, 101, 103, 104, 111, 120, 121, 123], 120: [1, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 65, 66, 67, 71, 89, 93, 96, 98, 101, 103, 104, 111, 115, 121, 123], 121: [1, 2, 5, 17, 18, 31, 33, 45, 48, 52, 55, 66, 67, 68, 71, 77, 93, 96, 98, 101, 103, 104, 111, 115, 120, 123], 123: [1, 2, 5, 13, 17, 18, 22, 31, 33, 45, 48, 52, 55, 65, 67, 68, 71, 77, 89, 93, 96, 101, 103, 104, 111, 115, 120, 121]}

#base = [29,44,83,85,79,60, 1,2,5,18,31,48,68,71,77,93,101,115,121,82]

#antiNeighbors = nonNeighClique(base, antiG)
#toDel = base + antiNeighbors
#graphRedux = delete(toDel, graph)
#print(graphRedux)
#size of graphRedux is 31; largest clique is of size 14
#testSample(graphRedux, 100, True, 125)

#print(antiG[60])

#print(revOOO(nonNeighClique([96, 35, 9, 99, 44, 45, 52, 117], antiG), [96, 35, 9, 99, 44, 45, 52, 117], graph))

#oddOneOut([3, 6, 10, 12, 14, 16, 24, 27, 30, 41, 43, 46, 51, 53, 58, 61, 64, 73, 76, 78, 81, 83, 86, 91, 92, 94, 95, 100, 106, 108, 119], [54, 114, 7, 125, 29, 80], graph)

'''
graph = graphConv(fileName)

nonNeighClique([54, 114, 7, 125, 29, 80], graph)
#[3, 6, 10, 12, 14, 16, 24, 27, 30, 41, 43, 46, 51, 53, 58, 61, 64, 73, 76, 78, 81, 83, 86, 91, 92, 94, 95, 100, 106, 108, 119]
#[(7, 2), (8, 1), (18, 1), (19, 5), (29, 1), (45, 14), (49, 1), (54, 28), (60, 23), (80, 1), (99, 5), (101, 5), (104, 9), (109, 1), (110, 3), (111, 7), (114, 16), (125, 2)]
#[(54, 28), (60, 23), (114, 16), (45, 14), (104, 9), (111, 7), (19, 5), (99, 5), (101, 5), (110, 3), (7, 2), (125, 2), (8, 1), (18, 1), (29, 1), (49, 1), (80, 1), (109, 1)]

    solution1 = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
    solution2 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 70, 71, 77, 79, 80, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    solution3 = [5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    solution4 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 65, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    solution5 = [5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    solution6 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 65, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    solution7 = [1, 5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    solution8 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    solution9 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 68, 70, 71, 77, 79, 80, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    solution10 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    solution11 = [1, 5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    solution12 = [1, 5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
    solution13 = [1, 5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    solution14 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 93, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
    solution15 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 70, 71, 77, 79, 80, 93, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
    solution16 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
    solution17 = [5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
    solution18 = [5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
    solution19 = [5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    solution20 = [5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    s21 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 93, 99, 101, 110, 111, 114, 115, 117, 121, 122, 123, 125]
    s22 = [5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    s23 = [1, 7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
    s24 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 70, 71, 77, 79, 80, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
    s25 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 68, 70, 71, 77, 79, 80, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
    s26 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    s27 = [1, 5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
    s28 = [1, 7, 9, 11, 13, 19, 22, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
    s29 = [1, 5, 7, 9, 11, 13, 19, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
    s30 = [5, 7, 9, 11, 13, 19, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
    s31 = [5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    s32 = [1, 5, 7, 9, 11, 19, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 121, 122, 125]
    s33 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 70, 71, 77, 79, 80, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s34 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
    s35 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
    s36 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
    s37 = [5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
    s38 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s39 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 85, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s40 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 40, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 83, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123]
    s41 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s42 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 85, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s43 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s44 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s45 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 83, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123]
    s46 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 83, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123]
    s47 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s48 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 85, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s49 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s50 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
    s51 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
    s52 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
    s53 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
    s54 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
    s55 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]

uniqueMembers = [1, 2, 5, 7, 9, 11, 13, 17, 18, 19, 22, 25, 29, 31, 33, 34, 40, 44, 45, 48, 49, 52, 54, 55, 60, 65, 66, 67, 68, 70, 71, 77, 79, 80, 82,
                83, 85, 92, 93, 96, 98, 99, 101, 103, 104, 110, 111, 114, 115, 117, 121, 122, 123, 125]
                #size of uniqueMembers = 54
                #...and that's WITH 3 "slightly" different base-sizes of 20...

...And thinking to test9.4_numpy.py, and how it's deleting things like so:
    [36, 83, 108, 76, 51, 90, 64, 15, 95, 68, 42, 97, 88, 16, 94, 75, 102, 55, 43, 27, 33, 112, 87, 56, 21, 100, 73, 121, 107, 105, 3, 14...
    (assumed that the rest is like the vector_projection method, so:)
    ... 113, 61, 109, 50, 124, 84, 72, 32, 37, 4, 57, 78, 20, 116, 28, 12, 53, 120, 106, 65, 98, 46, 115, 23, 62, 30, 118, 74, 52, 39, 111,
    6, 58, 103, 91, 89, 38, 86, 63, 81, 119, 10, 49, 92, 26]
...The very 2nd deleted node is part of a valid 34-clique...

Because, I mean, it's not an exaggeration to say if I just keep narrowing the graph by building the clique-solution from the ground up,
based off of which nodes in the graph are most heavily "pointed" towards, based off linear-algebraic approximations... 
'''

#'''
very_first_ordering = {1: [114, 99, 104, 54, 60, 29, 49, 98, 8, 19, 7, 110, 45, 111, 39, 40, 125, 80, 101, 48, 34, 10, 92, 22, 11, 122, 44, 115, 9, 2, 41, 96, 82, 123, 18, 70, 47, 24, 66, 26, 84, 107, 5, 57, 79, 25, 30, 46, 85, 119, 59, 65, 77, 14, 55, 93, 43, 121, 13, 58, 31, 91, 117, 6, 52, 74, 86, 103, 32, 61, 124, 20, 23, 72, 35, 69, 4, 81, 63, 28, 27, 62, 3, 33, 17, 68, 116, 38, 78, 89, 102, 71, 120, 15, 88, 94, 100, 21, 105, 112, 113, 50, 75, 51, 73, 16, 97, 76, 64, 90, 36, 83], 
2: [54, 60, 114, 19, 80, 110, 45, 104, 111, 77, 29, 40, 67, 11, 8, 7, 101, 99, 35, 91, 74, 65, 34, 92, 24, 39, 70, 49, 98, 125, 84, 44, 57, 48, 119, 22, 59, 10, 123, 18, 47, 1, 79, 115, 6, 41, 86, 96, 82, 71, 32, 121, 122, 58, 5, 31, 72, 109, 117, 118, 9, 25, 52, 30, 46, 85, 61, 93, 124, 26, 37, 106, 107, 4, 69, 81, 100, 63, 78, 102, 87, 23, 27, 43, 62, 116, 14, 38, 21, 89, 112, 105, 28, 97, 94, 113, 12, 17, 68, 103, 108, 16, 88, 95, 50, 33, 51, 42, 76, 73, 64, 36, 83], 
3: [111, 54, 45, 60, 114, 67, 19, 101, 7, 104, 82, 39, 18, 29, 40, 49, 125, 8, 99, 110, 4, 6, 25, 96, 10, 123, 47, 70, 24, 11, 98, 61, 20, 66, 106, 44, 58, 1, 13, 118, 46, 74, 86, 119, 59, 92, 53, 121, 5, 91, 69, 72, 79, 9, 30, 41, 48, 85, 77, 14, 32, 26, 23, 27, 62, 52, 33, 55, 28, 120, 37, 84, 107, 43, 122, 76, 109, 17, 103, 81, 63, 21, 112, 87, 93, 64, 113, 68, 38, 105, 71, 124, 15, 12, 50, 100, 78, 102, 73, 97, 116, 75, 51, 90, 88, 95, 36], 
4: [45, 111, 60, 54, 114, 19, 80, 110, 104, 39, 29, 49, 40, 125, 8, 99, 7, 101, 10, 47, 24, 28, 67, 11, 98, 109, 6, 70, 18, 62, 23, 106, 91, 35, 9, 30, 46, 52, 86, 119, 59, 65, 82, 92, 123, 89, 27, 37, 121, 5, 72, 13, 57, 58, 117, 118, 2, 74, 41, 96, 61, 87, 32, 120, 53, 107, 26, 43, 44, 1, 79, 115, 3, 100, 103, 55, 63, 38, 78, 105, 21, 56, 124, 66, 84, 20, 122, 64, 50, 12, 17, 116, 14, 112, 71, 42, 16, 33, 81, 102, 73, 94, 113, 51, 90, 15, 88, 95, 75, 97, 36, 108], 
5: [99, 104, 54, 60, 114, 70, 98, 125, 19, 7, 45, 111, 10, 40, 11, 8, 80, 101, 110, 25, 39, 18, 24, 47, 49, 67, 29, 66, 107, 31, 91, 9, 41, 48, 77, 82, 92, 26, 57, 44, 1, 79, 115, 13, 52, 46, 85, 2, 30, 86, 96, 119, 59, 65, 34, 123, 62, 37, 122, 58, 69, 72, 117, 118, 4, 35, 6, 61, 32, 53, 106, 121, 109, 116, 63, 55, 14, 21, 112, 71, 56, 93, 124, 43, 94, 50, 12, 17, 100, 102, 89, 105, 87, 120, 3, 81, 78, 51, 64, 16, 113, 68, 103, 73, 90, 15, 76, 75, 42, 88, 36, 83], 
6: [111, 7, 54, 60, 24, 19, 101, 45, 18, 67, 49, 11, 29, 99, 8, 80, 110, 77, 47, 40, 98, 13, 46, 119, 41, 22, 59, 10, 39, 43, 4, 2, 25, 86, 96, 34, 92, 123, 82, 27, 121, 106, 69, 57, 58, 72, 91, 117, 35, 79, 109, 48, 52, 85, 9, 74, 30, 12, 38, 112, 71, 28, 87, 20, 23, 26, 107, 122, 84, 5, 44, 1, 115, 118, 89, 61, 124, 32, 93, 66, 37, 53, 62, 3, 68, 81, 100, 103, 63, 21, 55, 56, 120, 94, 17, 102, 64, 42, 33, 50, 116, 51, 73, 90, 95, 113, 76, 88, 16, 97, 75, 36, 108], 
7: [54, 114, 45, 60, 80, 8, 104, 111, 29, 98, 11, 99, 110, 19, 101, 24, 18, 49, 40, 67, 125, 13, 6, 34, 65, 123, 70, 39, 47, 35, 25, 41, 119, 9, 92, 77, 22, 59, 82, 10, 5, 79, 44, 115, 117, 1, 46, 48, 52, 74, 85, 96, 2, 21, 28, 71, 87, 124, 84, 66, 20, 53, 106, 121, 122, 107, 72, 91, 57, 69, 109, 118, 4, 31, 68, 105, 112, 32, 93, 120, 62, 26, 37, 23, 27, 43, 50, 33, 38, 55, 63, 89, 61, 56, 12, 17, 102, 73, 42, 64, 3, 116, 103, 100, 90, 88, 94, 113, 75, 51, 95, 97, 15, 16, 36, 76, 83], 
8: [114, 104, 54, 60, 19, 80, 7, 45, 111, 98, 101, 99, 18, 29, 67, 11, 49, 125, 30, 10, 22, 34, 65, 92, 123, 24, 39, 47, 70, 43, 9, 46, 74, 85, 119, 59, 82, 77, 37, 53, 13, 57, 69, 91, 117, 1, 109, 25, 48, 41, 2, 6, 52, 86, 96, 56, 124, 20, 26, 27, 84, 106, 107, 122, 35, 72, 5, 31, 4, 79, 115, 118, 78, 89, 102, 23, 62, 66, 68, 38, 21, 105, 112, 28, 32, 61, 71, 87, 93, 120, 50, 33, 14, 55, 63, 90, 64, 16, 88, 17, 81, 103, 100, 116, 15, 42, 97, 3, 75, 76, 51, 73, 94, 95, 108, 83, 36], 
9: [114, 104, 60, 54, 67, 11, 8, 19, 80, 99, 110, 7, 45, 111, 49, 40, 29, 98, 101, 52, 85, 41, 34, 92, 47, 125, 13, 86, 82, 123, 39, 18, 70, 24, 27, 121, 5, 58, 1, 79, 115, 25, 30, 59, 10, 22, 65, 77, 87, 62, 66, 20, 37, 23, 84, 57, 44, 72, 4, 117, 118, 2, 6, 48, 96, 119, 33, 63, 28, 93, 120, 43, 107, 31, 35, 91, 109, 68, 17, 55, 14, 38, 56, 61, 71, 53, 106, 122, 50, 100, 78, 89, 102, 105, 15, 81, 103, 21, 112, 51, 16, 94, 97, 113, 3, 42, 95, 75, 73, 90, 83, 76, 36], 
10: [19, 114, 60, 125, 8, 99, 104, 111, 45, 98, 101, 110, 7, 80, 91, 46, 47, 24, 70, 49, 11, 40, 67, 5, 119, 86, 30, 22, 65, 77, 18, 39, 106, 107, 4, 1, 13, 69, 2, 6, 48, 52, 74, 123, 82, 92, 59, 53, 26, 43, 115, 117, 109, 58, 57, 79, 35, 9, 25, 96, 85, 41, 100, 61, 23, 66, 122, 62, 37, 31, 72, 44, 78, 89, 55, 124, 32, 27, 84, 103, 116, 17, 14, 21, 63, 120, 87, 93, 28, 56, 94, 3, 12, 112, 105, 102, 38, 113, 88, 42, 50, 68, 81, 75, 73, 15, 64, 97, 16, 76, 51, 108, 90, 36], 
11: [45, 114, 54, 60, 98, 67, 101, 19, 7, 104, 111, 125, 40, 49, 110, 80, 99, 8, 39, 70, 24, 29, 85, 9, 52, 92, 123, 77, 65, 82, 18, 47, 117, 44, 79, 96, 2, 6, 74, 119, 41, 48, 34, 10, 22, 107, 62, 118, 72, 109, 5, 13, 57, 58, 69, 25, 30, 46, 38, 120, 32, 122, 27, 66, 37, 91, 35, 115, 1, 4, 31, 81, 89, 28, 87, 71, 56, 106, 121, 23, 53, 84, 20, 43, 33, 102, 21, 63, 78, 105, 93, 61, 100, 103, 116, 12, 17, 55, 14, 75, 73, 94, 113, 95, 50, 68, 88, 16, 42, 64, 3, 90, 36, 83, 76, 108], 
12: [110, 60, 80, 45, 54, 29, 67, 99, 101, 7, 111, 6, 119, 70, 39, 18, 24, 11, 40, 125, 121, 19, 13, 77, 47, 49, 98, 52, 86, 96, 123, 10, 65, 82, 71, 120, 53, 37, 5, 31, 109, 4, 25, 30, 41, 46, 85, 2, 74, 22, 34, 59, 17, 87, 32, 124, 62, 106, 107, 57, 58, 72, 91, 44, 79, 115, 117, 118, 35, 69, 48, 50, 38, 105, 56, 23, 122, 26, 66, 116, 81, 63, 112, 21, 93, 28, 27, 20, 64, 14, 61, 113, 16, 55, 51, 94, 3, 103, 100, 73, 90, 42, 95, 97, 108, 75, 36, 83], 
13: [7, 111, 54, 60, 114, 8, 19, 99, 110, 45, 104, 18, 29, 67, 11, 80, 101, 41, 52, 6, 9, 10, 47, 39, 70, 24, 40, 49, 98, 125, 30, 59, 92, 22, 123, 43, 53, 5, 58, 25, 46, 85, 86, 119, 34, 77, 82, 28, 87, 120, 66, 20, 37, 106, 107, 44, 4, 57, 1, 35, 69, 72, 79, 115, 109, 74, 96, 12, 63, 21, 56, 62, 23, 122, 31, 118, 91, 117, 33, 81, 55, 89, 32, 93, 26, 27, 3, 50, 68, 103, 17, 38, 112, 105, 61, 15, 64, 116, 100, 90, 75, 51, 73, 42, 94, 95, 113, 16, 97, 36, 76, 83], 
14: [114, 45, 104, 25, 60, 54, 29, 19, 80, 110, 101, 31, 115, 79, 111, 96, 82, 40, 49, 67, 125, 8, 99, 1, 30, 85, 86, 34, 65, 39, 70, 11, 98, 9, 48, 10, 22, 59, 92, 77, 123, 18, 24, 47, 121, 43, 122, 5, 58, 41, 2, 46, 52, 74, 84, 20, 4, 117, 72, 109, 119, 81, 78, 61, 32, 56, 124, 62, 27, 66, 107, 44, 57, 69, 91, 118, 88, 3, 103, 33, 68, 38, 55, 112, 102, 120, 23, 37, 26, 53, 116, 17, 50, 100, 21, 105, 51, 90, 94, 12, 15, 95, 113, 16, 76, 75, 97, 83, 36], 
15: [104, 114, 54, 99, 45, 60, 125, 49, 19, 8, 101, 111, 70, 40, 98, 110, 7, 80, 44, 115, 118, 9, 48, 34, 59, 82, 18, 39, 29, 13, 1, 74, 85, 10, 65, 123, 92, 24, 47, 120, 107, 43, 117, 35, 31, 69, 30, 25, 46, 41, 52, 86, 22, 33, 103, 63, 28, 87, 20, 26, 53, 5, 4, 58, 72, 21, 112, 78, 89, 105, 23, 27, 37, 106, 62, 66, 84, 91, 100, 81, 14, 102, 55, 61, 56, 93, 121, 122, 3, 68, 113, 97, 50, 108, 76, 90, 73, 88, 95, 17, 116, 36, 42, 16, 64, 83], 
16: [110, 45, 114, 8, 57, 104, 85, 60, 54, 111, 70, 11, 99, 19, 7, 30, 22, 34, 65, 24, 39, 47, 67, 98, 125, 49, 28, 31, 35, 4, 96, 119, 2, 9, 86, 92, 123, 10, 82, 18, 78, 62, 106, 27, 109, 115, 117, 5, 69, 48, 52, 41, 46, 59, 77, 38, 121, 122, 37, 66, 84, 58, 91, 1, 6, 50, 105, 63, 32, 56, 71, 107, 20, 26, 79, 118, 13, 81, 12, 17, 55, 102, 14, 93, 61, 87, 120, 124, 43, 100, 75, 33, 51, 73, 113, 42, 88, 116, 95, 97, 94, 90, 15, 83, 108, 36], 
17: [54, 114, 110, 19, 101, 45, 92, 123, 24, 125, 80, 99, 7, 104, 52, 86, 65, 82, 18, 98, 11, 8, 46, 9, 41, 10, 34, 70, 39, 47, 67, 29, 40, 49, 93, 26, 44, 115, 25, 77, 71, 62, 106, 107, 69, 5, 91, 1, 4, 13, 118, 113, 6, 2, 85, 119, 59, 22, 12, 61, 120, 66, 37, 122, 27, 84, 23, 121, 58, 79, 117, 31, 48, 74, 96, 63, 55, 89, 28, 87, 57, 35, 109, 14, 21, 124, 32, 53, 20, 3, 81, 103, 100, 78, 105, 16, 38, 112, 36, 94, 76, 51, 73, 90, 75, 95, 15, 108], 
18: [54, 111, 60, 114, 8, 7, 19, 99, 101, 110, 45, 49, 125, 80, 123, 24, 47, 11, 29, 67, 40, 98, 44, 6, 48, 74, 59, 92, 34, 21, 70, 39, 26, 13, 72, 117, 41, 46, 10, 22, 77, 82, 43, 5, 31, 91, 118, 9, 2, 25, 86, 85, 96, 119, 71, 93, 20, 53, 23, 121, 58, 1, 4, 35, 69, 79, 109, 115, 50, 68, 38, 56, 61, 32, 120, 66, 37, 84, 106, 107, 122, 17, 12, 116, 63, 105, 112, 28, 87, 124, 3, 81, 103, 14, 55, 78, 89, 97, 33, 100, 51, 64, 15, 42, 88, 16, 95, 94, 75, 73, 90, 108, 76, 83, 36], 
19: [114, 54, 8, 80, 111, 104, 45, 47, 29, 67, 98, 11, 125, 110, 99, 101, 7, 22, 10, 39, 18, 24, 70, 40, 49, 91, 92, 123, 35, 6, 9, 30, 41, 52, 2, 46, 74, 85, 86, 65, 34, 59, 77, 82, 26, 107, 69, 5, 117, 1, 4, 13, 44, 58, 79, 96, 25, 48, 119, 56, 66, 43, 121, 122, 57, 115, 31, 72, 109, 118, 120, 87, 62, 23, 27, 37, 53, 84, 68, 100, 103, 17, 55, 14, 38, 93, 28, 71, 32, 42, 33, 116, 81, 63, 21, 112, 78, 102, 89, 105, 94, 3, 12, 50, 64, 95, 15, 51, 16, 88, 97, 75, 76, 73, 90, 36, 108], 
20: [54, 60, 45, 114, 67, 8, 7, 104, 111, 123, 29, 49, 125, 40, 80, 99, 110, 101, 22, 59, 77, 47, 18, 24, 98, 11, 66, 23, 118, 115, 30, 41, 96, 9, 82, 34, 33, 39, 84, 13, 58, 69, 79, 6, 25, 65, 92, 120, 43, 106, 53, 27, 1, 57, 117, 31, 46, 48, 52, 85, 86, 119, 55, 38, 56, 61, 124, 28, 71, 121, 26, 44, 72, 4, 109, 3, 14, 32, 37, 62, 122, 63, 112, 87, 93, 50, 68, 81, 78, 102, 21, 89, 105, 90, 95, 15, 42, 64, 97, 103, 12, 116, 17, 83, 75, 51, 16, 73, 88, 76, 36], 
21: [18, 7, 54, 114, 99, 104, 111, 60, 40, 29, 80, 8, 101, 45, 74, 123, 11, 49, 98, 125, 19, 44, 47, 67, 13, 52, 10, 34, 92, 70, 39, 112, 23, 5, 117, 4, 6, 41, 48, 119, 2, 46, 65, 77, 82, 22, 59, 120, 28, 53, 69, 72, 31, 91, 109, 115, 35, 118, 9, 96, 68, 93, 26, 121, 37, 43, 58, 1, 79, 64, 50, 103, 63, 89, 105, 56, 32, 61, 124, 84, 106, 107, 122, 20, 12, 17, 33, 100, 14, 87, 88, 15, 97, 3, 116, 55, 90, 113, 81, 76, 51, 95, 42, 108, 73, 94, 36, 75, 83], 
22: [54, 19, 80, 111, 60, 114, 8, 110, 45, 104, 29, 40, 98, 99, 101, 7, 34, 65, 39, 70, 47, 24, 49, 67, 11, 125, 69, 25, 41, 48, 10, 77, 123, 18, 35, 1, 52, 6, 74, 2, 46, 85, 59, 82, 92, 53, 27, 43, 20, 57, 79, 13, 115, 117, 30, 96, 9, 86, 119, 89, 28, 56, 120, 66, 26, 84, 122, 107, 72, 58, 44, 109, 118, 32, 87, 93, 62, 95, 33, 68, 103, 38, 78, 102, 14, 71, 124, 94, 50, 100, 116, 55, 63, 21, 105, 112, 42, 88, 16, 81, 12, 17, 83, 97, 113, 76, 73, 90, 64, 15, 108, 75, 51, 36], 
23: [125, 49, 99, 101, 111, 114, 54, 60, 67, 80, 110, 104, 45, 98, 40, 29, 8, 7, 19, 77, 59, 82, 47, 18, 11, 20, 4, 9, 74, 123, 10, 34, 39, 24, 84, 37, 118, 58, 96, 6, 85, 46, 48, 119, 52, 30, 92, 65, 63, 121, 66, 27, 91, 117, 72, 79, 1, 44, 13, 35, 31, 86, 2, 55, 21, 120, 124, 61, 28, 106, 122, 53, 62, 43, 26, 115, 109, 57, 89, 112, 38, 93, 71, 68, 116, 103, 33, 17, 78, 87, 56, 51, 64, 3, 12, 105, 14, 94, 42, 81, 88, 97, 15, 108, 76, 73, 113, 75, 90, 83, 36], 
24: [54, 45, 60, 110, 7, 19, 104, 47, 29, 125, 11, 8, 99, 101, 80, 6, 86, 92, 18, 40, 67, 98, 49, 25, 34, 123, 10, 22, 65, 77, 82, 39, 70, 117, 119, 2, 30, 41, 46, 52, 85, 59, 66, 27, 26, 5, 13, 35, 57, 118, 4, 31, 44, 69, 9, 74, 96, 48, 32, 61, 121, 20, 53, 106, 109, 115, 1, 58, 72, 79, 17, 68, 112, 55, 102, 93, 71, 23, 43, 84, 107, 122, 37, 12, 105, 38, 28, 56, 124, 87, 33, 100, 103, 116, 81, 14, 89, 63, 90, 88, 113, 3, 50, 42, 94, 16, 64, 51, 15, 95, 36, 75, 76, 73, 108], 
25: [114, 80, 101, 110, 104, 54, 60, 29, 99, 7, 45, 111, 85, 34, 24, 40, 98, 8, 19, 115, 77, 22, 70, 11, 67, 49, 125, 5, 79, 46, 48, 74, 86, 82, 123, 47, 39, 18, 14, 69, 72, 31, 6, 9, 30, 41, 96, 119, 65, 10, 59, 92, 112, 107, 44, 13, 57, 1, 52, 2, 102, 93, 61, 32, 62, 66, 53, 20, 84, 122, 27, 121, 58, 91, 117, 118, 109, 68, 105, 71, 120, 26, 37, 43, 106, 3, 17, 81, 100, 55, 89, 87, 56, 124, 76, 113, 12, 50, 103, 51, 90, 95, 88, 33, 116, 15, 94, 97, 108, 73, 83, 75, 36], 
26: [60, 125, 19, 54, 114, 18, 8, 101, 110, 99, 45, 104, 111, 123, 24, 40, 80, 7, 46, 48, 59, 92, 10, 39, 47, 29, 49, 67, 5, 44, 1, 74, 86, 65, 82, 22, 70, 61, 66, 69, 117, 91, 118, 41, 2, 6, 30, 119, 77, 34, 62, 43, 106, 122, 31, 58, 4, 35, 57, 25, 85, 96, 17, 116, 55, 28, 93, 124, 20, 23, 84, 13, 115, 109, 103, 63, 38, 71, 32, 53, 107, 121, 50, 21, 112, 102, 120, 87, 75, 3, 68, 12, 100, 14, 78, 89, 105, 51, 42, 64, 15, 88, 33, 81, 73, 16, 94, 76, 90, 36, 108], 
27: [111, 104, 45, 114, 60, 54, 110, 101, 8, 30, 24, 98, 11, 40, 49, 80, 99, 19, 7, 9, 52, 22, 47, 29, 67, 6, 41, 82, 123, 34, 65, 39, 70, 118, 79, 109, 4, 85, 86, 119, 25, 46, 48, 77, 92, 59, 10, 87, 56, 66, 23, 53, 20, 43, 117, 69, 58, 72, 96, 2, 102, 28, 84, 107, 37, 91, 115, 1, 13, 35, 97, 89, 55, 38, 124, 71, 61, 32, 122, 106, 121, 17, 33, 112, 78, 63, 14, 120, 88, 95, 16, 3, 100, 103, 68, 105, 90, 113, 81, 12, 51, 73, 94, 64, 15, 76, 42, 83, 108, 75, 36], 
28: [101, 45, 114, 7, 110, 80, 104, 111, 54, 60, 40, 99, 4, 11, 29, 98, 19, 8, 35, 22, 89, 39, 70, 49, 67, 125, 13, 115, 109, 6, 9, 52, 119, 34, 65, 123, 18, 24, 47, 46, 74, 30, 41, 86, 92, 10, 59, 82, 77, 87, 23, 26, 20, 27, 84, 1, 57, 58, 117, 69, 118, 2, 21, 63, 93, 37, 53, 62, 66, 122, 106, 91, 44, 16, 78, 105, 61, 124, 32, 71, 107, 121, 43, 73, 17, 50, 100, 33, 103, 55, 120, 56, 75, 15, 94, 3, 12, 81, 102, 95, 113, 42, 68, 116, 51, 88, 97, 90, 36, 83], 
29: [54, 80, 45, 60, 114, 40, 19, 101, 110, 7, 99, 104, 111, 39, 121, 8, 47, 24, 49, 67, 11, 98, 125, 25, 119, 59, 22, 34, 82, 70, 18, 122, 44, 72, 31, 1, 117, 9, 6, 2, 52, 74, 96, 86, 65, 92, 77, 123, 57, 91, 13, 79, 118, 4, 35, 41, 30, 48, 85, 68, 23, 84, 43, 20, 5, 58, 69, 109, 115, 14, 21, 38, 89, 112, 105, 28, 71, 93, 56, 27, 26, 37, 62, 66, 107, 12, 50, 103, 55, 87, 61, 120, 124, 32, 33, 81, 63, 102, 95, 3, 17, 100, 116, 90, 42, 64, 94, 83, 51, 88, 97, 15, 113, 75, 36, 108], 
30: [60, 8, 104, 111, 114, 19, 101, 110, 45, 80, 99, 10, 24, 11, 29, 67, 40, 49, 98, 125, 27, 57, 58, 52, 86, 119, 65, 82, 123, 47, 39, 70, 62, 43, 13, 91, 79, 9, 25, 41, 46, 85, 59, 77, 22, 34, 92, 102, 56, 66, 20, 5, 1, 4, 72, 115, 117, 109, 6, 2, 48, 74, 96, 14, 38, 78, 32, 53, 26, 23, 106, 107, 31, 69, 35, 118, 33, 89, 28, 93, 120, 124, 37, 84, 122, 121, 88, 81, 105, 61, 87, 16, 12, 116, 100, 103, 55, 63, 112, 64, 42, 94, 97, 3, 75, 73, 15, 51, 90, 95, 76, 36, 83], 
31: [45, 54, 114, 80, 104, 96, 60, 29, 67, 125, 110, 82, 65, 70, 49, 40, 19, 8, 99, 101, 7, 85, 34, 18, 24, 39, 98, 11, 5, 35, 117, 48, 25, 59, 77, 92, 123, 47, 14, 93, 57, 79, 69, 115, 72, 41, 74, 86, 2, 119, 10, 106, 121, 1, 44, 91, 9, 46, 52, 30, 62, 20, 23, 26, 43, 53, 122, 37, 58, 13, 118, 109, 50, 68, 38, 112, 56, 61, 124, 66, 84, 107, 12, 33, 116, 55, 78, 21, 105, 32, 71, 120, 16, 17, 81, 103, 63, 102, 15, 100, 108, 51, 42, 64, 88, 97, 95, 83, 90, 113, 75, 36], 
32: [114, 54, 60, 11, 99, 101, 7, 104, 111, 45, 77, 39, 47, 24, 40, 125, 8, 110, 80, 19, 79, 74, 2, 18, 70, 29, 98, 49, 115, 118, 69, 119, 25, 30, 41, 86, 10, 22, 59, 92, 5, 1, 4, 58, 6, 96, 46, 48, 123, 65, 102, 124, 107, 91, 35, 57, 72, 109, 13, 85, 52, 38, 89, 71, 27, 37, 106, 20, 26, 66, 44, 31, 88, 12, 81, 55, 112, 14, 105, 87, 28, 61, 93, 53, 43, 84, 121, 122, 73, 94, 97, 113, 42, 3, 50, 103, 21, 78, 56, 95, 33, 116, 17, 100, 16, 64, 90, 76, 36, 83], 
33: [54, 60, 67, 114, 7, 104, 111, 45, 123, 11, 20, 8, 19, 9, 70, 29, 80, 99, 110, 101, 41, 85, 96, 30, 82, 22, 39, 24, 40, 98, 125, 49, 58, 118, 13, 52, 59, 34, 92, 47, 18, 43, 84, 79, 1, 31, 35, 69, 74, 119, 65, 120, 23, 66, 121, 27, 44, 4, 72, 117, 109, 25, 2, 6, 46, 48, 88, 50, 14, 87, 28, 56, 61, 53, 106, 122, 57, 15, 97, 3, 68, 100, 38, 63, 78, 102, 21, 105, 93, 32, 37, 62, 26, 81, 103, 55, 112, 90, 64, 42, 73, 95, 75, 51, 16, 113, 83, 76], 
34: [80, 54, 114, 60, 8, 7, 110, 45, 104, 111, 29, 40, 67, 98, 19, 101, 99, 117, 25, 22, 82, 24, 47, 18, 11, 49, 125, 115, 9, 85, 2, 77, 59, 65, 123, 39, 70, 107, 31, 1, 57, 79, 86, 46, 48, 92, 93, 122, 72, 35, 69, 109, 6, 41, 30, 52, 74, 96, 119, 71, 27, 23, 37, 20, 84, 121, 5, 44, 13, 91, 68, 14, 63, 112, 28, 56, 61, 120, 124, 26, 43, 53, 66, 106, 17, 81, 100, 89, 21, 105, 87, 97, 113, 33, 50, 103, 116, 38, 55, 78, 102, 16, 88, 15, 12, 108, 90, 51, 95, 94, 83, 76, 73, 36], 
35: [45, 80, 54, 114, 19, 110, 7, 60, 67, 104, 111, 2, 70, 29, 99, 8, 101, 65, 77, 22, 39, 47, 24, 11, 40, 49, 98, 125, 91, 31, 96, 119, 82, 92, 10, 34, 123, 18, 63, 28, 66, 117, 4, 6, 52, 74, 86, 59, 53, 84, 106, 58, 5, 44, 57, 69, 72, 118, 13, 42, 85, 9, 41, 46, 48, 30, 89, 87, 93, 23, 122, 26, 107, 1, 109, 78, 38, 55, 56, 71, 124, 32, 62, 43, 27, 37, 121, 116, 50, 68, 33, 21, 102, 112, 105, 108, 64, 16, 100, 12, 75, 73, 97, 15, 17, 103, 51, 88, 95, 113, 36, 83], 
36: [54, 114, 49, 40, 19, 7, 45, 104, 59, 77, 47, 24, 29, 125, 11, 80, 99, 101, 110, 111, 118, 82, 92, 39, 98, 8, 66, 44, 58, 69, 35, 52, 48, 86, 2, 96, 10, 22, 67, 93, 84, 72, 13, 117, 6, 46, 30, 65, 123, 34, 18, 70, 17, 112, 120, 37, 53, 26, 106, 5, 1, 25, 9, 74, 119, 68, 116, 63, 89, 87, 124, 23, 20, 43, 121, 62, 107, 57, 31, 115, 4, 41, 21, 55, 28, 32, 27, 15, 12, 38, 14, 102, 42, 95, 113, 64, 94, 3, 108, 16, 83, 76], 
37: [60, 125, 8, 111, 54, 114, 99, 45, 104, 70, 67, 11, 49, 19, 80, 7, 101, 92, 59, 47, 29, 72, 109, 9, 52, 119, 65, 10, 34, 123, 18, 39, 24, 71, 23, 106, 121, 5, 4, 57, 13, 79, 118, 2, 74, 85, 86, 77, 82, 87, 58, 91, 31, 69, 6, 25, 30, 46, 96, 55, 61, 56, 27, 62, 53, 66, 122, 44, 35, 115, 117, 12, 63, 112, 102, 28, 32, 93, 120, 124, 20, 84, 107, 17, 116, 38, 21, 105, 113, 68, 50, 81, 14, 78, 89, 51, 73, 64, 16, 97, 3, 33, 100, 90, 15, 42, 108, 76, 94, 36, 83], 
38: [45, 60, 11, 101, 110, 54, 114, 29, 8, 7, 19, 80, 111, 18, 47, 40, 67, 49, 125, 6, 30, 85, 77, 123, 24, 39, 70, 58, 79, 117, 9, 48, 96, 74, 119, 22, 59, 82, 20, 31, 57, 72, 4, 35, 109, 2, 41, 52, 86, 10, 92, 34, 65, 78, 32, 26, 27, 66, 23, 62, 13, 91, 1, 69, 115, 89, 102, 71, 37, 43, 53, 84, 121, 122, 44, 118, 12, 14, 63, 56, 120, 124, 106, 107, 16, 42, 95, 33, 50, 68, 81, 103, 61, 87, 93, 75, 64, 55, 94, 3, 17, 116, 100, 73, 97, 90, 88, 83, 36], 
39: [45, 60, 54, 114, 29, 19, 80, 99, 110, 104, 111, 47, 11, 49, 98, 7, 8, 101, 70, 67, 40, 125, 119, 22, 65, 18, 24, 53, 107, 72, 1, 4, 2, 86, 10, 34, 92, 123, 5, 44, 58, 13, 118, 31, 35, 6, 25, 41, 9, 30, 46, 48, 52, 74, 85, 96, 105, 32, 62, 66, 26, 121, 57, 69, 91, 79, 109, 115, 89, 28, 120, 93, 20, 23, 27, 43, 37, 50, 12, 100, 55, 63, 14, 38, 71, 87, 61, 124, 73, 3, 17, 68, 81, 103, 33, 21, 78, 102, 112, 90, 42, 94, 97, 113, 116, 15, 16, 95, 64, 75, 88, 51, 83, 36, 108], 
40: [101, 110, 45, 54, 60, 114, 29, 80, 104, 111, 11, 98, 19, 7, 99, 59, 70, 49, 125, 67, 34, 22, 82, 18, 24, 39, 47, 44, 72, 79, 2, 9, 25, 96, 41, 48, 86, 10, 92, 65, 77, 123, 43, 5, 1, 4, 31, 58, 57, 69, 109, 117, 118, 52, 6, 30, 46, 119, 89, 28, 20, 26, 27, 23, 53, 62, 66, 121, 122, 35, 13, 91, 115, 103, 21, 32, 93, 71, 124, 107, 84, 106, 68, 81, 14, 38, 55, 78, 105, 63, 56, 87, 61, 120, 12, 102, 95, 64, 113, 3, 33, 50, 17, 100, 116, 51, 90, 15, 88, 97, 76, 94, 42, 36, 108, 83], 
41: [110, 45, 60, 54, 114, 7, 19, 99, 104, 11, 40, 67, 98, 8, 80, 101, 9, 22, 123, 18, 24, 29, 69, 13, 6, 48, 52, 59, 65, 39, 47, 87, 43, 107, 5, 58, 1, 25, 2, 30, 74, 85, 119, 10, 34, 77, 82, 92, 20, 27, 44, 57, 31, 79, 115, 109, 118, 46, 96, 86, 100, 120, 32, 26, 66, 53, 84, 4, 35, 72, 117, 91, 17, 33, 63, 61, 28, 56, 93, 62, 106, 121, 122, 95, 50, 81, 14, 21, 38, 112, 105, 71, 12, 103, 78, 89, 94, 88, 3, 68, 51, 15, 16, 97, 75, 73, 90, 64, 42, 113, 83, 76, 36], 
42: [19, 60, 114, 47, 67, 80, 7, 99, 35, 45, 111, 54, 125, 8, 101, 104, 39, 11, 29, 98, 110, 59, 77, 10, 22, 65, 18, 70, 24, 40, 49, 106, 58, 72, 91, 118, 4, 6, 30, 52, 74, 86, 56, 32, 53, 66, 84, 69, 79, 117, 9, 2, 48, 119, 96, 92, 89, 38, 102, 87, 23, 26, 20, 43, 31, 57, 13, 109, 41, 85, 64, 50, 55, 124, 28, 120, 62, 121, 37, 5, 44, 95, 68, 33, 116, 105, 71, 93, 27, 94, 103, 21, 78, 112, 73, 12, 100, 16, 88, 97, 108, 75, 90, 15, 36, 76], 
43: [60, 8, 111, 54, 114, 45, 104, 40, 19, 80, 110, 18, 29, 98, 99, 7, 101, 6, 30, 41, 10, 22, 59, 67, 11, 49, 117, 13, 52, 96, 82, 123, 39, 47, 24, 72, 1, 9, 86, 46, 48, 34, 77, 56, 84, 20, 26, 27, 4, 31, 44, 69, 79, 91, 115, 109, 88, 2, 74, 25, 85, 68, 78, 14, 87, 62, 66, 53, 23, 106, 122, 107, 5, 35, 58, 118, 57, 33, 89, 102, 120, 95, 50, 116, 103, 100, 38, 55, 112, 21, 32, 28, 61, 124, 64, 15, 81, 63, 105, 51, 42, 97, 3, 75, 90, 94, 76, 16, 83, 36, 108], 
44: [54, 110, 114, 60, 18, 49, 11, 29, 40, 19, 80, 99, 7, 101, 45, 104, 111, 67, 125, 2, 74, 39, 24, 98, 1, 118, 58, 48, 85, 96, 82, 77, 92, 47, 70, 93, 66, 26, 84, 5, 91, 117, 9, 25, 41, 52, 10, 22, 34, 59, 65, 123, 21, 62, 122, 121, 31, 72, 115, 13, 35, 6, 46, 86, 55, 23, 43, 107, 106, 4, 69, 79, 17, 103, 116, 78, 71, 87, 20, 37, 15, 3, 50, 68, 81, 63, 89, 105, 112, 28, 61, 32, 120, 113, 12, 33, 38, 14, 102, 100, 76, 64, 94, 95, 36, 75, 73, 88, 42, 51, 108, 83], 
45: [54, 114, 60, 80, 110, 7, 101, 104, 70, 11, 29, 40, 67, 19, 8, 99, 65, 24, 39, 98, 49, 125, 31, 35, 82, 123, 18, 47, 4, 41, 85, 96, 119, 34, 10, 22, 59, 77, 92, 62, 53, 72, 69, 115, 117, 109, 6, 9, 25, 2, 30, 52, 74, 86, 46, 48, 28, 87, 27, 20, 43, 84, 106, 5, 44, 57, 79, 1, 13, 58, 91, 118, 38, 89, 124, 56, 23, 26, 66, 121, 122, 37, 107, 50, 78, 14, 55, 105, 71, 32, 61, 120, 12, 17, 116, 33, 68, 21, 102, 63, 16, 3, 100, 103, 75, 42, 97, 15, 64, 73, 90, 51, 88, 95, 113, 108, 36, 83], 
46: [104, 60, 54, 114, 82, 8, 19, 101, 45, 111, 10, 49, 98, 125, 7, 80, 110, 18, 24, 11, 40, 67, 69, 6, 25, 34, 65, 123, 22, 77, 92, 70, 39, 61, 26, 106, 58, 30, 48, 74, 86, 59, 53, 107, 122, 5, 1, 4, 72, 13, 109, 115, 117, 79, 91, 2, 41, 85, 119, 96, 102, 78, 23, 27, 43, 62, 84, 31, 44, 35, 57, 118, 17, 103, 112, 89, 28, 71, 32, 124, 93, 37, 66, 20, 100, 116, 14, 21, 63, 55, 87, 56, 120, 88, 113, 3, 12, 68, 81, 76, 33, 51, 64, 15, 16, 97, 95, 73, 90, 94, 108, 36, 75], 
47: [19, 54, 60, 114, 125, 80, 45, 111, 104, 39, 24, 29, 49, 7, 8, 101, 110, 18, 70, 11, 40, 67, 98, 69, 52, 10, 34, 22, 59, 77, 109, 9, 6, 48, 86, 96, 65, 92, 123, 66, 106, 5, 4, 13, 35, 57, 72, 91, 79, 117, 118, 30, 2, 25, 41, 85, 119, 74, 32, 71, 37, 20, 26, 27, 23, 53, 121, 122, 107, 1, 31, 44, 58, 115, 38, 55, 105, 87, 56, 93, 120, 43, 42, 50, 21, 102, 112, 89, 28, 124, 12, 17, 68, 116, 81, 14, 63, 78, 64, 95, 97, 3, 33, 16, 88, 94, 113, 51, 73, 90, 15, 36, 76, 75, 108], 
48: [99, 54, 114, 60, 49, 125, 98, 110, 104, 111, 45, 18, 40, 11, 8, 80, 101, 19, 7, 85, 74, 22, 47, 29, 67, 72, 69, 1, 119, 2, 25, 41, 96, 77, 10, 34, 65, 92, 39, 70, 24, 26, 44, 5, 31, 118, 46, 82, 59, 123, 71, 53, 79, 6, 9, 86, 30, 102, 63, 87, 93, 120, 61, 66, 84, 106, 121, 122, 23, 43, 27, 117, 58, 91, 115, 35, 14, 38, 32, 124, 56, 62, 107, 20, 116, 81, 78, 112, 21, 89, 105, 15, 68, 100, 88, 3, 50, 17, 33, 12, 73, 42, 64, 97, 113, 16, 76, 75, 94, 36, 83, 108, 90], 
49: [111, 104, 60, 54, 114, 125, 99, 45, 67, 98, 11, 80, 101, 7, 8, 19, 110, 96, 77, 18, 39, 70, 47, 40, 29, 118, 48, 85, 65, 92, 24, 23, 1, 44, 9, 6, 74, 46, 10, 22, 34, 82, 123, 103, 122, 58, 117, 4, 31, 2, 25, 52, 86, 119, 30, 55, 93, 124, 66, 53, 121, 20, 27, 37, 107, 5, 13, 72, 79, 35, 91, 115, 116, 71, 62, 26, 84, 106, 43, 14, 21, 78, 38, 102, 112, 56, 61, 28, 120, 32, 68, 81, 63, 89, 15, 3, 17, 12, 50, 33, 64, 94, 88, 51, 73, 90, 95, 42, 97, 113, 16, 36, 75, 83, 108, 76], 
50: [60, 99, 45, 54, 7, 80, 110, 104, 111, 114, 18, 29, 8, 39, 70, 47, 67, 125, 98, 19, 40, 11, 49, 31, 72, 52, 9, 41, 85, 96, 34, 59, 77, 92, 22, 123, 24, 62, 121, 4, 5, 44, 35, 13, 118, 109, 86, 25, 74, 119, 65, 10, 63, 56, 71, 43, 26, 122, 57, 69, 91, 117, 6, 48, 2, 68, 33, 12, 116, 21, 105, 28, 32, 61, 87, 120, 124, 53, 37, 20, 84, 1, 115, 38, 14, 112, 66, 106, 42, 16, 100, 78, 73, 64, 97, 81, 103, 55, 51, 90, 3, 75, 94, 95, 15, 113, 108, 83], 
51: [99, 114, 60, 45, 104, 54, 82, 40, 67, 98, 8, 19, 110, 7, 101, 111, 77, 123, 18, 29, 49, 80, 57, 79, 109, 9, 86, 25, 85, 96, 59, 34, 24, 47, 55, 122, 23, 5, 91, 117, 46, 119, 2, 6, 52, 41, 10, 65, 92, 68, 39, 70, 61, 26, 37, 43, 31, 13, 115, 1, 4, 35, 30, 22, 14, 112, 71, 124, 66, 121, 20, 27, 72, 105, 87, 28, 56, 84, 53, 62, 106, 107, 44, 58, 118, 50, 12, 100, 21, 93, 75, 17, 33, 78, 88, 16, 3, 116, 90, 94, 97, 83, 64, 108, 76], 
52: [60, 80, 99, 104, 54, 114, 11, 19, 101, 110, 45, 111, 123, 47, 29, 98, 7, 8, 9, 65, 24, 40, 67, 49, 125, 13, 41, 30, 10, 22, 82, 39, 70, 66, 27, 57, 79, 118, 34, 77, 59, 120, 37, 43, 62, 5, 44, 4, 69, 109, 117, 35, 25, 2, 6, 96, 74, 85, 119, 86, 17, 28, 87, 23, 122, 121, 58, 1, 31, 91, 115, 103, 21, 55, 56, 93, 20, 53, 84, 106, 107, 50, 12, 33, 14, 38, 63, 105, 124, 32, 95, 88, 100, 112, 78, 89, 42, 113, 81, 68, 75, 51, 90, 15, 16, 94, 3, 64, 36, 83, 76, 73, 108], 
53: [45, 54, 114, 80, 8, 60, 39, 7, 110, 101, 104, 111, 40, 67, 49, 19, 99, 74, 119, 22, 65, 10, 18, 47, 70, 24, 56, 11, 98, 125, 106, 72, 13, 46, 48, 59, 123, 35, 109, 118, 25, 30, 41, 34, 92, 77, 82, 87, 120, 27, 20, 5, 58, 4, 31, 69, 117, 9, 6, 52, 85, 86, 96, 89, 124, 37, 23, 66, 43, 107, 115, 79, 12, 21, 78, 112, 28, 26, 62, 90, 3, 68, 103, 38, 63, 105, 61, 71, 32, 93, 42, 95, 33, 50, 100, 116, 14, 102, 64, 15, 17, 75, 73, 97, 113, 88, 51, 94, 36, 83, 108], 
54: [45, 60, 114, 7, 104, 111, 67, 29, 19, 99, 101, 8, 80, 110, 123, 18, 70, 24, 40, 98, 11, 49, 125, 96, 34, 22, 82, 39, 47, 20, 44, 115, 77, 59, 65, 92, 69, 118, 31, 35, 117, 2, 6, 25, 74, 85, 86, 9, 41, 48, 52, 46, 71, 43, 53, 106, 121, 72, 57, 1, 5, 58, 79, 4, 13, 109, 33, 120, 62, 66, 23, 26, 37, 107, 27, 84, 122, 21, 55, 78, 28, 61, 87, 93, 32, 56, 124, 50, 17, 68, 38, 63, 89, 14, 102, 112, 105, 95, 113, 15, 3, 100, 103, 116, 12, 81, 97, 90, 42, 64, 16, 88, 108, 76, 73, 51, 83, 75, 36], 
55: [125, 54, 114, 49, 45, 111, 60, 19, 99, 7, 101, 104, 77, 47, 24, 29, 40, 67, 98, 66, 8, 80, 110, 1, 96, 10, 92, 39, 70, 11, 44, 57, 91, 69, 9, 52, 86, 59, 65, 123, 18, 61, 23, 26, 20, 37, 84, 4, 5, 35, 115, 13, 118, 6, 46, 25, 22, 34, 82, 93, 124, 27, 106, 122, 31, 109, 117, 79, 30, 85, 102, 32, 43, 121, 58, 72, 51, 17, 103, 14, 28, 71, 62, 107, 88, 3, 68, 78, 112, 42, 33, 21, 38, 89, 105, 90, 15, 16, 113, 50, 12, 76, 64, 97, 75, 73, 94, 36, 108], 
56: [60, 8, 19, 45, 111, 54, 114, 80, 104, 29, 67, 11, 98, 53, 99, 101, 7, 110, 109, 30, 22, 123, 18, 47, 70, 40, 49, 125, 122, 117, 74, 119, 34, 59, 65, 77, 82, 92, 24, 27, 43, 13, 57, 91, 72, 79, 115, 9, 41, 48, 52, 85, 10, 87, 37, 20, 31, 5, 35, 4, 118, 25, 6, 46, 86, 96, 61, 120, 62, 107, 121, 58, 69, 90, 97, 50, 68, 81, 100, 116, 14, 71, 124, 23, 42, 64, 95, 33, 12, 103, 38, 21, 63, 102, 105, 78, 89, 28, 32, 112, 83, 16, 88, 94, 75, 51, 73, 15, 108, 76], 
57: [109, 104, 54, 60, 114, 8, 80, 110, 45, 111, 29, 67, 11, 40, 125, 7, 19, 99, 101, 2, 30, 34, 65, 47, 24, 98, 52, 119, 59, 10, 77, 22, 39, 70, 62, 1, 5, 31, 69, 117, 79, 9, 6, 25, 41, 85, 86, 96, 82, 92, 123, 71, 37, 66, 72, 13, 4, 35, 91, 115, 16, 46, 55, 56, 87, 124, 26, 84, 20, 106, 107, 121, 122, 58, 38, 102, 105, 32, 28, 93, 23, 43, 100, 116, 63, 78, 89, 61, 120, 51, 12, 50, 81, 14, 112, 94, 97, 17, 33, 68, 75, 42, 95, 113, 90, 64, 88, 76, 73, 83, 108, 36], 
58: [60, 110, 104, 54, 114, 67, 19, 101, 45, 111, 59, 77, 11, 49, 40, 99, 30, 39, 70, 29, 125, 44, 9, 46, 41, 10, 82, 92, 123, 18, 47, 24, 84, 13, 118, 79, 6, 2, 85, 86, 22, 65, 23, 66, 20, 5, 1, 72, 4, 35, 91, 115, 117, 52, 25, 48, 119, 74, 96, 38, 78, 93, 32, 62, 53, 26, 27, 37, 121, 106, 31, 57, 109, 69, 33, 14, 28, 61, 87, 43, 3, 81, 102, 63, 112, 89, 56, 71, 120, 42, 12, 17, 68, 103, 116, 55, 21, 73, 64, 95, 100, 75, 15, 88, 94, 97, 113, 16, 36, 76, 51, 83], 
59: [60, 111, 54, 114, 40, 45, 104, 77, 29, 67, 125, 80, 101, 8, 19, 99, 7, 110, 82, 18, 47, 70, 58, 72, 118, 96, 86, 34, 123, 24, 84, 69, 79, 117, 2, 6, 41, 92, 10, 22, 23, 26, 20, 37, 106, 43, 57, 31, 13, 109, 9, 52, 25, 30, 46, 48, 74, 85, 119, 71, 124, 53, 66, 121, 5, 91, 1, 4, 44, 35, 115, 105, 112, 87, 32, 93, 56, 27, 62, 81, 14, 55, 38, 28, 61, 120, 95, 33, 50, 68, 103, 63, 78, 89, 21, 102, 90, 15, 42, 97, 3, 17, 12, 116, 51, 64, 113, 36, 76, 16, 94, 88, 108, 83], 
60: [111, 54, 114, 80, 45, 104, 8, 7, 99, 101, 110, 11, 29, 49, 40, 67, 125, 59, 123, 47, 18, 39, 70, 24, 30, 52, 85, 10, 34, 22, 77, 65, 82, 43, 58, 72, 79, 91, 109, 9, 6, 2, 46, 25, 48, 41, 74, 86, 96, 119, 56, 61, 26, 37, 20, 66, 106, 122, 5, 1, 4, 44, 13, 31, 35, 57, 69, 115, 117, 118, 71, 87, 120, 124, 23, 27, 53, 62, 84, 107, 121, 33, 50, 38, 78, 102, 105, 28, 32, 93, 88, 97, 12, 81, 68, 100, 103, 116, 55, 14, 21, 63, 89, 112, 42, 64, 95, 113, 3, 51, 73, 90, 15, 16, 94, 75, 83, 76, 108], 
61: [60, 99, 54, 114, 92, 125, 45, 104, 111, 46, 123, 24, 98, 8, 7, 101, 110, 91, 10, 82, 18, 67, 29, 11, 40, 49, 26, 115, 2, 48, 86, 25, 34, 77, 39, 70, 106, 122, 1, 5, 69, 4, 118, 6, 9, 74, 85, 41, 119, 59, 65, 55, 23, 37, 20, 72, 58, 31, 109, 30, 96, 3, 100, 63, 71, 56, 62, 84, 27, 107, 121, 44, 57, 13, 117, 17, 81, 103, 14, 102, 87, 28, 32, 93, 124, 53, 43, 97, 50, 33, 78, 21, 105, 51, 90, 88, 113, 68, 38, 89, 112, 73, 12, 76, 64, 15, 16, 94, 75, 108], 
62: [45, 114, 111, 104, 54, 60, 11, 99, 101, 110, 82, 67, 40, 125, 8, 19, 7, 80, 30, 86, 92, 39, 70, 29, 49, 98, 57, 4, 118, 9, 85, 52, 65, 10, 120, 5, 44, 25, 46, 74, 119, 22, 59, 77, 123, 78, 26, 66, 31, 58, 91, 13, 109, 6, 2, 41, 48, 96, 116, 105, 87, 71, 23, 37, 121, 43, 84, 106, 69, 1, 35, 72, 79, 115, 17, 50, 38, 63, 89, 102, 28, 61, 56, 53, 20, 122, 107, 12, 81, 14, 124, 64, 113, 16, 3, 103, 100, 55, 75, 73, 33, 90, 42, 15, 94, 51, 95, 76, 36, 83, 108], 
63: [99, 110, 104, 54, 60, 114, 67, 125, 7, 35, 45, 111, 120, 11, 40, 98, 8, 19, 80, 9, 48, 119, 34, 123, 82, 39, 18, 29, 49, 23, 66, 13, 2, 41, 96, 10, 92, 47, 24, 70, 122, 5, 1, 4, 6, 46, 52, 86, 22, 59, 65, 28, 61, 87, 37, 62, 26, 107, 44, 57, 58, 72, 69, 91, 117, 118, 30, 85, 50, 71, 93, 20, 27, 53, 106, 121, 84, 31, 109, 17, 81, 116, 21, 38, 78, 89, 105, 56, 43, 73, 15, 64, 33, 12, 100, 68, 112, 124, 16, 3, 103, 113, 108, 75, 94, 97, 36, 83, 76], 
64: [60, 7, 8, 45, 111, 54, 40, 19, 101, 104, 47, 29, 49, 11, 67, 125, 99, 4, 96, 92, 123, 18, 39, 70, 24, 98, 13, 72, 35, 69, 6, 30, 74, 10, 59, 77, 21, 56, 23, 62, 43, 5, 58, 109, 117, 118, 46, 48, 119, 22, 65, 82, 63, 87, 71, 26, 37, 53, 20, 66, 44, 31, 79, 91, 2, 41, 52, 42, 12, 116, 68, 38, 89, 112, 32, 120, 124, 27, 107, 122, 121, 106, 1, 57, 3, 50, 33, 102, 61, 81, 55, 78, 103, 105, 76, 73, 90, 108, 94, 97, 83, 75, 51, 15, 95, 36], 
65: [114, 45, 104, 54, 60, 125, 8, 80, 7, 101, 70, 49, 11, 98, 19, 99, 110, 22, 39, 24, 29, 40, 67, 31, 52, 2, 10, 34, 92, 123, 47, 57, 35, 30, 41, 48, 46, 119, 77, 82, 53, 84, 122, 69, 117, 115, 9, 86, 25, 74, 85, 96, 78, 71, 124, 62, 26, 37, 27, 106, 107, 44, 5, 1, 58, 79, 4, 91, 109, 118, 17, 14, 89, 102, 28, 56, 87, 93, 20, 23, 66, 55, 105, 112, 61, 32, 120, 68, 100, 103, 12, 21, 38, 63, 42, 16, 88, 113, 50, 33, 116, 75, 73, 15, 94, 51, 64, 95, 97, 108, 76, 90, 83, 36], 
66: [60, 80, 110, 54, 114, 98, 125, 7, 99, 19, 45, 104, 111, 47, 24, 49, 11, 40, 67, 8, 101, 5, 52, 77, 39, 29, 20, 44, 1, 35, 9, 30, 96, 86, 119, 10, 22, 59, 82, 18, 70, 55, 26, 13, 57, 58, 91, 79, 118, 25, 48, 41, 85, 34, 65, 63, 120, 23, 27, 62, 84, 122, 69, 72, 117, 46, 6, 116, 71, 93, 53, 37, 43, 106, 4, 31, 109, 3, 38, 102, 78, 28, 32, 87, 107, 121, 94, 17, 33, 14, 105, 124, 42, 12, 89, 112, 75, 73, 64, 16, 113, 50, 68, 103, 81, 36, 51, 15, 88, 83, 76], 
67: [54, 45, 60, 114, 11, 19, 104, 111, 96, 70, 49, 7, 80, 99, 8, 101, 110, 77, 82, 29, 40, 98, 125, 9, 119, 59, 34, 123, 92, 39, 47, 18, 24, 58, 31, 118, 109, 35, 2, 6, 41, 85, 65, 10, 22, 106, 23, 20, 44, 57, 117, 13, 69, 79, 48, 52, 25, 46, 74, 30, 86, 33, 62, 53, 66, 37, 84, 5, 72, 4, 91, 63, 56, 71, 93, 120, 26, 27, 43, 122, 121, 12, 81, 55, 14, 38, 61, 28, 42, 3, 50, 116, 68, 103, 21, 78, 102, 112, 105, 89, 17, 100, 108, 51, 64, 95, 97, 88, 75, 113, 16, 73, 90, 76, 36, 83], 
68: [114, 29, 7, 104, 54, 60, 8, 19, 45, 111, 18, 24, 40, 80, 101, 117, 34, 92, 77, 82, 49, 67, 99, 110, 9, 25, 86, 22, 39, 47, 112, 11, 98, 125, 43, 121, 31, 6, 85, 96, 65, 59, 123, 70, 122, 44, 72, 13, 91, 1, 35, 48, 74, 2, 46, 119, 10, 21, 56, 93, 53, 23, 84, 58, 69, 79, 52, 41, 50, 14, 102, 71, 87, 20, 27, 37, 26, 5, 57, 118, 109, 115, 51, 33, 103, 55, 63, 38, 61, 124, 66, 107, 64, 116, 28, 42, 3, 76, 90, 15, 97, 100, 36, 75, 94, 95, 83, 108], 
69: [54, 125, 45, 60, 114, 47, 8, 19, 104, 111, 22, 70, 40, 67, 11, 7, 80, 99, 101, 110, 41, 48, 46, 59, 10, 77, 123, 24, 29, 98, 106, 109, 25, 74, 96, 34, 65, 82, 39, 18, 31, 57, 118, 52, 6, 86, 92, 32, 87, 26, 20, 5, 35, 72, 91, 13, 115, 30, 2, 119, 55, 89, 71, 61, 93, 124, 27, 53, 66, 37, 43, 107, 1, 44, 58, 79, 117, 102, 105, 28, 120, 62, 84, 122, 17, 33, 21, 38, 63, 78, 112, 56, 64, 88, 3, 12, 50, 68, 81, 14, 15, 42, 16, 95, 97, 113, 36, 76, 73, 90, 108, 75], 
70: [45, 54, 114, 125, 104, 111, 60, 67, 101, 19, 11, 40, 49, 7, 99, 8, 80, 110, 65, 47, 39, 29, 98, 5, 59, 77, 92, 123, 10, 22, 82, 18, 24, 31, 35, 69, 2, 25, 74, 85, 96, 119, 34, 37, 58, 115, 13, 91, 118, 30, 46, 9, 48, 52, 86, 124, 53, 62, 122, 106, 121, 44, 1, 4, 72, 57, 79, 117, 109, 78, 102, 105, 28, 32, 71, 93, 56, 66, 26, 27, 50, 12, 33, 81, 100, 14, 38, 55, 89, 61, 87, 120, 17, 116, 63, 21, 16, 113, 15, 3, 68, 103, 42, 64, 95, 97, 108, 73, 90, 88, 75, 51, 76, 83, 36], 
71: [54, 60, 7, 114, 125, 110, 104, 111, 45, 77, 92, 47, 18, 29, 67, 11, 40, 49, 19, 80, 8, 99, 101, 48, 2, 86, 34, 65, 59, 24, 70, 98, 37, 121, 57, 72, 79, 109, 118, 6, 85, 119, 96, 39, 69, 91, 9, 25, 46, 22, 82, 123, 87, 124, 20, 62, 66, 106, 5, 44, 35, 115, 41, 74, 12, 17, 78, 102, 32, 61, 93, 26, 27, 23, 84, 107, 122, 4, 31, 58, 117, 1, 50, 116, 38, 63, 89, 105, 28, 56, 120, 53, 68, 81, 112, 55, 64, 97, 113, 108, 51, 73, 94, 16, 90, 42, 95, 3, 83, 76], 
72: [111, 60, 80, 45, 104, 54, 114, 29, 40, 98, 110, 59, 18, 39, 11, 7, 8, 19, 99, 48, 47, 49, 67, 125, 79, 25, 96, 34, 92, 24, 70, 37, 53, 31, 9, 2, 30, 46, 6, 74, 85, 86, 119, 10, 22, 77, 82, 112, 71, 43, 106, 5, 44, 58, 4, 13, 35, 57, 69, 91, 109, 117, 118, 41, 102, 56, 66, 23, 27, 84, 121, 1, 115, 50, 116, 38, 89, 105, 61, 87, 32, 62, 20, 107, 97, 68, 81, 63, 14, 21, 93, 124, 120, 42, 64, 113, 3, 12, 33, 55, 78, 95, 100, 90, 15, 88, 94, 83, 51, 73, 36, 108], 
73: [7, 60, 39, 11, 125, 99, 45, 111, 54, 114, 98, 8, 101, 110, 104, 119, 49, 80, 19, 65, 10, 47, 70, 89, 67, 28, 32, 106, 58, 4, 35, 109, 48, 6, 74, 85, 30, 22, 77, 92, 123, 18, 24, 63, 105, 87, 62, 66, 37, 5, 1, 13, 91, 118, 115, 9, 46, 2, 41, 34, 78, 61, 71, 120, 53, 84, 26, 27, 69, 44, 72, 25, 52, 50, 56, 107, 121, 20, 23, 57, 33, 81, 100, 38, 75, 42, 94, 97, 113, 3, 12, 17, 103, 116, 21, 55, 102, 112, 124, 64, 15, 16, 88, 76, 83], 
74: [101, 54, 60, 114, 125, 19, 80, 8, 45, 104, 111, 18, 29, 49, 11, 99, 110, 7, 48, 2, 70, 67, 98, 44, 25, 92, 123, 10, 22, 77, 24, 47, 39, 53, 115, 69, 85, 46, 119, 41, 59, 65, 82, 34, 21, 32, 23, 26, 79, 91, 118, 31, 72, 35, 86, 96, 6, 30, 52, 120, 56, 84, 62, 107, 37, 117, 1, 4, 58, 13, 109, 95, 100, 103, 116, 89, 102, 38, 112, 87, 28, 61, 93, 124, 106, 121, 122, 43, 81, 14, 71, 3, 50, 33, 68, 12, 78, 105, 64, 88, 97, 15, 42, 113, 17, 108, 90, 73, 94, 75, 76, 83, 36], 
75: [45, 11, 110, 60, 114, 7, 8, 101, 54, 67, 98, 19, 80, 117, 111, 104, 10, 65, 39, 29, 49, 99, 26, 109, 35, 30, 52, 85, 77, 82, 123, 18, 24, 70, 47, 125, 28, 62, 66, 122, 58, 57, 91, 1, 13, 9, 48, 41, 119, 92, 22, 38, 53, 43, 20, 44, 4, 5, 6, 74, 96, 105, 56, 84, 107, 31, 69, 115, 86, 25, 46, 100, 63, 78, 61, 87, 120, 124, 106, 23, 73, 51, 16, 33, 50, 116, 68, 14, 55, 89, 93, 27, 121, 3, 12, 17, 21, 42, 94, 88, 102, 64, 95, 83], 
76: [101, 104, 111, 114, 54, 8, 25, 60, 40, 99, 19, 46, 82, 125, 98, 7, 79, 96, 85, 10, 59, 77, 92, 22, 34, 47, 24, 112, 67, 11, 49, 44, 6, 48, 86, 2, 3, 65, 123, 18, 70, 122, 106, 84, 5, 91, 115, 69, 118, 1, 58, 9, 52, 74, 30, 61, 93, 121, 37, 43, 23, 27, 26, 57, 41, 81, 103, 68, 14, 55, 78, 102, 21, 32, 62, 20, 107, 13, 109, 17, 105, 56, 71, 87, 120, 66, 64, 113, 15, 33, 124, 94, 100, 63, 90, 73, 88, 97, 116, 51, 95, 42, 108, 36], 
77: [104, 54, 60, 114, 67, 49, 101, 110, 45, 111, 59, 11, 125, 7, 8, 19, 80, 99, 2, 24, 47, 70, 29, 40, 98, 58, 91, 6, 25, 96, 86, 34, 10, 22, 92, 82, 18, 84, 122, 35, 79, 115, 69, 117, 118, 46, 74, 48, 85, 119, 65, 123, 32, 71, 124, 66, 20, 23, 5, 44, 57, 31, 109, 30, 52, 9, 41, 55, 93, 106, 72, 1, 13, 68, 38, 78, 102, 56, 61, 26, 27, 43, 53, 62, 121, 37, 12, 116, 89, 14, 112, 28, 87, 50, 17, 103, 21, 51, 42, 88, 94, 81, 64, 95, 97, 3, 36, 75, 76, 73, 113, 16, 83, 108], 
78: [114, 104, 54, 60, 8, 110, 45, 111, 101, 65, 70, 40, 49, 11, 125, 19, 99, 80, 30, 46, 85, 10, 77, 67, 98, 62, 58, 91, 2, 86, 22, 92, 39, 18, 47, 43, 84, 106, 122, 4, 35, 44, 117, 48, 9, 96, 59, 34, 82, 123, 38, 89, 71, 53, 66, 57, 69, 1, 31, 79, 109, 115, 52, 41, 74, 81, 14, 102, 28, 120, 23, 27, 5, 72, 118, 16, 88, 63, 61, 32, 56, 93, 124, 37, 26, 20, 107, 33, 103, 55, 87, 73, 15, 97, 50, 17, 100, 116, 76, 75, 42, 64, 95, 113, 3, 51, 94, 108, 83], 
79: [60, 101, 54, 114, 40, 11, 7, 80, 19, 45, 111, 104, 29, 67, 98, 8, 99, 110, 96, 25, 77, 82, 34, 59, 47, 125, 49, 115, 72, 109, 2, 9, 85, 30, 52, 22, 10, 18, 39, 24, 70, 14, 32, 5, 1, 117, 57, 58, 31, 6, 41, 119, 46, 48, 74, 86, 65, 92, 123, 71, 27, 107, 20, 37, 66, 122, 13, 91, 38, 56, 93, 120, 23, 43, 4, 44, 69, 81, 102, 112, 124, 106, 53, 62, 84, 121, 90, 95, 97, 116, 33, 100, 103, 55, 78, 89, 105, 87, 51, 88, 94, 3, 17, 68, 12, 21, 76, 42, 113, 64, 16, 83], 
80: [60, 114, 110, 45, 54, 29, 8, 7, 101, 19, 111, 104, 34, 40, 98, 99, 22, 39, 47, 11, 67, 49, 125, 35, 25, 52, 65, 18, 24, 70, 31, 117, 72, 115, 9, 2, 74, 85, 86, 119, 59, 10, 77, 82, 123, 53, 66, 44, 4, 57, 79, 6, 30, 41, 48, 46, 96, 89, 28, 120, 23, 43, 84, 107, 122, 5, 1, 13, 69, 91, 118, 109, 93, 56, 87, 124, 26, 27, 37, 62, 106, 20, 121, 12, 50, 14, 21, 38, 105, 112, 32, 71, 17, 68, 100, 103, 116, 55, 63, 78, 102, 42, 88, 33, 81, 94, 95, 97, 113, 90, 15, 83, 108, 75, 51, 73, 36], 
81: [111, 60, 11, 54, 40, 67, 19, 101, 104, 70, 29, 49, 125, 8, 80, 99, 110, 2, 85, 34, 59, 39, 18, 47, 24, 98, 122, 1, 13, 79, 41, 6, 25, 30, 48, 74, 82, 92, 123, 120, 107, 44, 91, 58, 72, 115, 9, 46, 86, 96, 119, 22, 10, 77, 14, 78, 32, 56, 61, 62, 121, 4, 5, 31, 57, 69, 109, 118, 52, 103, 63, 89, 105, 71, 93, 37, 43, 106, 20, 117, 12, 38, 28, 26, 66, 23, 27, 113, 3, 17, 33, 100, 116, 102, 90, 15, 16, 88, 50, 21, 76, 73, 64, 95, 94, 97, 108, 83], 
82: [54, 45, 60, 114, 67, 101, 104, 111, 46, 29, 11, 40, 7, 8, 19, 80, 110, 117, 96, 34, 59, 123, 70, 24, 49, 98, 125, 31, 77, 92, 18, 62, 122, 79, 115, 9, 52, 85, 86, 25, 30, 10, 65, 22, 23, 84, 106, 107, 5, 35, 58, 44, 69, 1, 109, 118, 6, 41, 2, 48, 74, 119, 14, 93, 61, 26, 66, 27, 20, 43, 121, 72, 4, 13, 57, 91, 17, 68, 63, 102, 56, 120, 53, 37, 3, 33, 38, 112, 28, 87, 71, 124, 51, 81, 100, 12, 103, 116, 55, 78, 21, 105, 15, 95, 76, 90, 88, 16, 113, 108, 75, 64, 36, 83], 
83: [54, 60, 114, 29, 80, 111, 22, 99, 45, 104, 34, 123, 11, 49, 7, 8, 110, 39, 98, 56, 101, 20, 31, 72, 117, 9, 48, 52, 65, 77, 82, 18, 70, 40, 67, 120, 121, 35, 25, 30, 41, 85, 2, 119, 59, 53, 43, 57, 1, 118, 79, 115, 74, 96, 92, 71, 23, 27, 37, 62, 122, 66, 109, 33, 68, 116, 14, 38, 105, 87, 84, 107, 5, 13, 44, 58, 95, 50, 21, 63, 78, 89, 28, 93, 124, 51, 64, 81, 102, 32, 16, 94, 97, 12, 112, 88, 75, 15, 36, 73, 90, 108], 
84: [114, 110, 45, 104, 54, 60, 99, 7, 8, 80, 111, 2, 59, 77, 29, 67, 19, 101, 65, 82, 49, 11, 40, 98, 125, 58, 1, 44, 9, 34, 22, 92, 18, 24, 106, 20, 23, 91, 35, 25, 46, 48, 85, 6, 41, 74, 86, 10, 123, 93, 124, 121, 43, 66, 57, 117, 115, 118, 72, 52, 30, 119, 96, 55, 78, 28, 26, 27, 62, 4, 31, 69, 79, 109, 33, 14, 105, 112, 102, 61, 87, 71, 37, 100, 17, 68, 38, 63, 89, 32, 120, 42, 88, 103, 50, 116, 21, 97, 113, 16, 95, 3, 76, 73, 94, 15, 36, 75, 51, 108, 83], 
85: [60, 114, 110, 45, 54, 11, 49, 8, 80, 19, 99, 101, 104, 111, 25, 67, 7, 9, 48, 96, 34, 24, 70, 29, 98, 125, 31, 86, 77, 82, 22, 18, 39, 47, 44, 79, 41, 74, 30, 59, 65, 10, 123, 92, 93, 62, 121, 57, 58, 72, 5, 115, 1, 13, 91, 118, 6, 46, 2, 52, 119, 38, 14, 78, 71, 66, 23, 27, 37, 84, 122, 107, 35, 117, 16, 33, 81, 100, 103, 105, 61, 56, 87, 26, 43, 53, 20, 106, 50, 68, 102, 32, 120, 17, 116, 12, 63, 55, 51, 15, 88, 3, 76, 75, 73, 113, 94, 97, 90, 42, 95, 83, 108], 
86: [114, 110, 54, 60, 24, 19, 80, 45, 104, 111, 40, 29, 98, 125, 8, 99, 101, 91, 59, 92, 10, 77, 39, 47, 67, 49, 9, 25, 30, 85, 34, 82, 18, 70, 62, 115, 2, 6, 96, 46, 119, 22, 65, 71, 66, 26, 5, 57, 13, 35, 58, 69, 72, 4, 31, 118, 79, 41, 48, 52, 74, 17, 14, 102, 61, 32, 93, 124, 37, 106, 27, 43, 121, 84, 107, 122, 1, 44, 109, 68, 116, 55, 78, 87, 28, 20, 23, 53, 113, 12, 100, 63, 38, 89, 112, 105, 56, 3, 50, 81, 103, 51, 42, 16, 94, 97, 15, 95, 108, 76, 90, 36, 75], 
87: [45, 60, 114, 7, 111, 54, 19, 80, 101, 109, 104, 41, 11, 8, 99, 110, 9, 47, 29, 40, 98, 125, 13, 69, 48, 52, 6, 119, 59, 22, 65, 123, 18, 24, 39, 70, 53, 27, 37, 106, 35, 4, 57, 118, 115, 2, 74, 85, 86, 10, 34, 77, 82, 28, 56, 71, 120, 62, 43, 107, 44, 72, 58, 25, 30, 46, 63, 89, 66, 84, 121, 91, 5, 79, 117, 12, 100, 105, 61, 32, 26, 20, 23, 122, 50, 17, 33, 68, 112, 73, 15, 42, 64, 94, 21, 38, 78, 102, 95, 97, 3, 51, 90, 16, 75, 113, 36, 108, 76, 83], 
88: [60, 114, 8, 80, 110, 101, 104, 111, 54, 43, 7, 45, 30, 24, 67, 11, 40, 49, 98, 19, 99, 46, 52, 65, 10, 22, 34, 77, 123, 47, 18, 102, 29, 125, 32, 93, 69, 1, 79, 25, 48, 41, 85, 2, 74, 82, 33, 39, 70, 14, 78, 27, 84, 91, 117, 115, 96, 59, 55, 89, 61, 124, 26, 106, 58, 31, 35, 72, 4, 109, 118, 6, 119, 103, 21, 56, 53, 23, 66, 107, 122, 5, 44, 57, 97, 81, 100, 105, 28, 20, 112, 38, 51, 73, 90, 15, 42, 95, 16, 113, 75, 3, 76, 94, 108, 83], 
89: [101, 80, 45, 114, 40, 8, 111, 104, 54, 60, 11, 29, 7, 99, 110, 22, 39, 28, 125, 19, 4, 10, 65, 70, 47, 49, 67, 98, 69, 35, 115, 30, 6, 46, 74, 119, 34, 92, 77, 24, 18, 53, 106, 13, 72, 117, 109, 9, 25, 48, 86, 2, 59, 123, 78, 32, 87, 62, 27, 43, 107, 23, 44, 58, 91, 1, 57, 79, 5, 41, 52, 38, 102, 93, 71, 120, 124, 84, 118, 73, 94, 17, 81, 21, 63, 105, 56, 26, 37, 66, 122, 20, 42, 88, 103, 61, 64, 15, 116, 100, 55, 113, 95, 97, 36, 108, 75, 90, 83], 
90: [8, 60, 54, 7, 45, 104, 114, 39, 24, 40, 29, 80, 101, 79, 111, 59, 123, 11, 49, 98, 56, 19, 99, 53, 25, 92, 34, 82, 47, 70, 67, 125, 13, 117, 115, 118, 109, 6, 74, 52, 119, 22, 18, 61, 120, 20, 27, 107, 5, 4, 72, 9, 30, 41, 46, 85, 96, 86, 65, 10, 14, 62, 37, 43, 31, 57, 69, 48, 33, 81, 21, 55, 112, 32, 71, 87, 124, 26, 106, 121, 1, 50, 68, 103, 100, 102, 23, 95, 17, 12, 38, 28, 15, 64, 88, 3, 89, 51, 42, 94, 113, 108, 76, 16, 83], 
91: [19, 60, 114, 111, 10, 8, 110, 45, 104, 2, 86, 77, 29, 98, 125, 7, 80, 99, 101, 18, 47, 70, 40, 49, 67, 11, 5, 35, 30, 92, 39, 61, 122, 44, 4, 6, 85, 46, 74, 119, 59, 65, 34, 82, 66, 26, 84, 106, 107, 121, 58, 1, 31, 72, 57, 69, 117, 79, 109, 115, 9, 25, 48, 52, 41, 96, 55, 78, 56, 71, 62, 37, 23, 43, 13, 118, 100, 116, 102, 105, 32, 93, 124, 27, 94, 17, 68, 81, 103, 38, 63, 21, 89, 87, 28, 42, 97, 3, 12, 50, 14, 112, 51, 88, 75, 73, 64, 16, 76, 15, 95, 108], 
92: [114, 104, 54, 8, 19, 45, 111, 24, 11, 49, 67, 98, 125, 7, 99, 101, 110, 18, 70, 29, 40, 2, 9, 86, 65, 77, 82, 123, 47, 39, 61, 1, 48, 74, 46, 96, 119, 22, 34, 59, 10, 71, 62, 26, 37, 121, 5, 58, 72, 31, 44, 13, 35, 91, 115, 117, 118, 6, 25, 41, 30, 85, 17, 116, 102, 93, 84, 57, 69, 4, 79, 109, 68, 55, 56, 32, 124, 53, 20, 23, 27, 106, 122, 107, 14, 21, 63, 78, 89, 28, 50, 33, 81, 103, 100, 38, 112, 105, 64, 3, 90, 15, 16, 113, 94, 95, 76, 51, 73, 42, 36, 75, 108, 83], 
93: [114, 110, 111, 54, 60, 49, 125, 80, 7, 99, 101, 104, 34, 18, 40, 67, 29, 8, 19, 44, 31, 96, 85, 77, 82, 92, 24, 47, 39, 70, 11, 98, 1, 25, 9, 2, 86, 48, 59, 65, 22, 123, 84, 35, 118, 58, 69, 79, 117, 6, 46, 30, 41, 52, 74, 119, 10, 17, 26, 66, 106, 122, 5, 13, 57, 91, 115, 55, 28, 71, 23, 37, 121, 107, 72, 88, 68, 112, 63, 89, 21, 124, 32, 61, 53, 20, 81, 103, 78, 102, 105, 120, 97, 113, 12, 33, 100, 38, 3, 36, 76, 15, 16, 42, 94, 95, 51, 108, 75, 83], 
94: [111, 19, 60, 114, 11, 99, 7, 80, 101, 104, 119, 10, 22, 39, 98, 29, 49, 8, 110, 5, 91, 6, 2, 77, 24, 47, 125, 40, 107, 66, 1, 79, 30, 41, 86, 9, 34, 65, 92, 18, 89, 32, 117, 4, 57, 109, 25, 52, 85, 74, 59, 28, 124, 87, 121, 23, 44, 58, 13, 72, 118, 46, 48, 116, 14, 56, 71, 43, 26, 27, 62, 84, 122, 100, 38, 112, 120, 61, 93, 53, 37, 42, 17, 81, 103, 12, 102, 63, 73, 50, 68, 55, 78, 21, 105, 113, 75, 76, 51, 90, 16, 64, 95, 88, 83, 36], 
95: [54, 60, 74, 114, 22, 29, 40, 11, 19, 101, 80, 110, 45, 104, 111, 41, 59, 47, 67, 8, 7, 79, 52, 82, 123, 39, 70, 49, 43, 109, 115, 25, 2, 96, 34, 77, 92, 18, 24, 56, 27, 53, 72, 58, 117, 118, 69, 6, 46, 9, 86, 65, 38, 102, 32, 120, 20, 84, 106, 44, 4, 13, 31, 35, 57, 30, 85, 100, 28, 87, 107, 121, 91, 42, 103, 14, 71, 93, 62, 33, 116, 81, 21, 78, 89, 105, 124, 90, 97, 12, 50, 68, 112, 15, 113, 88, 17, 83, 108, 16, 3, 36, 64, 94, 75, 76], 
96: [54, 67, 49, 45, 104, 60, 114, 99, 31, 111, 82, 40, 11, 29, 125, 7, 8, 19, 80, 110, 85, 59, 77, 47, 70, 98, 79, 48, 92, 123, 24, 18, 39, 72, 1, 35, 44, 69, 117, 118, 2, 6, 25, 86, 10, 22, 34, 65, 14, 93, 66, 20, 43, 5, 57, 9, 41, 46, 52, 74, 30, 55, 112, 71, 124, 23, 106, 121, 122, 13, 4, 58, 91, 33, 63, 38, 32, 53, 62, 26, 27, 37, 84, 107, 3, 12, 50, 68, 103, 78, 56, 61, 120, 64, 81, 116, 21, 102, 105, 51, 16, 95, 17, 108, 76, 42, 88, 97, 90, 113, 36, 75, 83], 
97: [60, 111, 54, 114, 45, 8, 80, 99, 110, 104, 34, 123, 18, 39, 47, 40, 67, 98, 19, 7, 101, 27, 72, 79, 2, 59, 70, 29, 49, 125, 56, 91, 109, 9, 30, 74, 86, 119, 10, 22, 77, 32, 61, 35, 57, 69, 115, 46, 48, 41, 85, 96, 65, 33, 71, 93, 37, 43, 20, 84, 122, 1, 31, 58, 6, 25, 21, 78, 112, 105, 87, 120, 124, 23, 53, 106, 107, 4, 13, 88, 50, 100, 102, 28, 121, 68, 81, 103, 38, 55, 63, 89, 73, 15, 95, 113, 3, 12, 14, 42, 108, 51, 16, 64, 76, 83], 
98: [99, 104, 54, 114, 11, 7, 8, 19, 80, 110, 45, 111, 40, 49, 125, 101, 39, 29, 67, 48, 34, 10, 65, 22, 92, 18, 47, 24, 70, 122, 72, 5, 1, 41, 52, 9, 25, 46, 86, 119, 77, 82, 66, 107, 91, 79, 6, 30, 74, 2, 96, 85, 23, 27, 43, 13, 31, 69, 4, 35, 44, 57, 115, 117, 109, 118, 105, 28, 56, 61, 124, 53, 62, 84, 106, 20, 121, 55, 21, 63, 112, 102, 32, 71, 87, 93, 120, 113, 50, 17, 116, 100, 103, 14, 78, 89, 33, 68, 12, 81, 73, 51, 15, 42, 88, 94, 97, 3, 75, 90, 16, 64, 108, 76, 36, 83], 
99: [104, 54, 60, 114, 98, 110, 45, 111, 49, 29, 125, 8, 7, 80, 19, 101, 39, 18, 11, 40, 67, 52, 48, 10, 123, 24, 70, 1, 5, 9, 25, 41, 85, 96, 119, 59, 34, 65, 77, 92, 22, 23, 44, 13, 118, 6, 2, 30, 74, 86, 61, 37, 62, 66, 84, 26, 121, 122, 58, 72, 4, 31, 35, 57, 79, 91, 115, 69, 117, 50, 21, 63, 105, 28, 32, 93, 120, 53, 27, 20, 43, 106, 107, 103, 55, 89, 112, 56, 71, 87, 124, 12, 17, 100, 14, 78, 51, 15, 42, 33, 68, 81, 116, 73, 94, 113, 97, 3, 16, 64, 88, 76, 90, 83, 108, 75, 36], 
100: [60, 114, 19, 54, 10, 80, 99, 101, 110, 45, 104, 111, 41, 39, 70, 11, 98, 107, 7, 8, 109, 2, 74, 85, 34, 24, 67, 29, 40, 91, 4, 115, 6, 25, 9, 46, 86, 119, 22, 65, 82, 92, 123, 18, 61, 5, 57, 79, 30, 52, 48, 103, 87, 56, 43, 84, 122, 13, 1, 117, 35, 105, 28, 120, 27, 26, 53, 62, 106, 121, 31, 58, 72, 44, 33, 116, 14, 63, 21, 32, 93, 37, 95, 17, 50, 81, 102, 89, 78, 124, 15, 88, 94, 97, 113, 38, 112, 75, 51, 73, 90, 16, 3, 12, 68, 42, 108, 76], 
101: [45, 54, 60, 114, 40, 80, 111, 11, 29, 7, 8, 19, 99, 110, 18, 70, 67, 49, 98, 125, 74, 25, 65, 77, 82, 123, 39, 47, 24, 79, 115, 6, 30, 46, 52, 85, 119, 59, 10, 34, 92, 22, 89, 28, 23, 44, 58, 109, 2, 86, 9, 48, 41, 26, 27, 53, 62, 107, 122, 5, 13, 57, 1, 4, 31, 35, 69, 91, 117, 118, 38, 102, 105, 32, 87, 93, 66, 37, 43, 20, 106, 84, 121, 17, 103, 14, 55, 21, 78, 56, 61, 71, 120, 124, 12, 68, 81, 100, 88, 3, 33, 116, 76, 15, 64, 42, 94, 95, 75, 51, 73, 90, 97, 113, 36, 108, 83], 
102: [104, 111, 60, 114, 8, 101, 54, 125, 45, 30, 92, 70, 24, 11, 49, 98, 7, 19, 80, 46, 48, 25, 86, 65, 82, 77, 47, 29, 67, 40, 72, 74, 2, 22, 39, 32, 27, 69, 57, 91, 79, 9, 85, 10, 34, 59, 71, 66, 62, 37, 43, 84, 107, 1, 5, 35, 58, 109, 115, 6, 96, 119, 88, 116, 38, 55, 78, 89, 61, 124, 26, 121, 31, 44, 4, 118, 117, 68, 14, 56, 93, 53, 20, 106, 122, 42, 95, 33, 28, 87, 113, 100, 103, 81, 105, 112, 15, 64, 16, 97, 76, 90, 94, 3, 73, 108, 36, 75, 83], 
103: [49, 60, 40, 19, 99, 101, 104, 54, 114, 29, 125, 80, 110, 111, 45, 11, 67, 98, 7, 8, 1, 74, 52, 46, 85, 10, 22, 39, 24, 18, 122, 4, 44, 118, 6, 96, 65, 34, 59, 92, 77, 82, 123, 70, 26, 13, 91, 79, 9, 25, 30, 41, 2, 86, 100, 61, 43, 53, 23, 121, 31, 58, 117, 119, 81, 14, 55, 21, 112, 28, 32, 93, 56, 27, 62, 84, 5, 35, 115, 15, 68, 38, 78, 89, 120, 124, 66, 20, 107, 88, 3, 17, 33, 116, 63, 102, 95, 113, 50, 76, 90, 42, 97, 94, 12, 73, 64, 108], 
104: [114, 54, 60, 8, 99, 45, 111, 49, 98, 125, 7, 19, 80, 110, 65, 70, 11, 40, 29, 67, 46, 92, 77, 39, 24, 47, 9, 25, 30, 52, 96, 10, 59, 34, 22, 82, 123, 1, 5, 57, 31, 58, 72, 117, 2, 41, 48, 86, 74, 119, 85, 112, 78, 43, 27, 62, 84, 122, 13, 4, 35, 44, 69, 79, 91, 109, 115, 118, 102, 28, 23, 37, 20, 26, 53, 66, 106, 107, 121, 68, 14, 63, 89, 21, 32, 56, 71, 93, 61, 87, 120, 124, 15, 33, 50, 103, 55, 113, 17, 116, 81, 100, 16, 88, 3, 76, 51, 90, 42, 64, 95, 97, 94, 73, 108, 75, 36, 83], 
105: [60, 114, 7, 101, 99, 45, 54, 39, 29, 98, 8, 80, 110, 111, 70, 47, 40, 11, 125, 19, 59, 18, 24, 67, 109, 25, 119, 85, 65, 34, 62, 106, 72, 4, 57, 69, 91, 2, 52, 30, 48, 9, 86, 41, 10, 22, 82, 92, 123, 84, 107, 121, 5, 31, 35, 44, 118, 13, 79, 74, 96, 71, 32, 87, 28, 120, 66, 37, 53, 1, 115, 50, 12, 81, 100, 63, 21, 89, 56, 61, 124, 93, 23, 26, 27, 20, 43, 73, 33, 14, 16, 15, 97, 113, 17, 116, 55, 102, 75, 51, 42, 88, 3, 95, 76, 64, 94, 83, 108], 
106: [54, 60, 45, 114, 67, 8, 7, 104, 111, 10, 47, 125, 80, 99, 101, 110, 69, 46, 59, 82, 24, 70, 11, 40, 49, 98, 53, 4, 6, 65, 77, 123, 18, 37, 84, 31, 13, 35, 91, 72, 109, 30, 86, 2, 48, 96, 34, 92, 87, 61, 20, 26, 5, 57, 44, 58, 115, 117, 118, 9, 25, 41, 52, 74, 85, 119, 78, 89, 105, 71, 93, 23, 43, 62, 66, 79, 3, 17, 55, 28, 32, 124, 120, 27, 107, 42, 113, 12, 63, 73, 16, 33, 81, 100, 21, 38, 102, 112, 95, 88, 50, 116, 76, 64, 15, 97, 108, 51, 90, 36, 75], 
107: [19, 54, 60, 114, 39, 11, 98, 8, 7, 80, 101, 45, 104, 111, 10, 34, 49, 110, 99, 5, 41, 82, 47, 40, 29, 125, 122, 1, 115, 25, 46, 22, 65, 123, 18, 24, 91, 79, 13, 6, 2, 9, 30, 74, 85, 86, 92, 100, 57, 4, 35, 44, 69, 109, 117, 48, 52, 96, 119, 32, 87, 27, 53, 43, 31, 72, 17, 81, 63, 89, 102, 105, 71, 56, 61, 93, 120, 26, 62, 37, 66, 106, 121, 94, 12, 116, 14, 112, 28, 15, 21, 38, 55, 78, 90, 113, 3, 68, 103, 16, 64, 95, 97, 88, 108, 75, 51, 73, 76, 36, 83], 
108: [54, 67, 80, 35, 45, 104, 111, 2, 60, 114, 34, 70, 40, 98, 8, 19, 99, 74, 96, 82, 29, 49, 125, 101, 110, 31, 86, 10, 59, 65, 22, 123, 18, 11, 71, 124, 117, 118, 48, 25, 46, 77, 92, 47, 24, 39, 23, 37, 106, 107, 69, 72, 91, 6, 85, 119, 63, 56, 26, 84, 122, 57, 44, 109, 52, 89, 21, 112, 61, 87, 93, 53, 27, 43, 62, 121, 4, 15, 68, 12, 50, 116, 78, 102, 105, 120, 64, 95, 42, 97, 113, 17, 81, 100, 55, 90, 103, 51, 88, 76, 16, 36, 83], 
109: [60, 57, 45, 54, 114, 67, 8, 101, 104, 111, 47, 40, 11, 7, 19, 80, 110, 119, 87, 29, 98, 125, 4, 79, 69, 77, 10, 34, 59, 82, 18, 24, 39, 70, 56, 37, 52, 6, 30, 46, 2, 41, 22, 65, 92, 123, 28, 71, 124, 53, 27, 106, 13, 72, 91, 117, 115, 9, 25, 86, 74, 100, 105, 43, 62, 107, 122, 5, 58, 31, 35, 38, 89, 32, 61, 26, 23, 66, 20, 84, 121, 50, 12, 116, 14, 55, 21, 78, 102, 112, 120, 51, 95, 97, 33, 81, 63, 75, 73, 90, 16, 64, 94, 17, 68, 42, 88, 3, 113, 108, 76, 83], 
110: [114, 80, 45, 54, 60, 40, 99, 104, 111, 29, 98, 19, 7, 101, 18, 39, 24, 11, 49, 67, 125, 41, 25, 85, 86, 22, 34, 77, 123, 47, 70, 44, 35, 58, 48, 2, 9, 30, 52, 119, 10, 65, 59, 82, 92, 66, 84, 72, 1, 31, 57, 91, 4, 13, 115, 117, 6, 46, 74, 96, 28, 93, 23, 53, 62, 26, 27, 43, 121, 122, 69, 5, 79, 109, 118, 12, 63, 38, 78, 71, 120, 20, 106, 107, 17, 50, 14, 89, 105, 32, 61, 87, 56, 124, 16, 100, 103, 116, 55, 112, 88, 113, 33, 68, 81, 75, 95, 97, 3, 51, 73, 42, 15, 94, 83, 36, 108], 
111: [60, 114, 54, 104, 49, 125, 8, 7, 80, 19, 99, 101, 110, 70, 18, 67, 11, 29, 40, 98, 6, 22, 59, 47, 39, 72, 30, 77, 10, 34, 82, 92, 123, 43, 4, 13, 91, 25, 9, 46, 48, 52, 74, 2, 85, 86, 96, 119, 37, 62, 23, 27, 121, 5, 1, 35, 44, 57, 58, 69, 79, 115, 117, 118, 109, 102, 28, 56, 87, 93, 120, 124, 66, 53, 26, 84, 106, 107, 20, 122, 81, 116, 55, 89, 78, 112, 21, 32, 71, 61, 3, 68, 33, 50, 63, 14, 38, 105, 94, 97, 12, 100, 103, 64, 42, 88, 76, 73, 16, 15, 113, 95, 51, 90, 83, 108, 75, 36], 
112: [104, 114, 7, 111, 60, 54, 29, 8, 80, 99, 25, 24, 49, 98, 19, 110, 72, 6, 96, 59, 34, 18, 47, 67, 125, 46, 74, 65, 77, 82, 68, 39, 21, 5, 31, 79, 117, 41, 48, 2, 86, 119, 22, 92, 10, 123, 124, 23, 37, 53, 84, 121, 13, 58, 69, 4, 35, 44, 109, 118, 9, 30, 52, 32, 93, 27, 26, 43, 20, 107, 122, 57, 91, 1, 115, 103, 14, 71, 87, 66, 106, 76, 12, 50, 63, 55, 61, 56, 120, 51, 15, 64, 97, 3, 33, 102, 90, 94, 113, 17, 116, 100, 36, 42, 88, 108, 73, 95, 83], 
113: [54, 104, 60, 114, 98, 125, 110, 11, 40, 7, 80, 99, 45, 111, 86, 34, 24, 70, 39, 101, 25, 46, 2, 10, 65, 17, 47, 29, 49, 67, 106, 72, 44, 52, 74, 9, 119, 59, 22, 92, 82, 32, 62, 37, 5, 69, 1, 4, 79, 115, 118, 48, 85, 6, 77, 71, 61, 93, 120, 27, 66, 84, 107, 121, 58, 35, 57, 41, 96, 81, 102, 105, 28, 53, 122, 13, 31, 109, 3, 12, 103, 100, 55, 14, 21, 63, 89, 112, 124, 23, 78, 87, 73, 15, 97, 33, 50, 116, 76, 16, 95, 88, 94, 108, 90, 36], 
114: [104, 45, 111, 54, 60, 8, 7, 80, 110, 125, 19, 99, 101, 65, 70, 11, 29, 40, 67, 49, 98, 9, 34, 92, 18, 39, 47, 84, 25, 85, 86, 59, 22, 10, 82, 123, 77, 1, 31, 35, 91, 115, 2, 30, 41, 48, 52, 46, 74, 96, 119, 78, 53, 62, 43, 58, 4, 5, 13, 57, 69, 72, 44, 79, 109, 117, 118, 14, 28, 87, 124, 93, 120, 23, 26, 66, 20, 27, 37, 107, 106, 121, 122, 68, 55, 89, 21, 102, 105, 112, 32, 56, 61, 71, 17, 33, 100, 63, 38, 15, 88, 50, 103, 116, 16, 42, 97, 113, 3, 51, 95, 94, 76, 75, 73, 90, 83, 36, 108], 
115: [54, 114, 80, 101, 45, 60, 7, 110, 104, 111, 25, 34, 123, 8, 19, 99, 77, 82, 70, 11, 49, 29, 40, 125, 98, 1, 79, 2, 9, 74, 86, 10, 65, 22, 92, 47, 18, 24, 39, 14, 20, 107, 5, 31, 117, 46, 30, 41, 85, 59, 61, 28, 32, 13, 44, 57, 58, 69, 91, 109, 6, 48, 52, 89, 56, 87, 124, 43, 122, 84, 106, 4, 72, 118, 17, 100, 55, 71, 120, 93, 62, 23, 26, 27, 53, 121, 37, 15, 81, 38, 21, 78, 102, 95, 12, 116, 105, 112, 90, 16, 88, 113, 97, 68, 50, 103, 51, 73, 76, 75, 83, 36], 
116: [111, 60, 49, 45, 54, 114, 92, 125, 19, 80, 110, 104, 18, 11, 67, 98, 8, 7, 99, 101, 74, 86, 10, 77, 47, 24, 70, 40, 29, 26, 62, 66, 5, 72, 91, 44, 2, 46, 48, 22, 34, 82, 39, 124, 1, 31, 117, 4, 35, 57, 79, 109, 30, 85, 119, 96, 59, 65, 102, 56, 71, 120, 23, 37, 43, 121, 107, 13, 58, 115, 6, 25, 50, 63, 53, 122, 84, 12, 100, 14, 32, 20, 106, 64, 94, 68, 81, 103, 89, 21, 78, 105, 28, 42, 38, 112, 95, 36, 75, 73, 113, 3, 83, 108, 51, 15, 16, 76], 
117: [54, 80, 45, 104, 60, 114, 34, 82, 11, 29, 8, 19, 7, 110, 111, 24, 18, 40, 49, 67, 99, 101, 59, 77, 123, 47, 98, 125, 31, 96, 10, 65, 92, 22, 70, 43, 122, 44, 57, 79, 35, 115, 6, 52, 9, 46, 2, 30, 68, 56, 26, 5, 91, 4, 58, 72, 1, 109, 25, 41, 48, 85, 74, 38, 93, 120, 20, 27, 53, 66, 84, 23, 106, 107, 13, 69, 21, 78, 89, 112, 28, 37, 121, 116, 14, 55, 63, 61, 71, 87, 124, 75, 12, 50, 17, 33, 103, 100, 102, 51, 90, 64, 15, 94, 16, 42, 88, 95, 81, 83, 108, 36], 
118: [54, 49, 60, 114, 67, 99, 111, 45, 104, 59, 11, 40, 29, 125, 8, 19, 80, 7, 101, 110, 77, 18, 24, 39, 47, 70, 98, 44, 52, 48, 96, 82, 92, 123, 62, 20, 69, 58, 86, 9, 2, 41, 74, 119, 85, 65, 22, 32, 71, 23, 26, 27, 37, 66, 53, 72, 5, 35, 4, 6, 25, 30, 46, 61, 87, 93, 84, 106, 121, 13, 31, 91, 115, 33, 103, 55, 56, 28, 124, 43, 122, 15, 3, 17, 50, 63, 21, 105, 112, 120, 42, 12, 81, 14, 38, 78, 89, 102, 90, 64, 95, 113, 68, 36, 73, 88, 94, 108, 76, 16, 51, 83], 
119: [45, 60, 114, 67, 29, 125, 7, 8, 80, 110, 99, 101, 104, 111, 39, 11, 98, 19, 10, 24, 70, 40, 49, 109, 6, 48, 2, 30, 65, 77, 92, 47, 18, 53, 35, 57, 25, 41, 86, 74, 59, 22, 34, 82, 123, 66, 37, 121, 5, 4, 13, 31, 72, 1, 79, 91, 118, 46, 52, 9, 85, 12, 63, 56, 28, 71, 87, 124, 32, 62, 23, 26, 27, 122, 58, 69, 38, 89, 105, 61, 93, 120, 84, 107, 20, 106, 94, 100, 21, 112, 73, 3, 50, 33, 68, 17, 81, 116, 14, 102, 16, 97, 113, 103, 51, 90, 64, 42, 75, 88, 83, 108, 36], 
120: [54, 60, 114, 80, 111, 11, 19, 7, 99, 110, 45, 104, 123, 67, 125, 8, 101, 52, 22, 63, 47, 39, 18, 40, 49, 29, 98, 62, 13, 48, 41, 74, 9, 34, 82, 70, 20, 53, 66, 4, 79, 117, 25, 30, 119, 59, 65, 10, 87, 23, 69, 115, 46, 6, 85, 96, 12, 81, 21, 56, 37, 43, 106, 107, 121, 57, 58, 72, 1, 5, 31, 44, 109, 118, 33, 17, 116, 89, 105, 78, 71, 27, 26, 84, 122, 15, 50, 100, 14, 38, 93, 124, 28, 90, 95, 113, 3, 103, 112, 73, 42, 64, 97, 83, 16, 94, 36, 75, 76, 108], 
121: [29, 54, 111, 60, 114, 19, 7, 99, 110, 45, 104, 40, 49, 125, 80, 101, 9, 92, 47, 18, 39, 24, 70, 11, 67, 98, 6, 2, 85, 119, 59, 34, 82, 71, 37, 1, 4, 44, 91, 31, 25, 48, 52, 86, 96, 77, 123, 12, 23, 122, 84, 5, 57, 58, 72, 118, 30, 41, 74, 68, 14, 62, 20, 35, 79, 117, 115, 109, 50, 105, 112, 56, 61, 87, 93, 124, 120, 26, 27, 66, 107, 3, 116, 17, 33, 81, 103, 55, 63, 21, 38, 102, 28, 32, 100, 16, 94, 113, 51, 64, 95, 42, 83, 76, 73, 90, 15, 97, 36, 108, 75], 
122: [60, 29, 98, 104, 54, 114, 49, 7, 8, 19, 80, 99, 101, 110, 45, 111, 77, 82, 11, 40, 125, 1, 65, 34, 123, 47, 70, 67, 107, 91, 117, 46, 2, 10, 22, 24, 18, 56, 44, 5, 79, 6, 25, 48, 52, 86, 85, 96, 119, 92, 61, 26, 66, 121, 13, 57, 31, 35, 109, 115, 41, 9, 30, 74, 103, 81, 63, 14, 78, 93, 43, 23, 37, 4, 118, 69, 68, 55, 71, 28, 124, 62, 20, 27, 50, 17, 100, 38, 112, 32, 87, 120, 51, 12, 33, 116, 21, 89, 102, 75, 16, 97, 3, 76, 64, 94, 88, 113, 15, 108, 83], 
123: [54, 60, 45, 114, 8, 19, 7, 99, 110, 101, 104, 111, 18, 11, 67, 80, 52, 82, 24, 70, 29, 40, 49, 125, 115, 41, 65, 22, 59, 92, 34, 47, 39, 26, 20, 69, 117, 25, 46, 2, 9, 74, 30, 96, 10, 77, 61, 120, 122, 1, 13, 35, 31, 58, 118, 6, 48, 85, 119, 33, 17, 21, 56, 23, 27, 37, 43, 53, 106, 107, 5, 44, 4, 57, 79, 109, 63, 38, 28, 87, 93, 62, 84, 121, 14, 55, 71, 32, 124, 97, 3, 50, 68, 12, 81, 103, 100, 112, 78, 89, 105, 51, 90, 64, 95, 88, 16, 15, 83, 75, 73, 108, 76, 36], 
124: [60, 114, 8, 7, 45, 111, 54, 49, 80, 104, 77, 70, 40, 98, 125, 110, 99, 101, 65, 59, 29, 109, 2, 86, 96, 119, 10, 34, 92, 18, 39, 24, 47, 84, 69, 57, 1, 115, 6, 30, 46, 48, 74, 22, 82, 123, 71, 32, 20, 23, 26, 53, 4, 31, 35, 79, 91, 5, 118, 25, 52, 116, 55, 112, 27, 37, 121, 122, 106, 72, 117, 12, 14, 89, 102, 61, 28, 56, 93, 62, 66, 43, 50, 21, 38, 78, 105, 120, 88, 94, 68, 17, 103, 63, 108, 51, 64, 42, 97, 100, 90, 16, 113, 3, 75, 95, 36, 73, 76, 83], 
125: [114, 111, 104, 54, 60, 70, 49, 19, 99, 45, 47, 11, 98, 8, 80, 7, 101, 110, 10, 65, 24, 18, 40, 29, 67, 69, 48, 74, 119, 59, 77, 92, 39, 23, 26, 37, 5, 31, 46, 86, 96, 82, 22, 34, 123, 55, 66, 44, 1, 4, 57, 91, 118, 52, 2, 9, 25, 30, 85, 61, 71, 93, 20, 62, 106, 121, 122, 13, 58, 72, 79, 35, 109, 115, 117, 63, 102, 120, 32, 124, 53, 84, 107, 17, 116, 103, 14, 38, 78, 89, 21, 105, 56, 28, 87, 113, 12, 50, 81, 112, 73, 42, 15, 3, 33, 68, 64, 16, 88, 94, 97, 76, 90, 36, 108, 75], 
}
solution1 = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
solution2 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 70, 71, 77, 79, 80, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
solution3 = [5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution4 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 65, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution5 = [5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution16 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
solution6 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 65, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution7 = [1, 5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution8 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution9 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 68, 70, 71, 77, 79, 80, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
solution10 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
solution11 = [1, 5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution12 = [1, 5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
solution13 = [1, 5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution14 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 93, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
solution15 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 70, 71, 77, 79, 80, 93, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
solution17 = [5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
solution18 = [5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
solution19 = [5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
solution20 = [5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s21 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 93, 99, 101, 110, 111, 114, 115, 117, 121, 122, 123, 125]
s22 = [5, 7, 9, 11, 13, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s23 = [1, 7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
s24 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 70, 71, 77, 79, 80, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
s25 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 49, 54, 68, 70, 71, 77, 79, 80, 98, 99, 101, 110, 114, 115, 117, 121, 122, 125]
s26 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s27 = [1, 5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
s28 = [1, 7, 9, 11, 13, 19, 22, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
s29 = [1, 5, 7, 9, 11, 13, 19, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
s30 = [5, 7, 9, 11, 13, 19, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
s31 = [5, 7, 9, 11, 13, 17, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 67, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s32 = [1, 5, 7, 9, 11, 19, 25, 29, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 121, 122, 125]
s33 = [1, 2, 5, 7, 9, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 70, 71, 77, 79, 80, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s34 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 93, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
s35 = [1, 5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
s36 = [5, 7, 9, 11, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 68, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 121, 122, 125]
s37 = [5, 7, 9, 11, 17, 19, 25, 29, 31, 34, 40, 44, 45, 49, 52, 54, 55, 65, 66, 67, 70, 77, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
s38 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s39 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 85, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s40 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 40, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 83, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123]
s41 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 40, 44, 48, 49, 54, 68, 70, 71, 77, 79, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s42 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 85, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s43 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s44 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s45 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 83, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123]
s46 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 83, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123]
s47 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s48 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 85, 92, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s49 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s50 = [1, 2, 5, 7, 9, 11, 18, 19, 25, 29, 31, 34, 44, 48, 49, 54, 68, 70, 71, 77, 79, 80, 85, 93, 99, 101, 110, 114, 115, 117, 121, 122, 123, 125]
s51 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
s52 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
s53 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
s54 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]
s55 = [1, 2, 5, 7, 9, 11, 18, 25, 29, 31, 34, 44, 48, 49, 54, 60, 68, 70, 71, 77, 79, 80, 82, 83, 85, 93, 101, 110, 114, 115, 117, 121, 122, 123]

solutionDatabase = [solution1, solution2, solution3, solution4, solution5, solution6, solution7, solution8, solution9, solution10, solution11, solution12, solution13, solution14, solution15, solution16, solution17]
solutionDatabase += [s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, s41, s42, s43, s44, s45, s46, s47, s48, s49, s50, s51, s52, s53, s54, s55]
#'''
#topExclusive(freqToClique(freqGet(very_first_ordering)), graph)

#base = [54, 60, 114, 7, 125, 80, 29, 45, 104, 99, 110]
'''for sol in solutionDatabase:
    flag = True
    for member in base:
        if member not in sol:
            flag = False
            break
    if flag:
        print(sol)'''
#antiNeighbors = nonNeighClique(base, antiG)
#toDel = base + antiNeighbors
#graphRedux = delete(toDel, graph)
#print(len(graphRedux.keys()))
##graph2, mapkey = mapping(graphRedux)
#topExclusive(freqToClique(freqGet(testSample(graphRedux, 100, True, 125))), graphRedux)

#base += [40, 44, 31, 65, 70, 49, 122, 96, 34, 35, 11, 50, 1, 117]
#antiNeighbors = nonNeighClique(base, antiG)
#toDel = base + antiNeighbors
#graphRedux = delete(toDel, graph)
#print(len(graphRedux.keys()))
#topExclusive(freqToClique(freqGet(testSample(graphRedux, 100, True, 125))), graphRedux)
#print(graphRedux)
#print(len(base))

#base += [68, 5, 9, 103]
#antiNeighbors = nonNeighClique(base, antiG)
#toDel = base + antiNeighbors
#graphRedux = delete(toDel, graph)
#print(graphRedux)
#print(len(graphRedux.keys()))
#topExclusive(freqToClique(freqGet(testSample(graphRedux, 100, True, 125))), graphRedux)
'''
Something's wrong with my topExclusive (or something in between):
printed:
2: [123, 52, 77, 63], 
66: [52, 77, 55, 63], 
77: [52, 55, 123, 2, 66], 
52: [123, 77, 2, 55, 63, 66], 
55: [77, 52, 123, 66], 
123: [52, 2, 77, 55, 63], 
63: [52, 123, 2, 66], 
52: []
123: []
77: []
...when 52, 123, and 77 all clearly have their own printed statements already
...

As I just found out it was the same for above:
33: [9, 68, 52, 63, 103, 123, 2, 66, 48, 55], 
2: [68, 63, 123, 5, 9, 48, 103, 77, 52, 71, 33, 116], 
68: [5, 9, 63, 2, 66, 77, 103, 123, 52, 71, 48, 55, 33, 116], 
5: [68, 77, 9, 71, 2, 63, 66, 123, 103, 52, 48, 55, 116], 
71: [5, 68, 48, 77, 9, 2, 63, 66, 123, 116, 55], 
9: [68, 123, 5, 52, 63, 66, 77, 103, 2, 55, 33, 48, 71], 
103: [68, 52, 9, 5, 123, 63, 2, 66, 77, 33, 55, 116], 
66: [68, 5, 9, 63, 77, 103, 52, 71, 48, 55, 33, 116], 
77: [5, 68, 9, 71, 66, 2, 123, 103, 52, 48, 55, 116], 
48: [68, 71, 2, 63, 5, 9, 66, 123, 77, 116, 33], 
52: [103, 9, 68, 123, 5, 55, 33, 66, 63, 77, 2], 
55: [9, 68, 52, 77, 103, 66, 123, 5, 71, 33], 
116: [5, 68, 71, 48, 63, 66, 77, 2, 103], 
123: [9, 68, 2, 5, 52, 63, 77, 103, 48, 71, 33, 55], 
63: [68, 2, 5, 9, 48, 66, 123, 103, 52, 71, 33, 116], 
68: []
5: []
9: []
103: []
'''
#base += [77, 123, 52]
#antiNeighbors = nonNeighClique(base, antiG)
#toDel = base + antiNeighbors
#graphRedux = delete(toDel, graph)
#print(graphRedux)
#print(len(graphRedux.keys()))

#base += [2, 55]
#print(base)
#print(len(base))
#antiNeighbors(base, graph)
#checkClique(base, graph, True)























