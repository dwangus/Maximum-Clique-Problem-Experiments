import math
import time
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

def nonNeigh(originNode, graph):
    unneighbors = []
    neighborhood = graph[originNode]
    for key in graph.keys():
        if key != originNode:
            if key not in neighborhood:
                unneighbors.append(key)
    return unneighbors

def closedNeighborhood(originNode, graph):
    unneighbors = []
    neighborhood = graph[originNode]
    for key in graph.keys():
        if key != originNode:
            if key not in neighborhood:
                unneighbors.append(key)
                for n in graph[key]:
                    graph[n].remove(key)
                graph[key] = []
    print("{0}'s Un-neighbors: {1}".format(originNode, unneighbors))
    return graph

def vectorFormation(startpoint, endpoint):
    vectorResult = []
    for dimension in range(len(endpoint)):
        vectorResult.append(float(endpoint[dimension]) - float(startpoint[dimension]))
    return vectorResult

def vectorAddition(vector_list):
    result = []
    for dimension in range(len(vector_list[0])):
        temp = 0.0
        for vector in vector_list:
            temp += float(vector[dimension])
        result.append(temp)
    return result

def vectorNegation(vector_list):
    result = []
    for dimension in range(len(vector_list[0])):
        temp = 0.0
        for vector in vector_list:
            temp -= float(vector[dimension])
        result.append(temp)
    return result

def vectorSubtraction(subXFromThis, X):
    for dimension in range(len(subXFromThis)):
        subXFromThis[dimension] -= X[dimension]
    return subXFromThis

def vectorMagnitude(vector):
    temp = 0.0
    for dimension in range(len(vector)):
        temp += float(vector[dimension])**2
    return math.sqrt(temp)

def unitVector(vector):
    magnitude = vectorMagnitude(vector)
    unit = []
    for dimension in range(len(vector)):
        unit.append(float(vector[dimension])/magnitude)
    return unit

def dot(A, B):
    dotScalar = 0.0
    for i in range(len(A)):
        dotScalar += A[i]*B[i]
    return dotScalar

def vectorProj(Aon, toB):
    toUnitB = unitVector(toB)
    dotScalar = dot(Aon, toUnitB)
    for i in range(len(toUnitB)):
        toUnitB[i] *= dotScalar
    return toUnitB

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

def update(coords, delNodes, memoized):
    '''
    for deleting in delNodes:
        for node in adjList[deleting]:
            adjList[node].remove(deleting)
        adjList[deleting] = []
    
    changes = {}
    for point1 in coords.keys():
        cur_point = coords[point1]
        vecList = []
        neighbors = adjList[point1]
        for point2 in neighbors:
            vecList.append(vectorFormation(cur_point, coords[point2]))
        if len(neighbors) == 0:
            changes[point1] = [0.0]*len(cur_point)
        else:
            changes[point1] = unitVector(vectorAddition(vecList))
    average = []
    for key in changes.keys():
        average.append(changes[key])
    averageVec = vectorNegation(average)
    similarity = []
    for key in changes.keys():
        sim = dot(changes[key], averageVec)
        similarity.append(sim)
    '''
    average = []
    for vector in memoized:
        average.append(vector)
    averageVec = vectorNegation(average)
    similarity = []
    for vector in memoized:
        sim = dot(vector, averageVec)
        similarity.append(sim)
    
    minIndex = 0
    for node in range(1, len(similarity)):
        if (node+1) not in delNodes:
            if similarity[node] < similarity[minIndex]:
                minIndex = node
            elif similarity[node] == similarity[minIndex]:
                continue
                #print("Symmetry")
                #print("Node {0}".format(node+1))
    
    nextToDel = minIndex + 1
    #solution = [3, 8, 21, 26, 30, 31, 34, 37, 41, 45, 58, 63, 70, 72, 84, 87, 90, 92, 96, 97, 99, 122, 129, 131, 136, 138, 147, 152, 161, 162, 163, 165, 177, 183, 186, 191, 197, 203, 207, 212, 214, 227, 235, 241]
    #solution = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
    '''
    if nextToDel == 68:
        for sim in range(len(similarity)):
            print("Node {0} Update: {1}".format(sim+1, similarity[sim]))
    #if nextToDel == 36:'''
    if True:
        order = []
        for sim in range(len(similarity)):
            order.append((sim+1, similarity[sim]))
            order = sorted(order, key=itemgetter(1), reverse=True)
        '''for node in order:
            if node[0] in solution:
                print("***{0}".format(node))
            else:
                print(node)
    #'''
    return nextToDel, similarity[minIndex], order

def iterate(graphOrig, paused = False, fromLastTime = [], retOrder = False):
    start_time = time.time()
    solution125 = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
    graph = {}
    for key in graphOrig.keys():
        copy = []
        for neighbor in graphOrig[key]:
            copy.append(neighbor)
        graph[key] = copy
    
    coords = equidistant_vectors(len(graph.keys()), 5000)
    toDel = []

    #vector_build_time = time.time()
    dims = len(coords[1])
    #summation vector of N-1 dimensions, for each node in the graph
    memoized = [[0.0 for i in range(dims)] for j in range(len(graph.keys()))]
    unitMemoized = [[0.0 for i in range(dims)] for j in range(len(graph.keys()))]
    for point1 in coords.keys():
        cur_point = coords[point1]
        vecList = []
        neighbors = graph[point1]
        for point2 in neighbors:
            vecList.append(vectorFormation(cur_point, coords[point2]))
        if len(neighbors) == 0:
            continue
        else:
            memoized[point1 - 1] = vectorAddition(vecList)
            unitMemoized[point1 - 1] = unitVector(memoized[point1-1])
    #print("Time to build/sum/unit vectors: {0}".format(time.time() - vector_build_time))
    
    if paused:
        toDel += fromLastTime
        for deleting in toDel:
            cur_point = coords[deleting]
            for node in graph[deleting]:
                vectorToSubtract = vectorFormation(coords[node], cur_point)
                graph[node].remove(deleting)
                memoized[node-1] = vectorSubtraction(memoized[node-1], vectorToSubtract)
                unitMemoized[node - 1] = unitVector(memoized[node-1])
            graph[deleting] = []
            memoized[deleting-1] = [0.0]*dims
            unitMemoized[deleting - 1] = [0.0]*dims
    ##################
    error = -10000.0
    i = 0
    while error < (-0.00000001):
        '''
        adjList = {}
        for key in graph.keys():
            copy = []
            for neighbor in graph[key]:
                copy.append(neighbor)
            adjList[key] = copy'''
        nextToDel, newError, order = update(coords, toDel, unitMemoized)
        if retOrder:
            return order
        if newError > (-0.00000001):
            #print("That's all, folks!")
            for node in order:
                print(node)
            break
        print("Deleting: {0}".format(nextToDel))
        '''if nextToDel in solution125:
            print("Doh!")
        if nextToDel == 68:
            for key in coords.keys():
                if key not in graph[54] and key not in toDel:
                    print(key)
            break'''
        #if nextToDel == 36:
        #    break
        toDel.append(nextToDel)
        #print(newError)
        error = newError

        cur_point = coords[nextToDel]
        for node in graph[nextToDel]:
            vectorToSubtract = vectorFormation(coords[node], cur_point)
            graph[node].remove(nextToDel)
            memoized[node-1] = vectorSubtraction(memoized[node-1], vectorToSubtract)
            unitMemoized[node-1] = unitVector(memoized[node-1])
        graph[nextToDel] = []
        memoized[nextToDel-1] = [0.0]*dims
        unitMemoized[nextToDel - 1] = [0.0]*dims
        #i += 1
        #if i == 19:
        #    print("#Nodes left: {0}".format(len(graph.keys()) - len(toDel)))
        #    i = 0

        
        #error = 0.0
    ##################
    clique = []
    for key in graphOrig.keys():
        if key not in toDel:
            clique.append(key)
    
    flag = True
    for member in range(len(clique)-1):
        neighbors = graphOrig[clique[member]]
        for other in range(member+1, len(clique)):
            if clique[other] not in neighbors:
                #print(clique[member])
                #print(clique[other])
                flag = False

    if flag:
        for node in graphOrig.keys():
            if node not in clique:
                neighbors = graphOrig[node]
                flag = True
                for member in clique:
                    if member not in neighbors:
                        flag = False
                        break
                if flag:
                    #print("Adding: {0}".format(node))
                    clique.append(node)
        print("Extracted Clique: {0}".format(clique))
        print("Time Spent: {0}".format(time.time() - start_time))
        return clique
    else:
        print("Extracted Clique: {0}".format(clique))
        print("Time Spent: {0}".format(time.time() - start_time))
        print("Didn't work")
        return []

def main(filename):
    graph = graphConv(filename)
    #graph = closedNeighborhood(x, graph)
    #[10, 30, 91, 94, 119]
    return iterate(graph)
    
#def sampleTest(graph):
#    return iterate(graph)

#'''
def metaMain(filename):
    graph = graphConv(filename)
    
    firstClique = iterate(graph)
    print(len(firstClique))
    order = iterate(graph, False, [], True)
    for node in order:
        print(node)
    weightTowards = []
    for item in order:
        if item[0] in firstClique:
            weightTowards.append(item[0])
            break
    print(weightTowards)
    
    delFirst = nonNeigh(weightTowards[0], graph)
    secondClique = iterate(graph, True, delFirst)
    print(len(secondClique))
    order = iterate(graph, True, delFirst, True)
    for node in order:
        print(node)
    for item in order:
        if item[0] in secondClique:
            if item[0] not in weightTowards and item[0] not in delFirst:
                weightTowards.append(item[0])
                break
    print(weightTowards)
    
    delFirst += nonNeigh(weightTowards[1], graph)
    thirdClique = iterate(graph, True, delFirst)
    print(len(thirdClique))
    order = iterate(graph, True, delFirst, True)
    for node in order:
        print(node)
    for item in order:
        if item[0] in thirdClique:
            if item[0] not in weightTowards and item[0] not in delFirst:
                weightTowards.append(item[0])
                break
    print(weightTowards)

    delFirst += nonNeigh(weightTowards[2], graph)
    fourthClique = iterate(graph, True, delFirst)
    print(len(fourthClique))
    order = iterate(graph, True, delFirst, True)
    for node in order:
        print(node)
    for item in order:
        if item[0] in fourthClique:
            if item[0] not in weightTowards and item[0] not in delFirst:
                weightTowards.append(item[0])
                break
    print(weightTowards)

    delFirst += nonNeigh(weightTowards[3], graph)
    fifthClique = iterate(graph, True, delFirst)
    print(len(fourthClique))
    order = iterate(graph, True, delFirst, True)
    for node in order:
        print(node)
    '''for item in order:
        if item[0] in fourthClique:
            if item[0] not in weightTowards and item[0] not in delFirst:
                weightTowards.append(item[0])
                break
    print(weightTowards)
    #104 is not connected to 101 or 18
#'''

#time spent finding clique of size 33 -- 52 seconds --> but on test8.py for some reason, even though error margins are nearly identical...
#time spent finding clique of size 39 -- 826 seconds --> ditto
#...it might be rounding issues...

#Each doubling of size of the graph contributes to about x8.35 in running time
file1 = "c125.txt" #time spent finding clique of size 32 -- 2.75 seconds --> Best known: 34 (also global optimum)
#solution = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99,]
#solution += [103, 104, 110, 111, 114, 117, 122, 125]
#vs. Extracted Clique = [1, 2, 5, 7, 11, 17, 18, 19, 25, 29, 31, 34, 40, 44, 45, 48, 54, 69, 70, 71, 77, 79, 80, 99, 101, 110, 114, 117, 122, 123, 125, 115]
#Shared = [7, 11, 19, 25, 29, 34, 40, 44, 54, 70, 79, 80, 99, 110, 114, 117, 122, 125], of size 18
'''
file2 = "c250.txt" #time spent finding clique of size 37 -- 23 seconds --> Best known: 44 (also global optimum)
#solution = [3, 8, 21, 26, 30, 31, 34, 37, 41, 45, 58, 63, 70, 72, 84, 87, 90, 92, 96, 97, 99, 122, 129, 131, 136, 138, 147, 152, 161, 162, 163, 165, 177, 183, 186, 191, 197, 203, 207, 212, 214, 227, 235, 241]
file3 = "c500.txt" #time spent finding clique of size 45 -- 192 seconds --> Best known: 57 (not global optimum)
file4 = "c1000.txt" #predicted that it'll take ~1600 seconds
file5 = "hamming8-4.txt" #time spent finding clique of size 0 -- 7.75 seconds --> Best known: 16 (also global optimum)
file6 = "brock200_2.txt" #time spent finding clique of size 8 -- 7.9 seconds --> Best known: 12 (also global optimum)
fileName = file1
clique = main(fileName)
#print(clique)
size = len(clique)
#graph1 = {1:[2,3,4,6], 2:[1,3,5,7], 3:[1,2,4,5], 4:[1,3,5,6,7], 5:[2,3,4,6,7], 6:[1,4,5,7], 7: [2,4,5,6]}
#graph2 = {1:[2,3,4], 2:[1,3], 3:[1,2], 4:[1]}
#sampleGraph = graph2
#size = sampleTest(sampleGraph)
print("Returned Clique Size: {0}".format(size))'''
#metaMain(file1)
main(file1)
'''
graph = graphConv(file1)
print(graph[45])
#print(graph[45])'''
'''
graph = graphConv(file1)
clique = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 45, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 96, 98, 99, 103, 104, 110, 114, 117, 122, 125]
for node in graph.keys():
    if node not in clique:
        neighbors = graph[node]
        flag = True
        for member in clique:
            if member not in neighbors:
                flag = False
                print("Nope, {0}".format(node))
                break
        if flag:
            print("Adding: {0}".format(node))
            clique.append(node)
print(len(clique))
#'''

'''
Deleting: 36
Deleting: 83
Deleting: 108
Deleting: 76
Deleting: 51
Deleting: 90
Deleting: 64
Deleting: 15
Deleting: 95
Deleting: 68
Deleting: 42
Deleting: 97
Deleting: 88
Deleting: 16
Deleting: 94
Deleting: 75
Deleting: 102
Deleting: 55
Deleting: 27
Deleting: 33
Deleting: 43
Deleting: 112
Deleting: 87
Deleting: 56
Deleting: 100
Deleting: 21
Deleting: 73

Deleting: 121

Deleting: 107
Deleting: 105
Deleting: 3
Deleting: 14
Deleting: 113
Deleting: 61
Deleting: 109
Deleting: 50
Deleting: 124
Deleting: 84
Deleting: 72
Deleting: 32
Deleting: 37
Deleting: 4
Deleting: 57
Deleting: 78
Deleting: 20
Deleting: 116
Deleting: 28
Deleting: 12
Deleting: 53
Deleting: 120
Deleting: 106
Deleting: 65
Deleting: 98
Deleting: 46

Deleting: 115

Deleting: 23
Deleting: 62
Deleting: 30
Deleting: 118
Deleting: 74
Deleting: 52
Deleting: 39
Deleting: 111
Deleting: 6
Deleting: 58
Deleting: 103
Deleting: 91
Deleting: 89
Deleting: 38
Deleting: 86
Deleting: 63
Deleting: 81
Deleting: 119
Deleting: 10

Deleting: 49

Deleting: 92
Deleting: 26
Deleting: 13
Deleting: 8
Deleting: 82
Deleting: 104
Deleting: 85
Deleting: 60
Deleting: 22
Deleting: 66
Deleting: 35
Deleting: 41
Deleting: 59
Deleting: 67
Deleting: 93

Deleting: 9

Deleting: 96
Deleting: 47
Deleting: 24

... And then I only added in 115, as opposed to 9, 49, 115, 121
... Because I left in 69 at some point...
'''

























