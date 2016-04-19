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

def distance(point_list1, point_list2):
    temp = 0.0
    for i in range(len(point_list1)):
        temp += (float(point_list1[i]) - float(point_list2[i]))**2
    return math.sqrt(temp)

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

def vectorSubtraction(vector_list):
    result = []
    for dimension in range(len(vector_list[0])):
        temp = 0.0
        for vector in vector_list:
            temp -= float(vector[dimension])
        result.append(temp)
    return result

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

def update(coords, adjList):
    #
    checking = 82
    #68, 55, 33, 98, 52, 111, 103, 49, 13, 22, 66, 93, 104
    delNodes = []
    delNodes += [36, 83, 108, 76, 51, 90, 64, 15, 95, 68, 42, 97, 88, 16, 94, 75, 102, 55, 27, 33, 43, 112, 87, 56, 100, 21, 73, 121, 107, 105, 3, 14]
    delNodes += [113, 61, 109, 50, 124, 84, 72, 32, 37, 4, 57, 78, 20, 116, 28, 12, 53, 120, 106, 65, 98, 46, 115, 23, 62, 30, 118, 74, 52, 39, 111, 6]
    delNodes += [58, 103, 91, 89, 38, 86, 63, 81, 119, 10, 49, 92, 26]
    #delNodes += [8, 13, 47, 69, 101, 60, 59, 22, 66, 41, 35, 93, 1, 99, 114, 85, 104]
    #delNodes += [25, 11, 70, 9, 7, 19, 110, 96, 29, 67, 54, 79, 80, 125, 122, 117, 34, 40, 44]
    #delNodes += [5, 2, 77, 71, 17, 24, 31, 123, 82, 18, 48, 45]
    #hello = [25, 11, 70, 9, 7, 19, 110, 96, 29, 67, 54, 79, 80, 125, 122, 117, 34, 40, 44] + [5, 2, 77, 71, 17, 24, 31, 123, 82, 18, 48, 45]
    #hello += [49, 121]
    #print(len(hello))
    edgeSimilarity = []
    #toDel = []
    #for node in toDel:
    for deleting in delNodes:
        for node in adjList[deleting]:
            adjList[node].remove(deleting)
    #for neighbor in toDel:
    #    adjList[checking].remove(neighbor)
        adjList[deleting] = []
    edgeVectors = []
    #
    changes = {}
    for point1 in coords.keys():
        cur_point = coords[point1]
        vecList = []
        neighbors = adjList[point1]
        for point2 in neighbors:
            vecList.append(vectorFormation(cur_point, coords[point2]))
        #
        edgeVectors.append(vecList)
        #
        if len(vecList) == 0:
            changes[point1] = [0.0]*len(cur_point)
        else:
            changes[point1] = unitVector(vectorAddition(vecList))

    #'''
    average = []
    for key in changes.keys():
        average.append(changes[key])
    averageVec = vectorSubtraction(average)
    similarity = []
    #for node in range(len(edgeVectors)):
    #    node_id = node + 1
    #    for edge in range(len(edgeVectors[node])):
    #        print("Node {0} (some edge) similarity: {1}".format(node_id, dot(edgeVectors[node][edge], averageVec)))
            #print("Node {0} (some edge) similarity: {1}".format(node_id, dot(edgeVectors[node][edge], changes[node_id])))

    for edge in range(len(edgeVectors[checking-1])):
        edgeSim = dot(edgeVectors[checking-1][edge], averageVec)
        edgeSimilarity.append(edgeSim)
        #print("Edge {0}-{2} similarity: {1}".format(checking, edgeSim, adjList[checking][edge]))

    for key in changes.keys():
        sim = dot(changes[key], averageVec)
        similarity.append(sim)
        print("Node {0} Update: {1}".format(key, sim))

    minIndex = 0
    for node in range(1, len(similarity)):
        if (node+1) not in delNodes:
            if node+1 == 13:
                print(similarity[node])
                print(similarity[7])
                print(similarity[node] > similarity[7])
            if similarity[node] < similarity[minIndex]:
                minIndex = node
            elif similarity[node] == similarity[minIndex]:
                print("Symmetry")
                print("Node {0}".format(node+1))
    #minEdge = min(edgeSimilarity)
    #minEdgeIndex = edgeSimilarity.index(minEdge)
    solution = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
    print("Min. Similarity for Node {0}: {1} / {2}".format(minIndex + 1, similarity[minIndex], len(delNodes)))
    if (minIndex + 1) in solution:
        print("AARGH!")
    #print("Min. Edge Similarity for Edge {0}-{2}: {1}".format(checking, min(edgeSimilarity), adjList[checking][minEdgeIndex]))
    '''
    if (len(delNodes) == (125-34)):
        clique = []
        for i in range(1, 126):
            if i not in delNodes:
                clique.append(i)
        print(len(clique))
        print(clique)
        flag = True
        for i in range(len(clique)-1):
            neighbors = adjList[clique[i]]
            for j in range(i+1, len(clique)):
                if clique[j] not in neighbors:
                    print(clique[i])
                    print(clique[j])
                    flag = False
                    #break
        print(flag)
    '''
    '''
    added = []
    for node in adjList.keys():
        if node not in hello:
            neighbors = adjList[node]
            flag = True
            for compare in hello:
                if compare not in neighbors:
                    flag = False
                    break
            if flag:
                added.append(node)
    print(added)
    if True:
        flag = True
        for i in range(len(hello)-1):
            neighbors = adjList[hello[i]]
            for j in range(i+1, len(hello)):
                if hello[j] not in neighbors:
                    print(hello[i])
                    print(hello[j])
                    flag = False
                    #break
        print(flag)
    '''
    #'''
    
    '''
    #print(changes)
    denoise = []
    for key in changes.keys():
        denoise.append(changes[key])
    denoising = vectorSubtraction(denoise)
    #print(denoising)

    for key in changes.keys():
        directionVec = changes[key]
        #print(directionVec)
        projection = vectorProj(denoising, directionVec)
        #print(projection)
        changes[key] = vectorAddition([directionVec,projection])
        #print(changes[key])
    '''
    
    for node in changes.keys():
        for dim in range(len(changes.keys())-1):
            coords[node][dim] += float(changes[node][dim])

def equidistant_vectors(N, spacing):
    #start_time = time.time()
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
    #for key in coords.keys():
    #    print("Node #{0}: {1}".format(key, coords[key]))
    #print(time.time() - start_time)
    #I wanna say... initializing N equidistant vectors runs in Average(2.5(n^2)) time
    '''
    for i in range(1, len(coords.keys())):
        point1_coords = coords[i]
        for j in range(i+1, len(coords.keys())+1):
            point2_coords = coords[j]
            dist = distance(point1_coords, point2_coords)
            print("Distance(#{0},#{1}) = {2}".format(i, j, dist))
    '''
    return coords

#graph1 is from Wikipedia page on Max Clique Problem -- max clique is [4,5,6,7]
#graph1 = {1:[2,3,4,6], 2:[1,3,5,7], 3:[1,2,4,5], 4:[1,3,5,6,7], 5:[2,3,4,6,7], 6:[1,4,5,7], 7: [2,4,5,6]}
#graph1 = {1:[2,4,6], 2:[1,5,7], 3:[], 4:[1,5,6,7], 5:[2,4,6,7], 6:[1,4,5,7], 7: [2,4,5,6]}
#adj = graph1
adj = graphConv("c125.txt")
coords = equidistant_vectors(len(adj.keys()), 5000)
#adj = {1:[2,3,4], 2:[1,3], 3:[1,2], 4:[1]}
#coords = equidistant_vectors(4, 5000)
iterations = 1
for i in range(iterations):
    update(coords, adj)
#print("Iterations: {0}\nNode Positions: {1}".format(iterations, coords))
#print("1-2 Distance: {0}".format(distance(coords[1], coords[2])))
#print("1-3 Distance: {0}".format(distance(coords[1], coords[3])))
#print("1-4 Distance: {0}".format(distance(coords[1], coords[4])))
#print("2-3 Distance: {0}".format(distance(coords[2], coords[3])))


def main(filename, guess_K):
    start_time = time.time()
    adjList = graphConv(filename)
    size = len(adjList.keys())
    ################
    spacing = 100.0
    iterations = 40
    ################
    print("Size of Graph: {0} Nodes".format(size))
    print("Iterations: {0}, Initialized Equidistance: {1}".format(iterations, spacing))
    coords = equidistant_vectors(size, spacing)
    for i in range(iterations):
        update(coords, adjList)
    edgeDistances = []
    for i in range(1, len(coords.keys())):
        point1_coords = coords[i]
        for j in range(i+1, len(coords.keys())+1):
            if (j) in adjList[i]:
                point2_coords = coords[j]
                dist = distance(point1_coords, point2_coords)
                edgeDistances.append((i,j,dist))
                #print("Distance(#{0},#{1}) = {2}".format(i, j, dist))
    edgeDistances = sorted(edgeDistances, key=itemgetter(2))
    numEdges = (guess_K)*(guess_K - 1)/2
    for i in range(numEdges):
        print(edgeDistances[i])
    print("Time Spent: {0}".format(time.time() - start_time))

def mainTest(sampleGraph, guess_K):
    start_time = time.time()
    adjList = sampleGraph
    size = len(adjList.keys())
    ################
    spacing = 30000.0
    iterations = 20000
    ################
    print("Size of Graph: {0} Nodes".format(size))
    print("Iterations: {0}, Initialized Equidistance: {1}".format(iterations, spacing))
    coords = equidistant_vectors(size, spacing)
    for i in range(iterations):
        update(coords, adjList)
    edgeDistances = []
    for i in range(1, len(coords.keys())):
        point1_coords = coords[i]
        for j in range(i+1, len(coords.keys())+1):
            if (j) in adjList[i]:
                point2_coords = coords[j]
                dist = distance(point1_coords, point2_coords)
                edgeDistances.append((i,j,dist))
                print("Distance(#{0},#{1}) = {2}".format(i, j, dist))
    edgeDistances = sorted(edgeDistances, key=itemgetter(2))
    numEdges = (guess_K)*(guess_K - 1)/2
    for i in range(numEdges):
        print(edgeDistances[i])
    print("Time Spent: {0}".format(time.time() - start_time))
    return edgeDistances

def metaMainTest(sampleGraph, guess_K):
    numPredictedEdges = (guess_K)*(guess_K - 1)/2
    numEdges = 0
    for node1 in sampleGraph.keys():
        numEdges += len(sampleGraph[node1])
    numEdges /= 2
    while numEdges != numPredictedEdges:
        edgeList = mainTest(sampleGraph, guess_K)
        delete = edgeList[-1]
        print("Deleting: {0}\n".format(delete))
        u = delete[0]
        v = delete[1]
        sampleGraph[u].remove(v)
        sampleGraph[v].remove(u)
        numEdges -= 1
    for node in sampleGraph.keys():
        print("{0}: {1}".format(node, sampleGraph[node]))
        
#graph1 is from Wikipedia page on Max Clique Problem -- max clique is [4,5,6,7]
#graph1 = {1:[2,3,4,6], 2:[1,3,5,7], 3:[1,2,4,5], 4:[1,3,5,6,7], 5:[2,3,4,6,7], 6:[1,4,5,7], 7: [2,4,5,6]}
#mainTest(graph1, 4)
#metaMainTest(graph1, 4)

#Jesus, each iteration takes about a second...
#main("c125.txt", 34)



















