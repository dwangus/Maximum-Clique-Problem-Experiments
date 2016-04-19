import math
import time
from operator import itemgetter
import threading

#STILL TOO SLOW!!!

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

def hasConverged(distances):
    for key in distances.keys():
        edgeDict = distances[key]
        for edge in edgeDict.keys():
            if edgeDict[edge][1] == False:
                return False
    return True

def nextDirection(node, neighbors, coords):
    position = coords[node]
    neighborsCoords = []
    for neighbor in neighbors:
        neighborsCoords.append(coords[neighbor])
    vecList = []
    for neighbor in neighborsCoords:
        vecList.append(unitVector(vectorFormation(position, neighbor)))
    return unitVector(vectorAddition(vecList))

def updatePosition(coords, node, toMoveNext):
    for dim in range(len(toMoveNext)):
        coords[node][dim] += toMoveNext[dim]

def updateDistances(coords, distances):
    for node in distances.keys():
        relativesDict = distances[node]
        position = coords[node]
        for relative in relativesDict.keys():
            spacing = relativesDict[relative]
            dist = distance(position, coords[relative])
            spacing[0] = dist
            if dist < 2.0:#Arbitrary value, based on step-distance = 1.0 for unit vector
                spacing[1] = True
            #Can spacing ever grow to be > 2.0 after it's converged?

def avgPosition(coords, nodes):
    average = [0.0]*len(coords[nodes[0]])
    for node in nodes:
        position = coords[node]
        for dim in range(len(average)):
            average[dim] += position[dim]
    numNodes = float(len(nodes))
    for dim in range(len(average)):
        average[dim] /= numNodes
    return average

def getNextMove(node_id, adjList, coords, movesDict):
    movesDict[node_id] = nextDirection(node_id, adjList[node_id], coords)

def convergence(coordinates, adjacency, space, onResume = False, resumedCoords = {}):
    start = time.time()
    nodes = adjacency.keys()
    distances = {}
    if onResume:
        coords = resumedCoords
        for i in range(len(nodes) - 1):
            lesser = nodes[i]
            distances[lesser] = {}
            for j in range(i+1, len(nodes)):
                greater = nodes[j]
                distances[lesser][greater] = [distance(resumedCoords[lesser], resumedCoords[nodes[j]]), False]
                if distances[lesser][greater][0] < 2.0:
                    distances[lesser][greater][1] = True
                    
    else:
        coords = coordinates
        for i in range(len(nodes)-1):
            lesser = nodes[i]
            distances[lesser] = {}
            for j in range(i+1, len(nodes)):
                distances[lesser][nodes[j]] = [space, False]

    iter = 0
    while not hasConverged(distances):
        iter += 1
        print("Round {0}".format(iter))

        #STILL TOO SLOW!!!
        moveUpdates = {}
        nodeThreads = []
        for node in nodes:
            newThread = threading.Thread(target=getNextMove, args = (node, adjacency, coords, moveUpdates))
            newThread.start()
            nodeThreads.append(newThread)
        for process in nodeThreads:
            process.join()

        for node in nodes:
            updatePosition(coords, node, moveUpdates[node])
        '''
        nodeThreads2 = []
        for node in nodes:
            newThread = threading.Thread(target = updatePosition, args = (coords, node, moveUpdates[node]))
            newThread.start()
            nodeThreads.append(newThread)
        for process in nodeThreads2:
            process.join()
        '''
        
        updateDistances(coords, distances)
        if (iter%90 == 0):
            for key in coords.keys():
                print("{0}: {1},".format(key, coords[key]))

    print("Time to Converge: {0}".format(time.time() - start))
    return avgPosition(coords, nodes)

def main(filename, origSize, sample = False, sampleGraph = {}, onResume = False, resumeCoords = {}):
    graph = {}
    if sample:
        graph = sampleGraph
    else:
        graph = graphConv(filename)

    toDel = []
    for node in toDel:
        neighbors = graph[node]
        for neighbor in neighbors:
            graph[neighbor].remove(node)
        graph.pop(node, None)
    
    if True:
        #Typically, spacing is 5000
        spacing = 5000
        coords = equidistant_vectors(origSize, spacing)
        coordsCopy = {}
        for key in coords.keys():
            position = []
            for dimension in coords[key]:
                position.append(dimension)
            coordsCopy[key] = position
        
        cPoint = convergence(coordsCopy, graph, spacing, onResume, resumeCoords)
        print(cPoint)

        distResults = []
        for node in graph.keys():
            distResults.append((node, distance(cPoint, coords[node])))
        distResults = sorted(distResults, key=itemgetter(1), reverse=True)

        parsed = []
        print(distResults)
        for item in distResults:
            parsed.append(item[0])
        print(parsed)
        return distResults[0][0]

#1 Iteration took 0.6 sec
#Size 4 took 3868 simulated rounds
graph1 = {1:[2,3,4], 2:[1,3], 3:[1,2], 4:[1]}
#1 Iteration took 1.74 sec
#Size 7 took 3440 simulated rounds
graph2 = {1:[2,3,4,6], 2:[1,3,5,7], 3:[1,2,4,5], 4:[1,3,5,6,7], 5:[2,3,4,6,7], 6:[1,4,5,7], 7: [2,4,5,6]}
#1 Iteration took 1.3 sec
graph2ver2 = {1:[2,4,6], 2:[1,5,7], 4:[1,5,6,7], 5:[2,4,6,7], 6:[1,4,5,7], 7: [2,4,5,6]}
#1 Iteration took 1.0 sec
graph2ver3 = {1:[4,6], 4:[1,5,6,7], 5:[4,6,7], 6:[1,4,5,7], 7: [4,5,6]}
#1 Iteration took 0.6 sec
graph2ver4 = {4:[5,6,7], 5:[4,6,7], 6:[4,5,7], 7: [4,5,6]}

graph = graph2
#main("c125.txt", 7, True, graph)

#'''
#1 Iteration took 10,385 sec
#Size 125 took 3526 simulated rounds
#Note to self: Converged at... [2500.1606507561632, 1442.297368316359, 1020.9035173351026, 790.785856318066, 645.8978943025529, 546.780878827789, 473.54237984520347, 416.6814071002168, 372.9346475198153, 337.7651684992841, 306.51289554920527, 282.87369095514794, 261.1904799309855, 242.45481892597658, 226.79674719720086, 213.50343814068427, 202.9363278769302, 192.44613404664068, 181.14223321611146, 171.84367302585264, 165.0118714827357, 156.9617862348561, 151.22024426072954, 144.57052309055285, 138.43254493037603, 133.21253740075556, 128.12656293375687, 125.03317070810411, 120.10125643266791, 115.92598780626956, 111.77610144562067, 107.85094011460671, 106.06061849071692, 102.50518518682027, 97.20686682844537, 96.72556952458811, 93.6688605552854, 92.66237650205935, 90.5313997528022, 87.6097190125494, 83.81641237507291, 83.04803859240326, 81.37941778421214, 80.97639167809864, 78.00210626524475, 76.79105055578769, 74.70182672337468, 73.8700940266351, 70.47985909838101, 68.35085557828651, 68.9494522965366, 67.15506063883844, 67.81642204926685, 64.1803035053063, 63.263473610474975, 62.62974105755576, 61.54690200492055, 60.97076611110323, 61.1350066893735, 57.97261298095041, 57.259912915578475, 55.87062323095282, 54.27455439531492, 55.34835220024454, 53.772732842528974, 54.144291137386176, 51.439188907041995, 51.65807840705602, 51.61842438628762, 49.69983994825094, 49.496681891558346, 47.1304434520029, 48.40525408914543, 45.58119223959545, 44.986307490534685, 46.79760723835413, 45.008888162710534, 45.13633276078471, 45.78243361653749, 43.05593024706185, 43.95763112615007, 40.53187058436222, 42.22093094721457, 42.196137780573785, 41.70535544361472, 40.5149772615826, 39.070176512929145, 39.36026790353056, 37.95235473044288, 39.23086163283957, 39.27864289014676, 37.88714180151359, 36.53653005618709, 36.14553675120005, 37.433319615377776, 35.3705773023133, 37.40330528309892, 37.284444024326774, 34.748258868911506, 36.565506646976985, 34.282233062240955, 33.68497693171059, 35.77969430901654, 33.27500676022936, 33.43224778109444, 33.124239700056954, 30.86281342774412, 32.773403380319564, 33.67990482506363, 33.61056635902326, 31.154523222616273, 30.137018117798718, 32.99954590664156, 31.02700405829687, 29.795995788605772, 30.500453773282228, 30.247979408465053, 30.226916319766907, 29.26186779081025, 29.263168756770042, 29.001778269039683, 29.49490611499247, 28.307327970685638, 29.48597598692083]
DIMACS1 = "c125.txt"
#1 Iteration took ___ sec
DIMACS2 = "c250.txt"

filename = DIMACS1
main(filename, 125)
#'''


'''
Round 3440
Time to Converge: 16.2550001144
[2499.9999999999804, 1437.3417006672228, 1087.4054124579886, 842.300610605118, 622.8527022591009, 526.4066113834788]
[(3, 3305.325574947178), (2, 3297.411321531699), (1, 3297.4113215316693), (6, 3293.573415453234), (7, 3293.5734154531806), (5, 3216.371452558667), (4, 3216.3714525586643)]
[3, 2, 1, 6, 7, 5, 4]
'''

'''
Round 3440
Time to Converge: 9.27600002289
[2499.9999999999804, 1437.3417006672228, 1087.4054124579886, 842.300610605118, 622.8527022591009, 526.4066113834788]
[(3, 3305.325574947178), (2, 3297.411321531699), (1, 3297.4113215316693), (6, 3293.573415453234), (7, 3293.5734154531806), (5, 3216.371452558667), (4, 3216.3714525586643)]
[3, 2, 1, 6, 7, 5, 4]
'''
































