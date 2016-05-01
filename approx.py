from operator import itemgetter
import math

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

def closedNeighborhood(node, graph):
    antiNeigh = []
    for key in graph.keys():
        if key not in graph[node]:
            antiNeigh.append(key)
    for anti in antiNeigh:
        neighbors = graph[anti]
        for n in neighbors:
            graph[n].remove(anti)
        graph.pop(anti, None)
    return graph
def kLog(graph):
    initDense = 0.898452
    densities = []
    for key in graph.keys():
        temp = makeGraphCopy(graph)
        closed  = closedNeighborhood(key, temp)
        avg = 0.0
        size = float(len(closed.keys()))
        for node in temp.keys():
            avg += float(len(temp[node]))/size
        avg /= size
        log = 1 + int(math.log((1.0/(size)), avg))
        densities.append((key,log, size))
    densities = sorted(densities, key=itemgetter(1))
    return densities
def nonNeigh(originNode, graph):
    unneighbors = []
    neighborhood = graph[originNode]
    for key in graph.keys():
        if key != originNode:
            if key not in neighborhood:
                unneighbors.append(key)
    return unneighbors
def minDegree(graph):
    min = 125
    for key in graph.keys():
        if len(graph[key]) < min:
            min = len(graph[key])
    print(min)
    return min
def maxDegree(graph):
    max = 0
    for key in graph.keys():
        if len(graph[key]) > max:
            max = len(graph[key])
    print(max)
    return max
def sumDegree(graph):
    sum = 0.0
    size = len(graph.keys())
    for key in graph.keys():
        sum += len(graph[key])
    sum = sum/2
    return sum/(size*(size-1)/2)
def avgDegree(graph):
    sumAvg = 0.0
    for key in graph.keys():
        sumAvg += len(graph[key])/125.0
    sumAvg /= 125.0
    print(sumAvg)
def medianDeg(graph):
    degrees = []
    '''for key in graph.keys():
        degrees.append(len(graph[key])/float(len(graph.keys())))
    degrees = sorted(degrees)'''
    #print((degrees[62] + degrees[61])/2.0)
    for key in graph.keys():
        degrees.append((key, len(graph[key])/float(len(graph.keys()))))
    degrees = sorted(degrees, key=itemgetter(1))
    for d in degrees:
        print(d)
graph = graphConv("c125.txt")
medianDeg(graph)
#sumDegree(graph)
#closed = closedNeighborhood(36, graph)
#print(kLog(closed))
#print(kLog(graph))
'''
def initEdgeDensities(graph):
    densities = {}
    for key in graph.keys():
        temp = makeGraphCopy(graph)
        closed = closedNeighborhood(key, temp)
        avg = 0.0
        size = float(len(closed.keys()))
        for node in temp.keys():
            avg += float(len(temp[node]))/size
        avg /= size
        densities[key] = avg
    return densities

def metaED(graph, iterations):
    densities = initEdgeDensities(graph)
    antiNeighbors = {}
    for key in graph.keys():
        antiNeighbors[key] = nonNeigh(key, graph)
    for i in range(iterations):
        for key in graph.keys():
            antiN = antiNeighbors[key]
            avg = densities[key]
            for n in antiN:
                avg += densities[n]
            avg /= float(len(antiN)+1)
            densities[key] = avg
    return densities

def convDictToListAscending(dic):
    lis = []
    for key in dic.keys():
        lis.append((key, dic[key]))
    return sorted(lis, key=itemgetter(1))

def convDictToListDescending(dic):
    lis = []
    for key in dic.keys():
        lis.append((key, dic[key]))
    return sorted(lis, key=itemgetter(1), reverse=True)
'''  

#dense = metaED(graph, 100)
#asc = convDictToListAscending(dense)
#for item in asc:
#    print(item)
#print("\n")
#desc = convDictToListAscending(dense)
#for item in desc:
#    print(item)



#[36, 83, 108, 76, 51, 90, 64, 15, 95, 68, 42, 97, 88, 16, 94, 75, 102, 55, 27, 33, 43, 112, 87, 56, 100, 21, 73, 121, 107, 105, 3, 14]
#[113, 61, 109, 50, 124, 84, 72, 32, 37, 4, 57, 78, 20, 116, 28, 12, 53, 120, 106, 65, 98, 46, 115, 23, 62, 30, 118, 74, 52, 39, 111, 6]
#[58, 103, 91, 89, 38, 86, 63, 81, 119, 10, 49, 92, 26]
    
