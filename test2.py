from random import shuffle

def graphConv(fileName):
    graph = [None]*2
    #edges, nodes in graph w/ respective neighbors
    graph[0] = []       #all edges in 2-slot-array-format
    graph[1] = {}       #nodes in graph, w/ keys of node-names leading to 2-slot-arrays of (# neighbors, arrays of their neighbors)
    flag = False
    with open(fileName,'r') as file:
        for line in file:
            if flag == False:
                flag = True
                continue
            else:
                edge = line.split()
                e1 = int(edge[1])
                e2 = int(edge[2])
                graph[0].append([e1,e2])
                if e1 not in graph[1]:
                    init = [e2]
                    graph[1][e1] = [0,init]
                else:
                    graph[1][e1][1].append(e2)
                if e2 not in graph[1]:
                    init = [e1]
                    graph[1][e2] = [0,init]
                else:
                    graph[1][e2][1].append(e1)
    for node in graph[1]:
        graph[1][node][0] = len(graph[1][node][1])
    #print(graph[0][17800:])
    #print(graph[1][328])
    return graph

def test():
    bench = graphConv("frb30-15-1.txt")
                
test()



















































