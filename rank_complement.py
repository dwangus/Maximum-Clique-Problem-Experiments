import numpy as np

def graphConv(filename):
    size = 0
    with open(filename, 'r') as file:
        for line in file:
            contents = line.split()
            if contents[0] == 'p':
                size = int(contents[2])
                break
    adjmat = [[0 for x in range(size)] for x in range(size)]
    with open(filename, 'r') as file:
        for line in file:
            edge = line.split()
            if edge[0] == 'e':
                e1 = int(edge[1]) - 1
                e2 = int(edge[2]) - 1
                adjmat[e1][e2] = 1
                adjmat[e2][e1] = 1
    return adjmat

def complement(filename):
    G = graphConv(filename)
    for i in range(len(G[0])):
        for j in range(len(G[0])):
            if i != j:
                if G[i][j] == 0:
                    G[i][j] = 1
                else:
                    G[i][j] = 0
    return G

def rank(G):
    return np.linalg.matrix_rank(np.matrix(G))

#print(rank(complement("c125.txt")))
#print(rank(complement("c250.txt")))
#print(rank(complement("c500.txt")))
