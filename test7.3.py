from operator import itemgetter
import random
########################################################################################################################################################################################
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
########################################################################################################################################################################################
def MISroot(root, graph):
    miset = [root]
    disconnected = []
    left = []
    adjroot = graph[root]
    for key in graph.keys():
        if key not in adjroot:
            if key != root:
                disconnected.append(key)
                left.append(key)
    for i in range(len(disconnected)):
        add = disconnected[i]
        if add in left:
            miset.append(add)
            adjadd = graph[add]
            for mutual in adjadd:
                if mutual in left:
                    left.remove(mutual)
    return miset
########################################################################################################################################################################################
def MISleftovers(graph):
    miset = []
    leftovers = []
    size = len(graph.keys()) - 1
    for key in graph.keys():
        if len(graph[key]) < size:
            leftovers.append(key)
    while len(leftovers) > 0:
        first = leftovers.pop(0)
        miset.append(first)
        adjacency = graph[first]
        for friend in adjacency:
            if friend in leftovers:
                leftovers.remove(friend)
    return miset
########################################################################################################################################################################################
def MIS(root, neighbors, graph):
    miset = []
        
    allnodes = []
    for key in graph.keys():
        allnodes.append(key)
    allnodes.remove(root)
    
    for i in range(len(neighbors)):
        first = neighbors[i]
        if first in allnodes:
            miset.append(first)
            adjacency = graph[first]
            allnodes.remove(first)
            for friend in adjacency:
                if friend in allnodes:
                    allnodes.remove(friend)
                        
    while len(allnodes) > 0:
        first = allnodes.pop(0)
        miset.append(first)
        adjacency = graph[first]
        for friend in adjacency:
            if friend in allnodes:
                allnodes.remove(friend)
    return miset
########################################################################################################################################################################################
def merge(bunch, graph):
    root = bunch[0]
    for i in range(1, len(bunch)):
        tomerge = bunch[i]
        for friend in graph[tomerge]:
            graph[friend].remove(tomerge)
            if friend not in graph[root]:
                graph[root].append(friend)
                graph[friend].append(root)
        graph.pop(tomerge, None)
    return graph
########################################################################################################################################################################################
def nodeweb(neighbors, graph):
    temp = []
    size = len(graph.keys()) - 1
    for n in neighbors:
        temp.append(n)
    i = 0
    while i < len(temp):
        if len(graph[temp[i]]) == size:
            temp.pop(i)
        else:
            i += 1
    return temp
########################################################################################################################################################################################       
def reduction(root, G):
    lower = 0
    while len(nodeweb(G[root], G)) > 0:
        MISAM = MIS(root, nodeweb(G[root], G), G)
        if len(MISAM) > 1:
            lower += 1
        G = merge(MISAM, G)
    return lower
def boundsreduction(G):
    keys = G.keys()
    root = keys[0]
    lower = reduction(root, G)
    while len(MISroot(root, G)) > 1:
        lower += 1
        G = merge(MISroot(root, G), G)
    while len(MISleftovers(G)) > 0:
        lower += 1
        G = merge(MISleftovers(G), G)
    upper = len(G.keys())
    lower = upper - lower
    print("Range[{0}, {1}]".format(upper, lower))
    return(G)
########################################################################################################################################################################################       
def reduction2(root, G, merged):
    while len(nodeweb(G[root], G)) > 0:
        MISAM = MIS(root, nodeweb(G[root], G), G)
        if len(MISAM) > 1:
            for i in range(1, len(MISAM)):
                merged[MISAM[0]].append(MISAM[i])
        G = merge(MISAM, G)
    return G
def MCP(G, root):
    merged = {}
    for key in G.keys():
        merged[key] = []
    
    reduction2(root, G, merged)
    unneighbors = []
    for key in G.keys():
        if key not in G[root]:
            if key != root:
                unneighbors.append(key)
    #unneighborclique = (1, True)
    unneighborclique = True
    for i in range(len(unneighbors) - 1):
        first = unneighbors[i]
        for j in range(i + 1, len(unneighbors)):
            second = unneighbors[j]
            if second not in G[first]:
                #unneighborclique = (1, False)
                unneighborclique = False
    #if len(unneighbors) == 0:
        #unneighborclique = (0, False)
    return unneighborclique
########################################################################################################################################################################################
def innercycle(graph):
    redundancy = {}
    for key in graph.keys():
        H = {}
        for k in graph.keys():
            temp = []
            for item in graph[k]:
                temp.append(item)
            H[k] = temp
        truth = MCP(H, key)
        redundancy[key] = truth
    remove = []
    for key in redundancy:
        if redundancy[key]:
            remove.append(key)
    return remove
######################################################################################################################################################################################## 
def outercycle(filename):
    remove = []
    G = graphConv(filename)
    prev = len(remove)
    flag = True
    while flag:
        H = {}
        for k in G.keys():
            temp = []
            for item in G[k]:
                temp.append(item)
            H[k] = temp
        for item in remove:
            friends = H[item]
            for friend in friends:
                H[friend].remove(item)
            H.pop(item, None)
        remove += innercycle(H)
        if len(remove) != prev:
            prev = len(remove)
        else:
            flag = False
    '''
    i = 0
    while i < 2:
        H = {}
        for k in G.keys():
            temp = []
            for item in G[k]:
                temp.append(item)
            H[k] = temp
        for item in remove:
            friends = H[item]
            for friend in friends:
                H[friend].remove(item)
            H.pop(item, None)
        remove += innercycle(H)
        i += 1
    '''
    print(len(remove))
    print(remove)
    newgraph = graphConv(filename)
    for item in remove:
        friends = newgraph[item]
        for friend in friends:
            newgraph[friend].remove(item)
        newgraph.pop(item, None)
    newgraph = boundsreduction(newgraph)
    return newgraph
########################################################################################################################################################################################    

#outercycle("c125.txt")
#outercycle("c250.txt")

########################################################################################################################################################################################    
'''
I think the reason this doesn't work is because, say, if I'm not part of the k max-clique, then in a graph with high edge density, I'm most likely connected to like... at most, k-2
members of the max clique. Therefore, when I do the neighbor-reduction, clearly a number greater than k members are my neighbor, and thus merging maximal independent sets with these neighbors
oftentimes will...
No, let's say I AM part of the k-max clique. When I do the neighbor reduction, there are three possibilities of what will occur: either there are no more un-neighbors left, there are
un-neighbors left that are connected to each other, and there are un-neighbors left that all exist in a maximal independent set with me. In the first case, it's definitive that
I am part of the k-max clique (because there then exists exactly one MIS left with me in it, and taking me out of the graph would change the bounds produced by exactly one). However,
in the other two cases... I think it's inconclusive.

...

The whole point of merging maximal independent sets is that, by chance... if the max-clique is of size k, and there is only one of them, then that means that there exists at most k maximally
independent sets (overlapped and not mutually exclusive, but each distinct by at least two differing nodes) that can be merged in succession that reduces the entire graph to the max-clique's
members. This is because each member of the max clique exists in different maximal independent sets, and because there is no greater clique in the graph, then all cliques must fall in at least
one of the k different maximal independent sets (if not several within the k). Thus, if you manage to find large MIS's within the graph, chances are, you'll have found one member within the
MCN in the MIS, and merging that member with its entire MIS reduces the graph's complexity at no change (or cost) to the size of the MCN.
However, what occurs when you merge an MIS that does not contain a member of the MCN? You effectively create a node attached to all members of the graph -- and therefore, the MCN -- and add
to the MCN +1.
'''





