from operator import itemgetter
import random
import itertools

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
def MIS2(root, neighbors, graph):
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
def reduction2(root, G, mergings):
    lower = 0
    while len(nodeweb(G[root], G)) > 0:
        MISAM = MIS2(root, nodeweb(G[root], G), G)
        if len(MISAM) > 1:
            lower += 1
            mergings.append(MISAM)
        G = merge(MISAM, G)
    return lower
def boundsnorm(root, H, mergings):
    lower = reduction2(root, H, mergings)
    while len(MISroot(root, H)) > 1:
        lower += 1
        MISAM = MISroot(root, H)
        mergings.append(MISAM)
        H = merge(MISAM, H)#doesn't run until completion, just until the root's un-connected neighbors are exhausted
    while len(MISleftovers(H)) > 0:
        lower += 1
        MISAM = MISleftovers(H)
        mergings.append(MISAM)
        H = merge(MISAM, H)
    upper = len(H.keys())
    lower = upper - lower
    #print("Inserted all edges for root {0}:".format(root))
    #print("Range[{0}, {1}]".format(upper, lower))
    return((upper, lower))
########################################################################################################################################################################################
def MCP(filename, x, n = [], othercall = False):
    G = graphConv(filename)
    mergings = []
    if len(n) != 0:
        for item in n:
            G[x].append(item)
            G[item].append(x)
    print("{0}, {1}".format(n, boundsnorm(x, G, mergings)))
    if othercall:
        for item in mergings:
            print(item)
    return mergings
    #if x == 19 and boundsnorm(x,G) == (55,7) and 60 not in n and 113 not in n:
    #    print("Happened")

def allcombos(filename, x):
    G = graphConv(filename)
    neighbors = G[x]
    print(neighbors)
    unneighborsaf = []
    for key in G.keys():
        if key not in neighbors and key != x:
            unneighborsaf.append(key)
    print(unneighborsaf)
    #'''
    unncombos = []
    for i in range(1, len(unneighborsaf) + 1):
        combo = itertools.combinations(unneighborsaf, i)
        for item in combo:
            tuplist = []
            for j in range(len(item)):
                tuplist.append(item[j])
            unncombos.append(tuplist)
    for item in unncombos:
        MCP(filename, x, item)
    #'''

#for x = 19, unneighbors: [20, 60, 61, 83, 106, 113, 124]
#allcombos("c125.txt", 19)
#for x = 19, only when both 60 and 113 are added as edges, do the bounds increase by 1
'''
What does this mean?
Let's assume x = 19 is NOT a part of the MCN: (case 1)
    It means that every time I add 60 and 113 as neighbors,
    60 and 113 would be the remaining members of the MCN that I am not attached to,
    and adding them as neighbors includes me as the MCN, now MCN + 1...
    OR
    Of my updated set of neighbors, I end up merging, additionally, two MIS's that do not include
    members of the MCN, and my overall bounds increase by 1.

And let's assume that x = 19 IS a part of the MCN: (case 2)
    It means that every time I add 60 and 113 as neighbors,
    of my updated set of neighbors, I end up merging together an additional MIS, which does NOT
    include ANY members of the MCN, and my overall bounds increase by 1 (because this MIS of non-MCN
    members are now, after merging, connected to every other member in the graph -- meaning, it is
    connected to all members of the MCN, and is included in the new MCN + 1).
'''
#for x = 113, unneighbors: [8, 18, 19, 20, 26, 30, 38, 42, 43, 51, 56, 64, 68, 75, 83, 91, 117, 123]
#for x = 60, unneighbors: [17, 19, 36, 92, 98]
#allcombos("c125.txt", 60)
#for x = 60, only when I add 92 as a neighbor, do the bounds increase by 1
'''
Let's attempt the following:
For x = 19:
    Print out all the merged MIS's in the regular case,
    Print out all the merged MIS's in the case where both 60 and 113 are added as neighbors,
    Compare the ordering of merged MIS's to tell the difference
'''
#MCP("c125.txt", 19, True)
#MCP("c125.txt", 19, [60, 113], True)
'''
Ok, so... holy shit, first of all,
the ONLY thing that's different between the first iteration and the second is that
in the second, at the very end, it does not have the merging [19,60].

What does this mean?
...It means...
If case 2 were the case, that would mean that I ended up merging an additional two MIS's, neither of
which included fellow members of the MCN, and my overall bounds thereby increased by 1.
HOWEVER, this is contradictory -- merging just one extra MIS, consisting of me and node 60, would
only add 1 to the overall bounds if I was NOT a member of the MCN (as merging an MIS only increases
the bounds if all members of the MIS are not part of the MCN).

This means... 19 is NOT part of the MCN.
Conversely, this means that... what does it imply about 60 or 113? 60 and 113 are already connected...

Hmm, I'm gonna try and see the difference if I just add 60.
'''
#MCP("c125.txt", 19, True)
#MCP("c125.txt", 19, [113], True)
#print(len(MCP("c125.txt", 19)))
#print(len(MCP("c125.txt", 19, [60])))
'''
So... what ends up happening is just the different last merging at the very end:
[19,60] vs. [19,113]
...Yea, and if I just add 113 as an edge, the result is the exact same.
Meaning at the very end, it looks like all nodes are connected to 19, 60, and 113,
but 19 is not connected to 60 or 113, while 60 and 113 are connected to each other.
Now, does that necessarily mean that both or one of 60 and 113 are part of the MCN?

Well, it means that at least one of 19's unneighbors: [20, 60, 61, 83, 106, 113, 124] are
part of the MCN.
'''
def compareOrder(filename, x, n = []):
    list1 = MCP(filename, x)
    list2 = MCP(filename, x, n)
    if len(list1) >= len(list2):
        print("Original is >=; Original Size: {0}".format(len(list1)))
        print("Difference: {0}".format(len(list1) - len(list2)))
        size2 = len(list2)
        flag = True
        for i in range(len(list1)):
            if i == size2:
                for i in range(i, len(list1)):
                    print(list1[i])
                break
            elif list1[i] != list2[i]:
                flag = False
        print(flag)
    else:
        print("Added is >; Added Size: {0}".format(len(list2)))
        print("Difference: {0}".format(len(list2) - len(list1)))
        size1 = len(list1)
        flag = True
        for i in range(len(list2)):
            if i == size1:
                for i in range(i, len(list2)):
                    print(list2[i])
                break
            elif list2[i] != list1[i]:
                flag = False
        print(flag)
#compareOrder("c125.txt", 19, [60, 113])
'''
Alright, let's test node 60, with the addition of 92.
'''
#compareOrder("c125.txt", 60, [92])
'''
Well, shit, the same result occurs as before -- the only difference
is a [60,92] in the first iteration. Which means...
My logic is incorrect. Unless there are multiple MCN's in this graph,
by that logic, it'd mean that 60 is not part of the MCN either.
'''
def unneighbors(filename, x):
    G = graphConv(filename)
    neighbors = G[x]
    unneighborsaf = []
    for key in G.keys():
        if key not in neighbors and key != x:
            unneighborsaf.append(key)
    print(unneighborsaf)
#unneighbors("c125.txt", 92)
#For x = 92, unneighbors: [12, 43, 52, 60, 66, 80, 87, 88, 97, 120]
#Hmm... 113 isn't in this list...
#unneighbors("c125.txt", 8)
#For x = 8, unneighbors: [12, 40, 44, 58, 110, 113, 121]
#Let's check allcombos for this one:
#allcombos("c125.txt", 8)
'''
For x = 8, only when [40, 58, 110, 113] are added do bounds increase by 1...
Also, 8 doesn't share any of its unneighbors except 113 with 19...
Can this conclusively mean anything?
'''
'''
...Ah, screw it. I'm just gonna check all possible combos of 34 in the graph,
and see if any of them are cliques.
'''
def bruteforce(filename, r):
    G = graphConv(filename)
    size = len(G.keys())
    allnodes = []
    for i in range(1, size + 1):
        allnodes.append(i)
    combo = itertools.combinations(allnodes, r)
    for item in combo:
        #print(item)
        flag = True
        for i in range(len(item)-1):
            friends = G[item[i]]
            innerflag = False
            for j in range(i + 1, len(item)):
                if item[j] not in friends:
                    innerflag = True
                    flag = False
                    break
            if innerflag:
                break
        if flag:
            print(item)
            break#
#bruteforce("c125.txt", 5)
'''
...Ok shit, maybe not. C(125, 34) = 4.7 x 10^30.
Shit, even C(125, 13) = 1.5 x 10^17
And even C(125, 8) = 1.17 x 10^12
C(125, 5) = 2.34*10^8
...Goddamnit.
'''

'''
"7 9 11 13 19 22 25 29 33 34 40 44 49 52 54 55 66 67 68 70 79 80 93 96 98 99 103 104 110 111 114 117 122 125"
OK, so, now that I know at least ONE clique solution for c125.txt,
it'll help speed up the debugging process to find out if my code is wrong
or not... right?
Crap, no it won't... there are probably multiple 34-cliques. DAMNIT.
'''
clique1sol = [7, 9, 11, 13, 19, 22, 25, 29, 33, 34, 40, 44, 49, 52, 54, 55, 66, 67, 68, 70, 79, 80, 93, 96, 98, 99, 103, 104, 110, 111, 114, 117, 122, 125]
def genMIS(mcn, graph):
    miset = [mcn]
    unneighbors = []
    friends = graph[mcn]
    for key in graph.keys():
        if key != mcn and key not in friends:
            unneighbors.append(key)
    
    while len(unneighbors) > 0:
        first = unneighbors.pop(0)
        miset.append(first)
        adjacency = graph[first]
        for friend in adjacency:
            if friend in unneighbors:
                unneighbors.remove(friend)
    return miset
def isComplete(G):
    keys = G.keys()
    size = len(keys) - 1
    complete = True
    for i in range(len(keys)):
        node = keys[i]
        if len(G[node]) != size:
            complete = False
            break
    return complete
def genRedux(filename, MCN):
    G = graphConv(filename)
    i = 0
    mergings = []
    numreduced = 0
    print("MC#: {0}".format(len(MCN)))
    while not isComplete(G) and i < len(MCN):
        MISAM = genMIS(MCN[i], G)
        numreduced += len(MISAM)
        print("Merged: {0}, Cumulative Size: {1}".format(MISAM, numreduced))
        if len(MISAM) > 1:
            mergings.append(MISAM)
        G = merge(MISAM, G)
        i += 1
    leftovers = []
    for key in G.keys():
        if key not in MCN:
            leftovers.append(key)
    print(leftovers)
    print(len(leftovers))
    print(len(G.keys()))
genRedux("c125.txt", clique1sol)
'''
OHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH....
You want to MAXIMIZE the size(MIS) w/ each MCN-member when merging...
    Because otherwise, you could end up with all MCN-members connected to all
    other nodes in the graph, but leftover nodes not part of the MCN that are
    also NOT connected to each other...
So, theoretically, if there exists a MCN of size k in a graph, then there exists
    exactly k MIS's of varying size that the entire graph can be merged sequentially
    into. But... finding exactly such k MIS's, where the cumulative merging of all
    included nodes in the k MIS's == size(G).
Also interestingly enough -- just knowing a unique max clique in a graph doesn't
    necessarily mean that the graph's MIS lies with one of the members of that
    MCN-solution... there could be multiple MCN-solutions, and the MIS lies with
    any one member of all total.
The difficulty of even a graph of size 125 is that you can't really confirm whether
    or not there's only one MCN-solution to that graph.
'''
'''
Hmm... this is just for approx. purposes, but...
All_Errs = []
Take the average "upper bound" computed for all nodes in the graph, after running
    my merge algorithm to completion using each node as a "seed".
    - Call it "avg_upps"
Use my algorithm to merge the first resulting MIS found for each individual node
    in the graph. Note the size of each MIS found, and take the average MIS-size
    over all nodes.
    - Call it "avg_mis"
For numbers 1 - lowest upper bound, let the number be called "test_k":
    For that number,
        Utilize my algorithm to continue merging an arbitrary set of size(test_k), given
        a particular seed as the "root" node.
        Note the size of the "leftovers" in the graph after attempting this merge.
        Do so for all nodes in the graph, and take the average over all attempts of the
        leftovers-size, call it "left_size".
    Expected_Extras = left_size/avg_mis
    Expected_Actual = avg_upps - Expected_Extras
    Rel_Err = abs(Expected_Actual - test_k)/test_k
    All_Errs.append((test_k, Rel_Err))
All_Errs = sorted(All_Errs, key=itemgetter(1), reverse=True)
print(All_Errs[0])
'''



















