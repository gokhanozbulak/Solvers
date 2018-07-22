'''
Created on Nov 22, 2017

@author: go

This is a solver to solve Graph Coloring Problem (GCP)

'''

import sys
from random import shuffle
from random import randint

graph = {}

def read_problem(file_name):
    # read the file to get information about the problem
    lines = open(file_name).readlines()
     
    # get the number of items (n) and capacity of knapsack (K)
    tokens = lines[0].split()
    n_nodes = int(tokens[0])
    n_edges = int(tokens[1])
     
    # get the value (v) and weight (w) of each item
    for line in lines[1:n_edges+1]:
        tokens = line.split()
        node_i = int(tokens[0])
        node_j = int(tokens[1])
        
        if node_i in graph:
            graph[node_i].append(node_j)
        else:
            graph[node_i] = [node_j]
        
        if node_j in graph:
            graph[node_j].append(node_i)
        else:
            graph[node_j] = [node_i]
    
    return n_nodes

def is_color_ok(neighbors, coloring, next_avail_color):
    for n in neighbors:
        if coloring[n] == next_avail_color:
            return False    
    return True

# simple greedy approach
def solve_greedy(n_nodes):
    obj = -1
    coloring = [-1] * n_nodes
    
    next_avail_color = 0
    for node,neighbors in graph.items():
        while not is_color_ok(neighbors, coloring, next_avail_color):
            next_avail_color += 1
        coloring[node] = next_avail_color
    
    obj = next_avail_color + 1
    
    return (coloring, obj)

# simple iterated greedy approach
def solve_iterated_greedy(n_nodes, iter=10000):
    min_obj = sys.maxsize
    min_coloring = []
    
    for i in range(iter):
        obj = -1
        coloring = [-1] * n_nodes
        
        nodes = list(graph.keys())
        shuffle(nodes)
        
        next_avail_color = 0
        for node in nodes:
            assigned_exist_color = False
            for curr_color in range(next_avail_color+1):
                if is_color_ok(graph[node], coloring, curr_color):
                    assigned_exist_color = True
                    color = curr_color
            if not assigned_exist_color:
                next_avail_color += 1
                color = next_avail_color
            coloring[node] = color
        
        obj = next_avail_color + 1
        
        if obj < min_obj:
            min_obj = obj
            min_coloring = coloring
    
    return (min_coloring, min_obj)

# DSATUR approach
# DSATUR rule
def get_next_node_for_only_dsatur(dsatur, degree):
    # get node(s) that have maximum degree of saturation
    max_satur_deg = [i for i,x in enumerate(dsatur) if x == max(dsatur)]
    if len(max_satur_deg) == 1:
        # return node that has max degree of saturation
        return max_satur_deg[0]
    elif len(max_satur_deg) > 1:
        # tie-break: get node that has max degree
        max_deg = [i for i,x in enumerate(degree) if x == max(degree)]
        # whether it's tie-break or not, return 1st node in max_deg
        return max_deg[randint(0,len(max_deg)-1)]
    
# DSATUR with SEWELL rule
def apply_sewell(max_dsaturs, degree, coloring):
    max_deg = -1
    max_node = -1
    
    max_avail_color = max(coloring)
    
    if max_avail_color == -1:
        # tie-break: get node that has max degree
        max_deg = [i for i,x in enumerate(degree) if x == max(degree)]
        # whether it's tie-break or not, return 1st node in max_deg
        return max_deg[randint(0,len(max_deg)-1)]
    
    # select node with maximum number of common available colors
    # in neighborhood of uncolored nodes
    for i in max_dsaturs:
        # get uncolored neighbors of current node with max dsatur
        uncolored_neighbors = [j for j in graph[i] if coloring[j] == -1]
        for avail_color in range(max_avail_color+1):
            deg = 0
            for n in uncolored_neighbors:
                colored = [k for k in graph[n] if coloring[k] == avail_color]
                if len(colored) == 0:
                    deg += 1
        if deg > max_deg:
            max_deg = deg
            max_node = i
    return max_node
        

def get_next_node_with_sewell(dsatur, degree, coloring):
    # get node(s) that have maximum degree of saturation
    max_satur_deg = [i for i,x in enumerate(dsatur) if x == max(dsatur)]
    if len(max_satur_deg) == 1:
        # return node that has max degree of saturation
        return max_satur_deg[0]
    elif len(max_satur_deg) > 1:
        # tie-break: apply SEWELL rule:
        return apply_sewell(max_satur_deg, degree, coloring)

def colorless_node_exist(coloring):
    if -1 in coloring:
        return True
    return False

def compute_node_degrees(nodes):
    degree = [-1] * (max(nodes)+1)
    for node in nodes:
        degree[node] = len(graph[node])
    return degree

def compute_dsatur(nodes, coloring):
    dsatur = [-1] * (max(nodes)+1)
    for node in nodes:
        # get neighbor colors of node
        color = [coloring[j] for j in graph[node]]
        # compute degree of saturation by counting different number of colors in neighborhood
        dsatur[node] = len(set(color))
            
    return dsatur

def solve(n_nodes, iter=1):
    min_obj = sys.maxsize
    min_coloring = []
    
    for i in range(iter):
        obj = -1
        coloring = [-1] * n_nodes
        
        nodes = list(graph.keys())
        nodes_updated = list(nodes)        
        
        next_avail_color = 0
        while colorless_node_exist(coloring):
            degree = compute_node_degrees(nodes_updated)
            dsatur = compute_dsatur(nodes_updated, coloring)
            node = get_next_node_with_sewell(dsatur, degree, coloring)
            assigned_exist_color = False
            for curr_color in range(next_avail_color+1):
                if is_color_ok(graph[node], coloring, curr_color):
                    assigned_exist_color = True
                    color = curr_color
            if not assigned_exist_color:
                next_avail_color += 1
                color = next_avail_color
            coloring[node] = color
            nodes_updated.remove(node)
        
        obj = next_avail_color + 1
        
        if obj < min_obj:
            min_obj = obj
            min_coloring = coloring
    
    return (min_coloring, min_obj)

def main():
    if len(sys.argv) != 2:
        print("Usage: %s input_file" % sys.argv[0])
        sys.exit()
        
    file_name = sys.argv[1]

    n_nodes = read_problem(file_name)
    
    coloring, obj = solve(n_nodes)
    
    print(obj)
    print(coloring[0], end='')
    for t in coloring[1:]:
        print(' %d' % t, end='')
    print()

if __name__ == '__main__':
    main()