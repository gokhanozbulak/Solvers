'''
Created on Nov 2, 2017

@author: go

This is a solver to solve Single Knapsack Problem (KSP)

'''

import sys
import random

def read_problem(file_name):
    # read the file to get information about the problem
    lines = open(file_name).readlines()
     
    # get the number of items (n) and capacity of knapsack (K)
    tokens = lines[0].split()
    n_items = int(tokens[0])
    capacity = int(tokens[1])
     
    # get the value (v) and weight (w) of each item
    items = []
    for line in lines[1:n_items+1]:
        tokens = line.split()
        value = int(tokens[0])
        weight = int(tokens[1])
        item = (value, weight)
        items.append(item)

    return (capacity, items)

# simple greedy approach
def solve_greedy(capacity, items, p=0.5):
    taken = []
    obj = 0
    total_weight = 0
    
    for item in items:
        value = item[0]
        weight = item[1]
         
        if random.random() < p and total_weight + weight <= capacity:
            obj += value
            total_weight += weight
            taken.append(1)
        else:
            taken.append(0)
    
    return (taken, obj)

# Uninformed Random Walk (URW)
def solve_urw(capacity, items, p=0.2, N=10000):
    taken = []
    max_taken = []
    max_obj = -1
    
    taken = []
    for k in range(len(items)):
        if random.random() < p:
            taken.append(1)
        else:
            taken.append(0)
    
    for k in range(N):
        i = random.randint(0, len(items)-1)
        if taken[i] == 1:
            taken[i] = 0
        else:
            taken[i] = 1
        
        o, w = satisfy(capacity, items, taken)
        if w <= capacity and o > max_obj:
            max_taken = list(taken)
            max_obj = o
    
    return (max_taken, max_obj)

def satisfy(capacity, items, taken):
    obj = 0
    total_weight = 0
    
    for i,item in enumerate(taken):
        if item == 1:
            obj += items[i][0]
            total_weight += items[i][1]
            
    return (obj, total_weight)

# dynamic programming, not optimized
def solve_dp_notoptimized1(capacity, items):
    obj = 0
    total_weight = 0
    
    n_items = len(items)
    
    taken = [0] * n_items 
    
    values = []
    weights = []
    for v,w in items:
        values.append(v)
        weights.append(w)
    
    M = [i[:] for i in [[0]*(capacity+1)]*(n_items+1)]
    
    for i in range(n_items+1):
        for c in range(capacity+1):
            if i == 0 or c == 0:
                M[i][c] = 0
            elif weights[i-1] > c:
                M[i][c] = M[i-1][c]
            else:
                M[i][c] = max(M[i-1][c],
                              M[i-1][c-weights[i-1]] + values[i-1])

    print('burada')
    i = n_items
    c = capacity
    while i>0 and c>0:
        if M[i][c] != M[i-1][c]:
            taken[i-1] = 1
            c -= weights[i-1]            
        i -= 1
    
    obj = M[n_items][capacity]
    
    return (taken, obj)

# dynamic programming, not optimized
def solve_dp_notoptimized2(capacity, items):
    obj = 0
    total_weight = 0
    
    n_items = len(items)
    
    taken = [0] * n_items 
    
    values = []
    weights = []
    for v,w in items:
        values.append(v)
        weights.append(w)
    
    M = [i[:] for i in [[0]*(capacity+1)]*(2)]
    
    keep = []
    keep.append([])
    last = [0] * (capacity+1)
    before_last = [0] * (capacity+1)
    M[0] = before_last
    M[1] = last
    for i in range(1,n_items+1):
        before_last = M[0]
        last = M[1]
        keep.append([])
        for c in range(capacity+1):
            if weights[i-1] <= c and before_last[c-weights[i-1]] + values[i-1] > before_last[c]:
                last[c] = max(before_last[c],
                              before_last[c-weights[i-1]] + values[i-1])
                keep[i].append(c)
            else:
                last[c] = before_last[c]
        
        del M
        M = []
        
        M.append(last)
        M.append([0] * (capacity+1))
        
        del before_last
        del last
    
    obj = M[0][capacity]
    
    K = capacity
    for i in range(n_items,0,-1):
        if K in keep[i]:
            taken[i-1] = 1
            K = K - weights[i-1]
    
    return (taken, obj)

def solve(capacity, items):
    n_items = len(items)
    
    taken = [0] * n_items 
    
    values = []
    weights = []
    for v,w in items:
        values.append(v)
        weights.append(w)
    
    M = [i[:] for i in [[0]*(capacity+1)]*(2)]
    
    keep = []
    keep.append([])
    M[0] = [0] * (capacity+1)
    M[1] = [0] * (capacity+1)
    for i in range(1,n_items+1):
        keep.append([])
        for c in range(capacity+1):
            if weights[i-1] <= c and M[0][c-weights[i-1]] + values[i-1] > M[0][c]:
                M[1][c] = max(M[0][c],
                              M[0][c-weights[i-1]] + values[i-1])
                keep[i].append(c)
            else:
                M[1][c] = M[0][c]
        
        M[0] = M[1]
        M[1] = [0] * (capacity+1)
            
    obj = M[0][capacity]
    
    K = capacity
    for i in range(n_items,0,-1):
        if K in keep[i]:
            taken[i-1] = 1
            K = K - weights[i-1]
    
    return (taken, obj)

def main():
    if len(sys.argv) != 2:
        print("Usage: %s input_file" % sys.argv[0])
        sys.exit()
             
    file_name = sys.argv[1]

    capacity, items = read_problem(file_name)
    
    taken, obj = solve(capacity, items)
    
    print(obj)
    print(taken[0], end='')
    for t in taken[1:]:
        print(' %d' % t, end='')
    print()

if __name__ == '__main__':
    main()