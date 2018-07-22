'''
Created on Dec 23, 2017

@author: go

This is a solver to solve Vehicle Routing Problem (VRP)

'''

import sys
import random
from random import shuffle, randint
from math import sqrt

# Genetic Algorithm Class for Minimization Problems
class GA(object):
    def __init__(self, pop_size, chrom_len, pc=0.95, pm=0.1):
        self. population = []
        self.fitness = []
        self.pop_size = pop_size
        self.chrom_len = chrom_len
        self.pc = pc
        self.pm = pm
        
        self.seed = "13579"
        random.seed(self.seed)
    
    def create_chromosome(self):
        chromosome = []
        for i in range(1,self.chrom_len+1):
            chromosome.append(i)
        random.shuffle(chromosome)
        
        return chromosome
    
    def create_population(self):
        for i in range(self.pop_size):
            chromosome = self.create_chromosome()
            self.population.append(chromosome)
    
    def select_by_tournement(self, k=2):
        mating_pool = []
        # create mating pool with same size of population
        for i in range(self.pop_size):
            # k-tournament
            for j in range(k):
                tournament = []
                parent_ind = random.randint(0, self.pop_size-1)
                while(parent_ind in tournament):
                    parent_ind = random.randint(0, self.pop_size-1)
                tournament.append(parent_ind)
            winner_ind = self.fitness.index(min([self.fitness[i] for i in tournament]))
            mating_pool.append(self.population[winner_ind])
        return mating_pool
    
    def crossover(self, mating_pool):
        # pick two parents from mating pool randomly
        parents_ind = []
        parents_ind.append(random.randint(0, self.pop_size-1))
        parents_ind.append(random.randint(0, self.pop_size-1))
        while(parents_ind[0] == parents_ind[1]):
            parents_ind[1] = random.randint(0, self.pop_size-1)
        parent1 = mating_pool[parents_ind[0]]
        parent2 = mating_pool[parents_ind[1]]
        
        # generate a random variable between [0,1)
        prob = random.random()
        if prob < self.pc:
            # create two offsprings by recombination
            # two-point crossover
            folds = []
            folds.append(random.randint(0, self.chrom_len-1))
            folds.append(random.randint(0, self.chrom_len-1))
            while(folds[0] == folds[1]):
                folds[1] = random.randint(0, self.chrom_len-1)
            fold1 = min(folds)
            fold2 = max(folds)
            
            offspring1 = list(parent1)
            offspring2 = list(parent2)
            for i in range(fold1, fold2):
                offspring1[offspring1.index(offspring2[i])] = offspring1[i]
                offspring1[i] = offspring2[i]
                offspring2[offspring2.index(offspring1[i])] = offspring2[i]
                offspring2[i] = offspring1[i]
                    
            return [offspring1, offspring2]
        else:
            # create two offsprings asexually
            return [parent1, parent2]
    
    def mutate(self, offspring):
        offspring_new = list(offspring)
        
        for i in range(self.chrom_len):
            # generate a random variable between [0,1)
            prob = random.random()
            if prob < self.pm:
                ind = random.randint(0, self.chrom_len-1)
                gene = offspring_new[i]
                offspring_new[i] = offspring_new[ind]
                offspring_new[ind] = gene
        return offspring_new
    
    def reproduct(self):
        population_new = []
        # select
        mating_pool = self.select_by_tournement()
        for i in range(self.pop_size//2):
            # crossover
            [offspring1, offspring2] = self.crossover(mating_pool)
            # mutate
            offspring1_mutated = self.mutate(offspring1)
            offspring2_mutated = self.mutate(offspring2)
            # copy offsprings (new parents) into new population
            population_new.append(offspring1_mutated)
            population_new.append(offspring2_mutated)
        # update current population with new generated population
        self.population = population_new
    
    def get_best_solution(self):
        # determine best route for given population
        # fitness must be evaluated first by problem using GA
        if self.fitness != []:
            best_ind = self.fitness.index(min(self.fitness))
            return self.population[best_ind]
        # in case of missing fitness list, return empty list
        return []
    
    def generate(self, gen=1):
        for i in range(gen):
            if self.population == []:
                self.create_population()
            self.reproduct()

class VRP(object):
    def __init__(self, capacity, customers, n_vehicles, ga):
        self.capacity = capacity
        self.customers = customers
        self.n_vehicles = n_vehicles
        self.ga = ga
        self.solution = []
        self.solution_chromosome = []
        self.cost = sys.maxsize
    
    def create_vehicle_routes(self, customer_order):
        # customer_order = chromosome in population
        
        routes = []
        # start with one vehicle
        vehicle_capacity_left = [self.capacity]
        # route starts with warehouse indexed as 0
        route = [0]
        vehicle_ind = 0
        for c in customer_order:
            [demand,_,_] = self.customers[c]
            # start a new vehicle
            if demand > vehicle_capacity_left[vehicle_ind]:
                # return last vehicle to warehouse back
                route.append(0)
                routes.append(route)
                # a new vehicle is necessary
                vehicle_ind += 1
                vehicle_capacity_left.append(self.capacity)
                route = [0]
            # continue with current vehicle
            if demand <= vehicle_capacity_left[vehicle_ind]:
                route.append(c)
                vehicle_capacity_left[vehicle_ind] -= demand
                if c == customer_order[-1]:
                    # no more customer then return last vehicle to warehouse back
                    route.append(0)
                    routes.append(route)
        
        n_used_vehicles = len(vehicle_capacity_left)             
        if n_used_vehicles <= self.n_vehicles:
            # mark unused vehicles as not operated
            for i in range(self.n_vehicles-n_used_vehicles):
                routes.append([0,0])
        else:
            # we exceeded number of vehicle in fleet, so no proper routes exist
            routes = []
            
        return routes
            
    
    def evaluate_population_fitness(self):
        fitness = []
        for chromosome in self.ga.population:
            routes = self.create_vehicle_routes(chromosome)
            total_route_distance = self.compute_total_route_distance(routes)
            fitness.append(total_route_distance)
        
        elite = [self.solution_chromosome, self.cost]
        if elite[1] != 0.0:
            best_chromosome,cost = elite
            # if best chromosome from previous population is 
            # better than the worst chromosome in current population 
            # then replace the best with the worst
            if cost < max(fitness):
                self.ga.population[fitness.index(max(fitness))] = best_chromosome
                fitness[fitness.index(max(fitness))] = cost
        
        self.ga.fitness = fitness
        self.check_for_better_solution()
        
    def check_for_better_solution(self):
        best_chromosome = self.ga.get_best_solution()
        solution = self.create_vehicle_routes(best_chromosome)
        cost = self.compute_total_route_distance(solution)
        if cost < self.cost:
            self.solution = solution
            self.solution_chromosome = best_chromosome
            self.cost = cost
    
    def compute_distance(self, customer1_ind, customer2_ind):
        [_,x1,y1] = self.customers[customer1_ind]
        [_,x2,y2] = self.customers[customer2_ind]
        
        # compute euclidean distance between two customers
        return sqrt(pow((x1-x2),2) + pow((y1-y2),2))
    
    def compute_route_distance(self, route):
        route_distance = 0.0
        for i in range(len(route)-1):
            route_distance += self.compute_distance(route[i], route[i+1])
        
        return route_distance
    
    def compute_total_route_distance(self, routes):
        total_route_distance = 0.0
        for r in routes:
            total_route_distance += self.compute_route_distance(r)
        
        # In case of no route found, return big number not zero
        if total_route_distance == 0.0:
            return sys.maxsize
        
        return total_route_distance
    
    def get_solution(self):
        return [self.solution, self.cost]

def read_problem(file_name):
    # read the file to get information about the problem
    lines = open(file_name).readlines()
     
    # get the number of customers (N), number of vehicles (V) and capacity of each vehicle (c)
    tokens = lines[0].split()
    n_customers = int(tokens[0])
    n_vehicles = int(tokens[1])
    capacity = int(tokens[2])
     
    # get the demand (d) and point (x and y) of each customer
    customers = []
    for line in lines[1:n_customers+1]:
        tokens = line.split()
        demand = int(tokens[0])
        x = float(tokens[1])
        y = float(tokens[2])
        customer = (demand, x, y)
        customers.append(customer)

    return (capacity, customers, n_vehicles)

# simple genetic algorithm
def solve(capacity, customers, n_vehicles):
    population_size = 50
    # exclude warehouse indexed as 0. item
    chromosome_length = len(customers) - 1
    generations = 100000
    
    ga = GA(population_size, chromosome_length)
    ga.create_population()
    
    vrp = VRP(capacity, customers, n_vehicles, ga)
    
    for i in range(generations):
        # evaluate individuals in populations
        vrp.evaluate_population_fitness()
        # generate population with SGA
        ga.generate()
    
    return vrp.get_solution()

def main():
    if len(sys.argv) != 2:
        print("Usage: %s input_file" % sys.argv[0])
        sys.exit()
                 
    file_name = sys.argv[1]

    capacity, customers, n_vehicles = read_problem(file_name)
    
    solution, cost = solve(capacity, customers, n_vehicles)
    
    print(cost)
    for s in solution:
        print(s[0], end='')
        for t in s[1:]:
            print(' %d' % t, end='')
        print()

if __name__ == '__main__':
    main()
