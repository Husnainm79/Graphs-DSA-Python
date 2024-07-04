def generate_adjacency_matrix(num_vertices, edge_list):
    # Initialize the adjacency matrix with zeros
    matrix = [[0] * num_vertices for _ in range(num_vertices)]
    
    # Fill in the adjacency matrix based on edges
    for edge in edge_list:
        start, end = edge
        matrix[start-1][end-1] = 1
        matrix[end-1][start-1] = 1  # because the graph is undirected
    
    return matrix

def convert_to_adjacency_list(matrix):
    num_vertices = len(matrix)
    adjacency_list = [[] for _ in range(num_vertices)]
    
    # Transform the adjacency matrix to an adjacency list
    for i in range(num_vertices):
        for j in range(num_vertices):
            if matrix[i][j] == 1:
                adjacency_list[i].append(j + 1)
    
    return adjacency_list

# Input values
num_vertices = 8
num_edges = 7
edges = [[1, 2], [2, 3], [4, 5], [1, 5], [6, 1], [7, 4], [3, 8]]

# Create adjacency matrix
adj_matrix = generate_adjacency_matrix(num_vertices, edges)

# Display adjacency matrix
print("Adjacency Matrix:")
for row in adj_matrix:
    print(' '.join(map(str, row)))

# Convert to adjacency list
adj_list = convert_to_adjacency_list(adj_matrix)

# Display adjacency list
print("\nAdjacency List:")
for i in range(len(adj_list)):
    print(f"{i+1}: {adj_list[i]}")

#q2
nodes = ['a', 'b', 'c', 'd', 'e', 'f']
connections = {
    ('a', 'b'): 7,
    ('a', 'c'): 9,
    ('a', 'f'): 14,
    ('b', 'c'): 10,
    ('c', 'd'): 11,
    ('c', 'e'): 2,
    ('d', 'e'): 6,
    ('e', 'f'): 9
}

def build_adjacency_matrix(node_list, edge_dict):
    size = len(node_list)
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    node_index = {node: idx for idx, node in enumerate(node_list)}

    for (node1, node2), weight in edge_dict.items():
        idx1, idx2 = node_index[node1], node_index[node2]
        matrix[idx1][idx2] = weight
        matrix[idx2][idx1] = weight

    return matrix

def show_adjacency_matrix(matrix):
    print("Adjacency Matrix:")
    for row in matrix:
        print(" ".join(map(str, row)))

def display_shared_edge_counts(node_list, matrix):
    shared_edge_counts = [sum(1 for weight in row if weight != 0) for row in matrix]

    print("\nVertices sharing common edges count:")
    for node, count in zip(node_list, shared_edge_counts):
        print(f"{node}: {count} edges")

# Generate and display the adjacency matrix
adj_matrix = build_adjacency_matrix(nodes, connections)
show_adjacency_matrix(adj_matrix)

# Display the number of shared edges for each vertex
display_shared_edge_counts(nodes, adj_matrix)

#q3
from collections import deque

def breadth_first_search(graph, start_node, target_node):
    queue = deque([[start_node]])
    visited_nodes = set()
    
    while queue:
        current_path = queue.popleft()
        current_node = current_path[-1]
        
        if current_node == target_node:
            return current_path
        
        elif current_node not in visited_nodes:
            for neighbor in graph[current_node]:
                new_path = list(current_path)
                new_path.append(neighbor)
                queue.append(new_path)
                
            visited_nodes.add(current_node)

def depth_first_search(graph, start_node, target_node):
    stack = [[start_node]]
    visited_nodes = set()
    
    while stack:
        current_path = stack.pop()
        current_node = current_path[-1]
        
        if current_node == target_node:
            return current_path
        
        elif current_node not in visited_nodes:
            for neighbor in graph[current_node]:
                new_path = list(current_path)
                new_path.append(neighbor)
                stack.append(new_path)
                
            visited_nodes.add(current_node)

# Create the graph as an adjacency list
graph = {
    'A': ['B', 'C', 'F'],
    'B': ['D', 'E'],
    'C': [],
    'D': ['G', 'H'],
    'E': ['I'],
    'F': ['J'],
    'G': ['K', 'L'],
    'H': [],
    'I': ['M'],
    'J': [],
    'K': [],
    'L': [],
    'M': []
}

# Find paths from A to G using BFS and DFS
bfs_result = breadth_first_search(graph, 'A', 'G')
dfs_result = depth_first_search(graph, 'A', 'G')

print("\nBFS path from A to G:", bfs_result)
print("DFS path from A to G:", dfs_result)


#q4
import random

def generate_random_numbers(count, min_value=1, max_value=100000):
    numbers = []
    for _ in range(count):
        numbers.append(random.randint(min_value, max_value))
    return numbers

def check_prime(number):
    if number <= 1:
        return False
    if number <= 3:
        return True
    if number % 2 == 0 or number % 3 == 0:
        return False
    i = 5
    while i * i <= number:
        if number % i == 0 or number % (i + 2) == 0:
            return False
        i += 6
    return True

def find_next_prime(number):
    while not check_prime(number):
        number += 1
    return number

class LinearProbingHashTable:
    def __init__(self, capacity):
        self.capacity = capacity
        self.table = [None] * capacity
        self.total_probes = 0

    def compute_hash(self, key):
        return key % self.capacity

    def add(self, key):
        index = self.compute_hash(key)
        initial_index = index
        probe_count = 0
        while self.table[index] is not None:
            probe_count += 1
            index = (index + 1) % self.capacity
            if index == initial_index:
                raise Exception("Hash table is full")
        self.table[index] = key
        self.total_probes += probe_count

    def get_average_probes(self, number_of_elements):
        return self.total_probes / number_of_elements

class QuadraticProbingHashTable:
    def __init__(self, capacity):
        self.capacity = capacity
        self.table = [None] * capacity
        self.total_probes = 0

    def compute_hash(self, key):
        return key % self.capacity

    def add(self, key):
        index = self.compute_hash(key)
        probe_count = 0
        k = 1
        while self.table[index] is not None:
            probe_count += 1
            index = (index + k ** 2) % self.capacity
            k += 1
        self.table[index] = key
        self.total_probes += probe_count

    def get_average_probes(self, number_of_elements):
        return self.total_probes / number_of_elements

def execute_tests(probing_method, load_ratios, test_iterations=10, elements_to_insert=10000):
    outcomes = {}
    for load_ratio in load_ratios:
        outcomes[load_ratio] = []
        table_capacity = find_next_prime(int(elements_to_insert / load_ratio))
        for _ in range(test_iterations):
            if probing_method == 'linear':
                hash_table = LinearProbingHashTable(table_capacity)
            elif probing_method == 'quadratic':
                hash_table = QuadraticProbingHashTable(table_capacity)
            else:
                raise ValueError("Invalid probing method")
            
            random_numbers = generate_random_numbers(elements_to_insert)
            for number in random_numbers:
                hash_table.add(number)
            
            avg_probes = hash_table.get_average_probes(elements_to_insert)
            outcomes[load_ratio].append(avg_probes)
    
    for load_ratio in load_ratios:
        min_probes = min(outcomes[load_ratio])
        max_probes = max(outcomes[load_ratio])
        avg_probes = sum(outcomes[load_ratio]) / test_iterations
        print(f"Load Ratio {load_ratio}:")
        print(f"  Minimum Average Probes: {min_probes}")
        print(f"  Maximum Average Probes: {max_probes}")
        print(f"  Mean Average Probes: {avg_probes}")
        print()

def main():
    probing_method = input("Enter probing method (linear or quadratic): ").strip().lower()
    load_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    execute_tests(probing_method, load_ratios)

# Initiate the process by calling the main function
main()

