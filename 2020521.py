import pandas as pd
import numpy as np
import queue
from collections import deque
from haversine import haversine as haversine_formula
from scipy.spatial.distance import euclidean

# the data frame that contains the entire data
data = pd.read_csv("Road_Distance.csv")
data.replace("-", 0, inplace=True)

# the dictionary that contains the coordinates for the heuristic
city_coordinates = pd.read_csv("coordinates.csv").to_dict(orient='dict')

# the set of all the cities
cities = set()
array = data.columns
for i in range(1,len(array)):
    cities.add(array[i])

# the array of data values
table = data.values
for i in range(0,len(table)):
    cities.add(table[i][0])

# Initialize an empty graph dictionary
graph = {}

# Add city names to the graph dictionary
for city in cities:
    graph[city] = {}

# Now, the graph dictionary contains all city names with empty dictionaries as values
columns = data.columns
for i in range(0,len(table)):
    # table[i][0] = city name
    city_name = table[i][0]
    # Now i want to turn each value into another dictionary with keys as the columns of the data data frame
    # The value of the keys will be the distance between the two cities
    for j in range(1, len(columns)):
        # Get the city as columns[j]
        neighbor_city = columns[j]

        # Get the distance as table[i][j]
        distance = table[i][j]

        # Enter these two as key-value pair in the dictionary graph[city_name]
        graph[city_name][neighbor_city] = float(distance)
        graph[neighbor_city][city_name] = float(distance)
        
# remove all the self edges
for city in cities:
    if city in graph and city in graph[city]:
        del graph[city][city]

# the function that performs relaxation on edges
def relaxation(node, distances, parent):
    # We find out all the neighbours of this node
    for key,value in graph[node].items():
        # For each neighbour we find out their distance to the start node
        dist = distances[key]
        edge_weight = value
        dist_node = distances[node]
        # If the distance of the neighbour to teh start node is greater than the distance between neighbour and node
        if dist_node + edge_weight < dist:
            distances[key] = dist_node + edge_weight
            parent[key] = node
        # Then we update the distance and the parent 

# distances dictionary will have the min distance of all the cities to the start city.
distances = {}
# the parents to get the path after dijkstra's algorithm
parent = {}
# the dijkstra's algorithm
def dijkstra(start):
    # We fill the dictionary of distances
    global distances
    distances = {}
    for city in cities:
        if city == start:
            distances[city] = 0
        else:
            distances[city] = float('inf')
    
    # We fill the dictionary of parents.
    global parent
    parnet = {}
    for city in cities:
        parent[city] = 'NA'
        
    
    frontier = queue.Queue()
    reached = set()
    
    # Step 1 = Put start in the frontier
    frontier.put(start)
    # Step 2 = While frontier is not empty we start a while loop
    while frontier.empty() == False:
        # Step 3 = Perform relaxation for all the edges of the curr node
        curr = frontier.get()
        relaxation(curr,distances, parent)
        # Step 4 = Put all the neighbours of the current node that have not been reached into the frontier
        for item in graph[curr]:
            if item not in reached:
                frontier.put(item)
            reached.add(curr)

# Get the path after performing dijkstra's algorithm
def get_Path(start, goal):
    stack = deque()
    curr = goal
    while curr != start:
        stack.append(curr)
        curr = parent[curr]
    stack.append(start)
    while len(stack) > 0:
        print(stack.pop())

# the function to get the haversine distance between two cities
def haversine(node, neighbour):
    nc = city_coordinates[node]
    nec = city_coordinates[neighbour]
    ans = haversine_formula(nc, nec)
    return ans

# the function to get the euclidean distance between two cities
def euclidean_distance(node, neighbour):
    # Calculate Euclidean distance
    nc = tuple(city_coordinates[node])
    nec = tuple(city_coordinates[neighbour])
    ans = euclidean(nc, nec)
    return ans

g = {}
def a_star_search(start,goal,h):
    # the nodes i can visit
    frontier = queue.PriorityQueue()
    frontier.put((0,start))
    
    # The current distance of a node from teh starting node
    global g
    g = {} 
    g[start] = 0
    for city in cities:
        if city != start:
            g[city] = float('inf')
            
    # What all nodes have i reached
    reached = {}
    
    # parent node required while tracing back
    parents = {}
    parents[start] = start

    # stores the heuristic distance from teh node to the goal node.
    heuristic = {}
    for city in cities:
        if h == "inadmissible":
            heuristic[city] = euclidean_distance(city,goal)
        else:
            heuristic[city] = haversine(city,goal)
        
    # minimum distance to the goal node
    minimum_to_goal = float('inf')
    
    while frontier.empty() == False:
        (f, city) = frontier.get()
        if f <= minimum_to_goal:
            print(f'Current city = {city} at dist = {f}. Min distance to goal = {minimum_to_goal}')
            # we expand this node
            for neighbour,distance in graph[city].items():
                # if we can reach this neighbour in distance smaller than g[neighbour] then we update it
                # print(f"{neighbour} {distance}")
                if g[city] + distance < g[neighbour]:
                    g[neighbour] = distance + g[city]
                    if neighbour == goal:
                        minimum_to_goal = min(minimum_to_goal,g[neighbour])
                    frontier.put((g[neighbour]+heuristic[neighbour],neighbour))
                reached[city] = f

if __name__ == "__main__":
    flag = True
    while flag:
        user_input = input("Do you want to find the route? (yes/no) => ")
        if(user_input == "yes"):
            print("Alright then you have to give me some input.")
            start = input("Enter starting location => ")
            goal = input("Enter final location => ")
            print("What kind of approach should I take?")
            algo = input("UCS/A* Search (u/a) => ")
            if algo == "u":
                dijkstra(start)
                get_Path(start,goal)
            elif algo == "a":
                admissibility = input("Do you want admissible heuristic? (yes/no) => ")
                if admissibility == "yes":
                    a_star_search(start,goal,"admissible")
                else:
                    a_star_search(start,goal,"inadmissible")
        elif(user_input == "no"):
            print("Thank you.")
            flag = False
        else:
            print("You seem to have made a mistake.")