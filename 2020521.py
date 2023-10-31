import pandas as pd
import numpy as np
import random
import queue
from collections import deque
from haversine import haversine as haversine_formula
from scipy.spatial.distance import euclidean

# the data frame that contains the entire data
data = pd.read_csv("Road_Distance.csv")
data.replace("-", 0, inplace=True)

# the dictionary that contains the coordinates for the heuristic
city_coordinates = {
    'Cochin': (9.9312, 76.2673),
    'Jabalpur': (23.1815, 79.9864),
    'Vishakapatnam': (17.6868, 83.2185),
    'Bombay': (19.0760, 72.8777),
    'Hyderabad': (17.3850, 78.4867),
    'Lucknow': (26.8467, 80.9462),
    'Imphal': (24.8170, 93.9368),
    'Amritsar': (31.6340, 74.8723),
    'Bangalore': (12.9716, 77.5946),
    'Indore': (22.7196, 75.8577),
    'Madras': (13.0827, 80.2707),
    'Pune': (18.5204, 73.8567),
    'Shimla': (31.1048, 77.1734),
    'Asansol': (23.6760, 86.9784),
    'Shillong': (25.5760, 91.8820),
    'Panjim': (15.4909, 73.8278),
    'Gwalior': (26.2183, 78.1828),
    'Kolhapur': (16.7050, 74.2433),
    'Varanasi': (25.3176, 82.9739),
    'Baroda': (22.3072, 73.1812),
    'Calcutta': (22.5726, 88.3639),
    'Allahabad': (25.4358, 81.8463),
    'Meerut': (28.6139, 77.2828),
    'Nasik': (19.9975, 73.7898),
    'Jaipur': (26.9124, 75.7873),
    'Surat': (21.1702, 72.8311),
    'Jamshedpur': (22.8046, 86.2029),
    'Chandigarh': (30.7333, 76.7794),
    'Calicut': (11.2588, 75.7804),
    'Madurai': (9.9252, 78.1198),
    'Hubli': (15.3647, 75.1240),
    'Vijayawada': (16.5062, 80.6480),
    'Ahmedabad': (23.0225, 72.5714),
    'Ranchi': (23.3441, 85.3096),
    'Bhopal': (23.2599, 77.4126),
    'Bhubaneshwar': (20.2961, 85.8245),
    'Delhi': (28.6139, 77.2090),
    'Nagpur': (21.1458, 79.0882),
    'Agra': (27.1767, 78.0081),
    'Coimbatore': (11.0168, 76.9558),
    'Kanpur': (26.4499, 80.3319),
    'Trivandrum': (8.5241, 76.9366),
    'Pondicherry': (11.9139, 79.8145),
    'Ludhiana': (30.9010, 75.8573),
    'Agartala': (23.8315, 91.2868),
    'Patna': (25.5941, 85.1376),
    'Jullundur': (31.3260, 75.5762),
}

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
        city = stack.pop()
        if len(stack) == 0:
            print(city ,end=" : ")
        else:
            print(f"{city} -> ",end ="")
    print(f"Distance to the goal = {distances[goal]}")

# the function to get the haversine distance between two cities
def haversine(node, goal):
    nc = city_coordinates[node]
    nec = city_coordinates[goal]
    given_distance_node_goal = float('inf') if goal not in graph[node] else graph[node][goal]
    given_distance_goal_node = float('inf') if goal not in graph[goal] else graph[goal][node]
    given_distance = min(given_distance_node_goal, given_distance_goal_node)
    ans = min(haversine_formula(nc, nec),given_distance)
    return ans

# the function to get the euclidean distance between two cities
def h_random(start, goal):
    base_heuristic = haversine(start, goal)
    max_random_offset = base_heuristic*5
    random_offset = random.uniform(0, max_random_offset)
    return base_heuristic + random_offset

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
            heuristic[city] = h_random(city,goal)
        else:
            heuristic[city] = haversine(city,goal)

    # minimum distance to the goal node
    minimum_to_goal = float('inf')
    count = 1
    while frontier.empty() == False:
        (f, city) = frontier.get()
        if f <= minimum_to_goal:
            # print(f'{count} Expand {city}. Min distance to goal = {minimum_to_goal}')
            count = count + 1
            # we expand this node
            for neighbour,distance in graph[city].items():
                # if we can reach this neighbour in distance smaller than g[neighbour] then we update it
                # print(f"{neighbour} {distance}")
                if g[city] + distance < g[neighbour]:
                    g[neighbour] = distance + g[city]
                    parent[neighbour] = city
                    if neighbour == goal:
                        minimum_to_goal = min(minimum_to_goal,g[neighbour])
                    frontier.put((g[neighbour]+heuristic[neighbour],neighbour))
                reached[city] = f

    # to get the path
    print()
    print("Here comes the path")
    stack = deque()
    curr = goal
    while curr != start:
        stack.append(curr)
        curr = parent[curr]
    stack.append(start)
    while len(stack) > 0:
        city = stack.pop()
        if len(stack) == 0:
            print(city ,end=" : ")
        else:
            print(f"{city} -> ",end ="")
    print(f"Distance to the goal = {g[goal]}")

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
    print("------------------------------------------------------------------------------------")
    '''
    for city1 in cities:
        for city2 in cities:
            if city1 != city2:
                print("First dijkstra:-")
                dijkstra(city1)
                get_Path(city1,city2)

                print("A* with admissible heuristic:-")
                a_star_search(city1,city2,"admissible")

                print("A* with non admissible heuristic:-")
                a_star_search(city1,city2,"inadmissible")
    '''


'''
Surat to Delhi gives the wrong answer with in admissible heuristic
Coimbatore to Delhi as well

To find out cases in which the nodes expanded were less:-
Delhi to Imphal
Delhi to Kanpur
'''
