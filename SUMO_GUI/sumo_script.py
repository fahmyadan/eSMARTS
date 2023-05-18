import os
import sys
import sumolib
import traci


def run_sumo_simulation(map_file, route_file, step_limit, num_episodes):
    # Connect to the SUMO simulation
    sumo_cmd = ["/home/fahmy/anaconda3/bin/sumo-gui","-c", "tsc.sumocfg"]
    traci.start(sumo_cmd)

    # Load the network and route files
    net = sumolib.net.readNet(map_file)
    # traci.route.add("route_0", ["edge_0", "edge_1"])  # Replace with your route definition

    for episode in range(num_episodes):
        traci.load(["-c", "tsc.sumocfg", "--start"])  # Reload the simulation for each episode
        edges_list = traci.edge.getIDList()
        avg_delay = []
        avg_speed= []
        avg_edge_delay= [] 
        edge_tt= [] 
        n_stops= [] 
        queue= []
        flow = []
        reached_goal = 0 
        n_collisions = 0 
        violation = 0 
        entered_vehs = 0 
        for step in range(step_limit):
            traci.simulationStep()
            # [avg_edge_delay.append(traci.edge.getWaitingTime(i)) for i in edges_list]
            # [edge_tt.append(traci.edge.getTraveltime(i)) for i in edges_list]
            # entered_vehs+=traci.simulation.getDepartedNumber()
        
        vehicle_ids = traci.vehicle.getIDList()
        [avg_edge_delay.append(traci.edge.getWaitingTime(i)) for i in edges_list]
        [edge_tt.append(traci.edge.getTraveltime(i)) for i in edges_list]
        time_loss = [traci.vehicle.getTimeLoss(i) for i in vehicle_ids]
        travel_time_metric = sum(edge_tt) / traci.vehicle.getIDCount()
        total_delay_metric = sum(avg_edge_delay)




    # End the simulation
    traci.close()


if __name__ == "__main__":
    map_file = "map.net.xml"  # Replace with your map file
    route_file = "Merging_traffic.rou.xml"  # Replace with your route file
    step_limit = 35
    num_episodes = 10

    run_sumo_simulation(map_file, route_file, step_limit, num_episodes)
