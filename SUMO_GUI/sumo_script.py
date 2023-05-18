import os
import sys
import sumolib
import traci
import numpy as np 

def run_sumo_simulation(map_file, route_file, step_limit, num_episodes):
    # Connect to the SUMO simulation
    sumo_cmd = ["/home/fahmy/anaconda3/bin/sumo-gui","-c", "tsc.sumocfg"]
    traci.start(sumo_cmd)

    goals = {'agent-1':['edge-north-SN', 48], 'agent-2':['edge-south-NS', 47.5],'agent-3':['edge-north-SN', 46], 'agent-4':['edge-south-NS', 45.2]}
    # Load the network and route files
    net = sumolib.net.readNet(map_file)
    # traci.route.add("route_0", ["edge_0", "edge_1"])  # Replace with your route definition

    info  = {'complete': [], 'time_loss':[], 'n_stops':[], 'Delay':[]}
    for episode in range(num_episodes):
        traci.load(["-c", "tsc.sumocfg", "--start"])  # Reload the simulation for each episode
        edges_list = traci.edge.getIDList()
        avg_veh_delay = []
        avg_cumm_time_loss = []
        avg_edge_delay= [] 
        edge_tt= [] 
        n_stops= [] 
        queue= []
        flow = []
        reached_goal = 0 
        n_collisions = 0 
        violation = 0 
        entered_vehs = 0 
        edges = traci.edge.getIDList()
        for step in range(step_limit):
            step_delay = []
            step_time_loss = []
            step_total_stops = [] 
            traci.simulationStep()

            for edge in edges:
                veh_list = traci.edge.getLastStepVehicleIDs(edge)
                no_stops = traci.edge.getLastStepHaltingNumber(edge)
                step_total_stops.append(no_stops)
                for ids in veh_list: 
                    acc = traci.vehicle.getAccumulatedWaitingTime(ids)
                    time_loss = traci.vehicle.getTimeLoss(ids)
                    step_delay.append(acc)
                    step_time_loss.append(time_loss)
            
            total_stops = np.sum(step_total_stops)
            avg_delay = np.mean(step_delay)
            avg_time_loss = np.mean(step_time_loss)

            avg_veh_delay.append(avg_delay)
            avg_cumm_time_loss.append(avg_time_loss)
            n_stops.append(total_stops)


            for agents in goals.keys():
                if agents in traci.vehicle.getIDList():
                    if traci.vehicle.getRoadID(agents) == goals[agents][0] and traci.vehicle.getLanePosition(agents) >= goals[agents][1]:
                        reached_goal+=1


        info['n_stops'].append(sum(n_stops))
        info['complete'].append(reached_goal)
        info['time_loss'].append(sum(avg_cumm_time_loss))
        info['Delay'].append(sum(avg_veh_delay))



    # End the simulation
    traci.close()


if __name__ == "__main__":
    map_file = "map.net.xml"  # Replace with your map file
    route_file = "Merging_traffic.rou.xml"  # Replace with your route file
    step_limit = 35
    num_episodes = 10

    run_sumo_simulation(map_file, route_file, step_limit, num_episodes)
