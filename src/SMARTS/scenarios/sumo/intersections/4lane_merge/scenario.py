import os
from pathlib import Path

from numpy import require

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    EndlessMission,
    Flow,
    JunctionEdgeIDResolver,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    Via,
    SocialAgentActor,
)

ego_missions = [
    Mission(
        route=Route(begin=("edge-west-WE", 0, 2), end=("edge-north-SN", 0, "max")),
        via=(Via("edge-west-WE",lane_index=1, lane_offset=30, required_speed=8.0),
             Via("edge-west-WE",lane_index=1, lane_offset=40, required_speed=8.0),
             Via("edge-west-WE",lane_index=1, lane_offset=50, required_speed=8.0),
             Via("edge-west-WE",lane_index=1, lane_offset=60, required_speed=8.0),
             Via("edge-west-WE",lane_index=1, lane_offset=100, required_speed=5.0)), 
            start_time=0.2),#Merging + turning agent
    Mission(route=Route(begin=("edge-east-EW", 0, 2), end=(("edge-south-NS", 0,'max'))),
            via=(Via("edge-east-EW", lane_index=1, lane_offset=30, required_speed=8.0), 
                 Via("edge-east-EW", lane_index=1, lane_offset=40, required_speed=8.0),
                 Via("edge-east-EW", lane_index=1, lane_offset=50, required_speed=8.0),
                 Via("edge-east-EW", lane_index=1, lane_offset=60, required_speed=8.0),
                 Via("edge-east-EW", lane_index=0,lane_offset=80, required_speed=5.0)),
                 start_time=0.2), #Merging + turning agent

    Mission(route=Route(begin=("edge-south-SN", 0, 10), end=(("edge-north-SN", 0,'max'))),
            start_time=0.2), #Agents going straight

    Mission(route=Route(begin=("edge-north-NS", 0, 10), end=(("edge-south-NS", 0,'max'))),
           start_time=0.2), #Agents going straight
    
]

scenario = Scenario(
    traffic={
        "Merging_traffic": Traffic(
            flows=[ #Queue for merging 1
                Flow(
                    route=Route(begin=("edge-west-WE", 1, 5), end=("edge-north-SN", 0, "max")),
                    rate=200,
                    actors={TrafficActor(name="car"): 1.0},
                ),

                #Queue for merging 2
                Flow(
                    route=Route(begin=("edge-east-EW", 1, 5), end=("edge-south-NS", 0, "max")),
                    rate=200,
                    actors={TrafficActor(name="car"): 1.0},
                ),
    
                # Create conflicting Traffic on major roads 
                    #Going straight
                Flow(
                    route=Route(begin=("edge-west-WE", 0, 20), end=("edge-east-WE", 0, "max")),
                    rate=200,
                    actors={TrafficActor(name="car"): 1.0},
                ),
       

    #             Flow(
    #                 route=Route(begin=("edge-east-EW", 0, 20), end=("edge-west-EW", 0, "max")),
    #                 rate=1000,
    #                 actors={TrafficActor(name="car"): 1.0},
    #             ),
  
                    #Turning right/left 
                Flow(
                    route=Route(begin=("edge-west-WE", 0, 20), end=("edge-south-NS", 0, "max")),
                    rate=200,
                    actors={TrafficActor(name="car"): 1.0},
                ),
    #             Flow(
    #                 route=Route(begin=("edge-east-EW", 0, 20), end=("edge-north-SN", 0, "max")),
    #                 rate=500,
    #                 actors={TrafficActor(name="car"): 1.0},
    #             ),
                
    #             #Create conflicting traffic from minor roads (around the CAVs)

                Flow(
                    route=Route(begin=("edge-south-SN", 0, 20), end=("edge-north-SN", 0, "max")),
                    rate=200,
                    actors={TrafficActor(name="car"): 1.0},
                ),
    #             Flow(
    #                 route=Route(begin=("edge-south-SN", 0, 50), end=("edge-north-SN", 0, "max")),
    #                 rate=500,
    #                 actors={TrafficActor(name="car"): 1.0},
    #             ),
                Flow(
                    route=Route(begin=("edge-north-NS", 0, 20), end=("edge-south-NS", 0, "max")),
                    rate=500,
                    actors={TrafficActor(name="car"): 1.0},
                ),
    #             Flow(
    #                 route=Route(begin=("edge-north-NS", 0, 50), end=("edge-south-NS", 0, "max")),
    #                 rate=500,
    #                 actors={TrafficActor(name="car"): 1.0},
    #             ),
                

        ]
        ),
    
    },
    ego_missions=ego_missions,
    # social_agent_missions= social_agent_missions
)

gen_scenario(
    scenario=scenario,
    output_dir=Path(__file__).parent,
)