import os
import sys
import random

import gym
from gym import error, spaces, utils
from  gym.utils import seeding
import numpy as np


# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


class SumoEnv(gym.Env):
    def __init__(self):
        self.cur_step = 0
        self.actions = 3
        self.action_space = spaces.Discrete(self.actions)
        self.num_cars = 4

        # Create an array with the number of dimensions for every lane in the map
        self.num_lanes = 48
        low_observation_space = []
        for i in range(self.num_lanes):
            low_observation_space.append(0)
        high_observation_space = []
        for i in range(self.num_lanes):
            # The max number of cars in a lane is the total number of cars
            high_observation_space.append(self.num_cars)

        self.observation_space = spaces.Box(low=np.array(low_observation_space), high=np.array(high_observation_space))
        self.seed()
        self.nogui = True
        self.prediction_type = "DQN"  # DQN, random, or timed
        self.debug = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.cur_step += 1
        if self.debug and self.cur_step % 10 == 0:
            print("step {} {} {} {}".format(str(self.cur_step), str(action), traci.simulation.getMinExpectedNumber(), self._get_total_vehicle_speed()))
        assert self.action_space.contains(action)

        traci.simulationStep()
        total_speed = self._get_total_vehicle_speed()
        cars_in_lanes = self._get_lanes_info()

        if self.prediction_type == "DQN":
            self._dqn_action(action)
        elif self.prediction_type == "random":
            self._random_action()
        elif self.prediction_type == "timed":
            self._timed_action()

        # Check if simulation is finished
        finished = False
        if traci.simulation.getMinExpectedNumber() <= 0:
            finished = True
            traci.close()
            sys.stdout.flush()

        # Observation space, reward, finished
        return cars_in_lanes, total_speed, finished, {}

    def _dqn_action(self, action):
        if action == 0:
            # Switch to north-south
            traci.trafficlight.setPhase("0", 3)
        elif action == 1:
            # Switch to east-west
            traci.trafficlight.setPhase("0", 2)
        else:
            # Do nothing
            pass

    def _random_action(self):
        random_action = random.randrange(0, 2)
        if random_action == 0:
            # Switch to north-south
            traci.trafficlight.setPhase("0", 3)
        elif random_action == 1:
            # Switch to east-west
            traci.trafficlight.setPhase("0", 2)
        else:
            # Do nothing
            pass

    def _timed_action(self):
        if traci.trafficlight.getPhase("0") == 2:
            # we are not already switching
            if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
                # there is a vehicle from the north, switch
                traci.trafficlight.setPhase("0", 3)
            else:
                # otherwise try to keep green for EW
                traci.trafficlight.setPhase("0", 2)

    def reset(self):
        self.cur_step = 0
        if self.nogui:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')
        self._generate_routefile()
        traci.start([self.sumoBinary, "-c", "sumoenv/data/cross.sumocfg", "--tripinfo-output", "tripinfo.xml"])
        traci.trafficlight.setPhase("0", 2)
        # Run one simulation step to get it started
        traci.simulationStep()
        observation_space = self._get_lanes_info()
        return observation_space

    def _get_total_vehicle_speed(self):
        total = 0
        vehicle_count = 0
        for veh_id in traci.vehicle.getIDList():
            position = traci.vehicle.getSpeed(veh_id)
            total += position
            vehicle_count += 1
        return total / vehicle_count

    def _get_lanes_info(self):
        """
        Returns the number of cars in each lane sorted by the lane id and the values of
        that returned as a list
        """
        lanes = {}
        for lane_id in traci.lane.getIDList():
            lanes[lane_id] = traci.lane.getLastStepVehicleNumber(lane_id)
        sorted_lanes = []
        for key in sorted(lanes.keys()):
            sorted_lanes.append(lanes[key])
        return sorted_lanes

    def _generate_routefile(self):
        random.seed(42)  # make tests reproducible
        N = 3600  # number of time steps
        # demand per second from different directions
        pWE = 1. / 10
        pEW = 1. / 11
        pNS = 1. / 30
        with open("sumoenv/data/cross.rou.xml", "w") as routes:
            print("""<routes>
                <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
        guiShape="passenger"/>
                <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>

                <route id="right" edges="51o 1i 2o 52i" />
                <route id="left" edges="52o 2i 1o 51i" />
                <route id="down" edges="54o 4i 3o 53i" />""", file=routes)
            vehNr = 0
            for i in range(N):
                if random.uniform(0, 1) < pWE:
                    print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pEW:
                    print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                if random.uniform(0, 1) < pNS:
                    print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
            print("</routes>", file=routes)


