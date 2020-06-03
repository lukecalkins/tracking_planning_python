import yaml
from target import *
from sensor import BearingSensor
from robot import Robot
import dataAssociation as DA
import dataAssociationPlan as DAP
import planner as plan
import cost_function as cf

class Parameters:

    def __init__(self, yaml_file):

        with open(yaml_file, 'r') as file:
            node = yaml.load(file, Loader=yaml.FullLoader)
            target_config = node['targetConfig']
            robots_node = node['Robots']
            sensor_config = robots_node[0]['sensorConfig']
            with open(sensor_config, 'r') as sense_file:
                sense_node = yaml.load(sense_file, Loader=yaml.FullLoader)
                self.detection_prob = sense_node['detection_prob']
                self.unresolved_resolution = sense_node['unresolved_resolution']
            with open(target_config, 'r') as targ_file:
                targ_node = yaml.load(targ_file, Loader=yaml.FullLoader)
                self.y_dim = targ_node['targ_dim']
            planner_config = node['plannerConfig']
            self.samp = node['samp']
            self.Tmax = node['Tmax']
            self.map_min = node['map_min']
            self.map_max = node['map_max']
            self.gate_level = node['gate_level']
            self.n_controls = node['n_controls']
            self.horizon = node['horizon']
            self.clutter_density = node['clutter_density']
            self.estimator_verbose = node['estimator_verbose']

            self.robots = self.buildRobots(yaml_file)
            self.world = self.buildTMM(target_config)
            self.planner = self.buildPlanner(planner_config)
            self.estimator = self.buildEstimator()

    def getRobots(self):
        return self.robots

    def getWorld(self):
        return self.world

    def getPlanner(self):
        return self.planner

    def getEstimator(self):
        return self.estimator

    def buildRobots(self, yaml_file):

        with open(yaml_file, 'r') as file:
            node = yaml.load(file, Loader=yaml.FullLoader)
            robots_node = node['Robots']
            target_config = node['targetConfig']


            info_target_model = InfoTargetModel()
            with open(target_config, 'r') as targ_file:
                targ_node = yaml.load(targ_file, Loader=yaml.FullLoader)
                targ_dim = targ_node['targ_dim']
                IDs = targ_node['IDs']
                y0 = targ_node['y0']
                cov_pos = targ_node['cov_pos']
                cov_vel = targ_node['cov_vel']
                process_cov_pos = targ_node['process_cov_pos']
                process_cov_vel = targ_node['process_cov_vel']
                for i in range(len(y0)):
                    info_target = InfoTarget(y0[i], process_cov_pos, process_cov_vel, self.samp, IDs[i], targ_dim,
                                             cov_pos, cov_vel)
                    info_target_model.addTarget(IDs[i], info_target)

            robots = []
            self.targ_dom = targ_dim
            for robot in robots_node:
                initial_state  = robot['initial_state']
                sensor_config = robot['sensorConfig']
                with open(sensor_config, 'r') as sense_file:
                    sense_node = yaml.load(sense_file, Loader=yaml.FullLoader)
                    sense_min_range = sense_node['min_range']
                    sense_max_range = sense_node['max_range']
                    sense_min_hang = sense_node['min_hang']
                    sense_max_hang = sense_node['max_hang']
                    sense_b_sigma = sense_node['b_sigma']
                    detection_prob = sense_node['detection_prob']

                    sensor = BearingSensor(sense_min_range, sense_max_range, sense_min_hang, sense_max_hang,
                                           sense_b_sigma, detection_prob)

                robots.append(Robot(initial_state, sensor, info_target_model))

        return robots

    def buildTMM(self, target_yaml):
        """
        builds the target motion model that is used for simulation of  target motion during simulation
        :param target_yaml: target model yaml file
        :return:
        """
        target_model = TargetModel([self.map_min, self.map_max])
        with open(target_yaml, 'r') as targ_file:
            targ_node  = yaml.load(targ_file, Loader=yaml.FullLoader)
            targ_dim = targ_node['targ_dim']
            IDs = targ_node['IDs']
            y0 = targ_node['y0']
            cov_pos = targ_node['cov_pos']
            cov_vel = targ_node['cov_vel']
            for i in range(len(y0)):
                target = Target(y0[i], cov_pos, cov_vel, self.samp, IDs[i], targ_dim)
                target_model.addTarget(IDs[i], target)

        return target_model

    def buildPlanner(self, planner_yaml):
        """
        builds planner object for simulation
        :param planner_yaml: planner yaml file
        :return:
        """
        with open(planner_yaml, 'r') as plan_file:
            plan_node = yaml.load(plan_file, Loader=yaml.FullLoader)
            planner_log_flag = plan_node['log_flag']
            planner_log_file = plan_node['log_file']
            planner_final_cost = plan_node['final_cost']
            planner_verbose = plan_node['verbose']
            speed = plan_node['speed']
            turn_radius = plan_node['turn_radius']
            filter_type = plan_node['filter_type']

            # list actions available to vehicle for planning (speed, turn_rate)
            actions = [[speed, 0], [speed, speed / turn_radius], [speed, -speed / turn_radius]]

            JPDAF_simulator = DAP.JPDAF_simulate(self.robots[0].getSensor(), gate_level=self.gate_level, verbose=False)

            cost = plan_node['cost']
            if cost == 'log_det':
                cost_func = cf.LogDetCost(self.y_dim)
            elif cost == 'max_eig':
                cost_func = cf.MaxEigCost(self.y_dim)
            elif cost == 'gate_cost':
                cost_func = cf.GateOverlapCost(self.y_dim, self.gate_level)
            elif cost == 'delta_bearing':
                cost_func = cf.DeltaBearingCost(self.y_dim)
            else:
                print("Cost function not recognized in initialization")
                exit()
            planner = plan.Planner(actions, cost_func, filter_type, JPDAF_simulator, log_file=planner_log_file,
                                   log_flag=planner_log_flag, final_cost=planner_final_cost)

        return planner

    def buildEstimator(self):

        JPDA = DA.JPDAF(detection_prob=self.detection_prob, clutter_density=self.clutter_density,
                        gate_level=self.gate_level, verbose=self.estimator_verbose)

        return JPDA
