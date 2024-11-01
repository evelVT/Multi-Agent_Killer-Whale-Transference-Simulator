'''
The class for the environment of the simulation
'''
import numpy as np
from time import perf_counter
import cv2
from skimage import draw
from copy import deepcopy
import scipy.stats as stats

from utils import manhattan, euclid, manhattan_observe, euclidean_observe

class Environment:
    def __init__(self, size, agents, parameters):
        self.size = size
        self.agents = agents
        self.env_layer = np.zeros(size, dtype=np.uint8) # drawing canvas for land/water distinction
        self.food_layer = np.zeros(size, dtype=np.float32)
        self.env_parameters = parameters
        self.agent_positions = np.array([a.position for a in self.agents], dtype=np.uint16)
        self.pod_positions = np.array([a.position for a in self.agents], dtype=np.uint16)
        self.communication = None ###
        # self.communication = {} ###
        self.pod_position_to_index = {pos: idx for idx, pos in enumerate(self.pod_positions)}
        self.agent_tuple_pos = {agent: agent.get_position() for agent in self.agents}
        self.leader_cache = {}
        self.vector_layer = np.zeros_like(self.env_layer, dtype=np.float32)
        self.land_edges = cv2.Canny(self.env_layer,1,1)
        self.offspring = []
        
        # Create the border layer with high values along the edges
        border_thickness = 5
        self.border_layer = np.zeros_like(self.env_layer, dtype=np.float32)
    
    def add_agents(self, agent_array):
        self.agents.extend(agent_array)

    def add_agent(self, agent):
        self.agents.append(agent)
    
    def add_child(self, agent):
        if np.random.rand() < .5: # half the births (to account for males)
            return 1

        child = deepcopy(agent)
        child.position += np.ceil(np.random.uniform(-1, 1, 2)).astype(np.int16)
        # position validation
        if (np.any(child.position < 0) or              # any coord under 0
            child.position[0] >= self.size[0] or       # width >= surface width
            child.position[1] >= self.size[1] or       # height >= surface height
            self.env_layer[tuple(child.position)] or          # on land
            child.position in self.agent_positions):   # on other agent
            return 0
        else:
            child.mutate_dialect()
            child.death_age = np.random.normal(70, 10)
            child.age = 0
            child.reset_id()
            a, mu, sigma = 3.2, 8.5, 1.5
            s = stats.skewnorm(a, mu, sigma).rvs(1)
            child.reproduce_age = s[0] # random sample (7.5 - 14, with mu 8.5 and sigma 1.5)
            self.offspring.append(child)
            return 1


    def reset_agents(self):
        self.agents = []
        self.agent_positions = []
        self.offspring = []
    
    def update_food(self):
        if len(self.agent_positions) > 0:
            self.border_layer[:5, :] = 1  # Top border
            self.border_layer[-5:, :] = 1  # Bottom border
            self.border_layer[:, :5] = 1  # Left border
            self.border_layer[:, -5:] = 1  # Right border
            self.border_layer = cv2.blur(self.border_layer,(5,5))
            self.border_layer = 0.9*self.border_layer
            self.border_layer = np.clip(self.border_layer, 0, 255)

            self.food_layer[self.agent_positions[:,0], self.agent_positions[:,1]] += 200
            self.food_layer = cv2.blur(self.food_layer,(3,3), borderType=cv2.BORDER_REFLECT)
            self.food_layer[:2, :] *= 1.005  # Top border
            self.food_layer[-2:, :] *= 1.005  # Bottom border
            self.food_layer[:, :2] *= 1.005  # Left border
            self.food_layer[:, -2:] *= 1.005  # Right border
            self.food_layer = np.where(self.env_layer == 1, self.food_layer * 1.005, self.food_layer) # Land
            self.food_layer = 0.9955*self.food_layer
            self.food_layer = np.clip(self.food_layer, 0, 255)
        return self.food_layer
    
    def reset_food(self):
        self.food_layer = np.zeros(self.size, dtype=np.float32)

    def update_positions(self):
        self.agent_positions = np.array([a.position for a in self.agents])
        self.pod_positions = [a.position for a in self.agents if a.following is None]
        self.pod_position_to_index = {tuple(pos): idx for idx, pos in enumerate(self.pod_positions)}
        self.agent_tuple_pos = {agent: agent.get_position() for agent in self.agents}
        self.leader_cache = {}
        env_weight = 2; food_weight = 1; border_weight = 2;
        self.vector_layer = ((self.env_layer.astype(np.float32)*env_weight) + 
                             ((self.food_layer/255)*food_weight) + 
                             (self.border_layer*border_weight))

    # communication check
    def can_communicate(self, a1, a2):
        rr, cc = draw.line(a1.position[0], a1.position[1], a2.position[0], a2.position[1])
        rr_sampl = rr[::10]
        cc_sampl = cc[::10]
        
        if np.any(self.env_layer[rr_sampl, cc_sampl]) or not (np.sum(a2.dialect != a1.dialect) <= len(a1.dialect)*a1.grouping_dialect):
            return 0
        else:
            return 1

    # create communication matrix
    def update_communicate(self):
        if 'Follow_range' in self.env_parameters: 
            follow_range = self.env_parameters['Follow_range']
        else:
            follow_range = 200
        pods = len(self.pod_positions)
        self.communication = np.zeros((pods, pods), dtype=np.uint8)

        for i, me_pos in enumerate(self.pod_positions[::-1]):
            a_idx = pods -1 - i

            for b_idx, you_pos in enumerate(self.pod_positions):
                if euclid(me_pos, you_pos) > 2*follow_range:
                    self.communication[a_idx, b_idx] = 0
                    continue
                if a_idx == b_idx:
                    self.communication[a_idx, b_idx] = 1
                    break
                
                rr, cc = draw.line(me_pos[0], me_pos[1], you_pos[0], you_pos[1])
                rr_sampl = rr[::10]
                cc_sampl = cc[::10]
                
                if np.any(self.env_layer[rr_sampl, cc_sampl]):
                    self.communication[a_idx, b_idx] = 0
                else:
                    self.communication[a_idx, b_idx] = 1
        self.communication += self.communication.T
    

    # def send_communication(self, agent_idx):
    #     comm_agent = self.agents[agent_idx]

    #     bit = 0
    #     for i, agent in enumerate(self.agents):
    #         if bit==0 and i>=agent_idx:
    #             bit = 1
    #         if agent_idx < i+bit:
    #             lowid = agent_idx
    #             highid = i+bit
    #         else:
    #             lowid = i+bit
    #             highid = agent_idx
            
    #         if self.communication[(lowid, highid)]:
    #             if self.communication[(lowid, highid)] == 1:
    #                 agent.communicate(comm_agent.dialect)
    #         else:
    #             if self.can_communicate(comm_agent, agent):
    #                 agent.communicate(comm_agent.dialect)
    #                 self.communication[(lowid, highid)] = 1
    #             else:
    #                 self.communication[(lowid, highid)] = 0

    # Helper function for finding the leader index to use in caching
    def _find_leader_idx(self, agent_idx):
        leader = self.agents[agent_idx]
        while leader.following:
            leader = leader.following
        return self.pod_position_to_index[self.agent_tuple_pos[leader]]

    def send_communication(self, agent_idx):
        comm_agent = self.agents[agent_idx]
        comm_leader = comm_agent.following

        # find pod leader of communicating agent
        if comm_leader == None:
            comm_leader = comm_agent
        while comm_leader.following is not None:
            comm_leader = comm_leader.following
        
        # get the index of the leader of communicating agent
        comm_leader_idx = self.pod_position_to_index[self.agent_tuple_pos[comm_leader]]
        self.leader_cache[agent_idx] = comm_leader_idx

        # find the pod leader of every agent
        for i, agent in enumerate(self.agents):
            if manhattan(agent.position, comm_agent.position) > (comm_agent.comm_range)+10:
                continue
            follow = agent.following

            if follow is None:
                leader = self.pod_position_to_index[self.agent_tuple_pos[agent]]
            else:
                if i not in self.leader_cache:
                    while follow.following != None:
                        follow = follow.following
                    self.leader_cache[i] = self.pod_position_to_index[self.agent_tuple_pos[follow]]
                leader = self.leader_cache[i]

            # when agents can communicate (unbroken connection; no land inbetween), send communication
            if self.communication[comm_leader_idx, leader]:
                agent.communicate(comm_agent.dialect) # agent receives communication from communicating agent
    
    
    def update_agent(self, agent):
            # get parameters:
            agent_age = agent.age
            agent_vector_position = agent.position.copy()
            surface_size = self.env_parameters['Surface_size']

            # set parameters
            if 'Follow range' in self.env_parameters: agent.set_follow_range(self.env_parameters['Follow range'])
            if 'Size' in self.env_parameters: agent.set_size(self.env_parameters['Size'])
            if 'Movement distance' in self.env_parameters: agent.set_max_distance(self.env_parameters['Movement distance'])
            if 'Transference' in self.env_parameters: agent.set_transference(self.env_parameters['Transference']/10000)
            if 'Mutation' in self.env_parameters: agent.set_mutation(self.env_parameters['Mutation']/10000)
            if 'Grouping Dialect' in self.env_parameters: agent.set_grouping_dialect(self.env_parameters['Grouping Dialect']/10000)
            if 'Communication range' in self.env_parameters: agent.set_comm_range(self.env_parameters['Communication range'])
            if 'Max comm difference' in self.env_parameters: agent.set_comm_dialect_difference(self.env_parameters['Max comm difference']/10000)

            # find new following if
            if agent.following is None or manhattan(agent.position, agent.following.position) > agent.follow_range or not self.can_communicate(agent, agent.following) or agent.following not in self.agents:
                # Calculate the distance of all agents in a vectorized way
                #distances =  np.linalg.norm(self.agent_positions - agent_vector_position, axis=1) # euclidean
                distances =  np.sum(np.abs(self.agent_positions - agent_vector_position), axis=1) # manhattan

                # Get the indices of agents within the threshold distance
                close_indices = np.where(distances <= agent.follow_range)[0]

                # Use the indices to filter the original list of agents
                close_agents = [self.agents[i] for i in close_indices if self.can_communicate(agent, self.agents[i])]

                # find the oldest agent to follow within follow range
                if agent.following not in close_agents:
                    agent.find_leader(close_agents)

            # move agent to a new position
            # observation coordinates:
            if agent.following is None:
                observed_env = euclidean_observe(agent.position, agent.max_distance, self.vector_layer)
            else: observed_env = [0.0, 0.0]

            # move agent to a new position
            counter = 0
            max_tries = 5

            while( counter < max_tries ):
                # move
                if counter > 0:
                    agent.move(np.random.normal(-1, 1, 2), self.agent_positions, self.env_layer.shape)
                else:
                    agent.move(observed_env, self.agent_positions, self.env_layer.shape)
                

                # get new position
                new_vec_pos = agent.position.copy()
                new_pos = tuple(new_vec_pos)
                
                # POSITION VALIDATION:
                # reset position if outside accessible environment
                if (np.any(new_vec_pos < 0) or              # any coord under 0
                new_vec_pos[0] >= surface_size[0] or        # width >= surface width
                new_vec_pos[1] >= surface_size[1] or        # height >= surface height
                self.env_layer[new_pos]):                   # on inaccessible (e.g. land)
                    agent.set_position(agent_vector_position)   # reset position
                    counter += 1                                # add 1 to tries counter
                    continue
                
                # reset position if overwriting other agents #!old code: manhattan(closest_agent.position, new_vec_pos) <= 2*agent.size
                if (new_vec_pos in self.agent_positions):
                    agent.set_position(agent_vector_position)
                    counter += 1
                    continue

                # if position is valid, break the while loop. Otherwise try again (max 5 times)
                break
            
            # offspring and aging
            agent.grow()
            agent.reproduce(self)

    def update(self):
        # remove agents that die of age or if stranded on inaccessible area (e.g. land)
        self.agents = [agent for agent in self.agents if not (agent.die() or self.env_layer[agent.get_position()] or self.food_layer[agent.get_position()] > 220)]
        self.update_positions()

        # update agents
        for i, agent in enumerate(self.agents):
            self.agent_positions = np.delete(self.agent_positions, i, axis=0)
            self.agents = np.delete(self.agents, i)
            self.update_agent(agent)
            self.agent_positions = np.insert(self.agent_positions, i, agent.position, axis=0)
            self.agents = np.insert(self.agents, i, agent).tolist()

        
        self.update_positions()
        self.update_communicate()

        for i, agent in enumerate(self.agents):
            self.send_communication(i)
        
        if len(self.offspring) > 0:
            self.add_agents(self.offspring)
            self.offspring = []
        # self.communication = {} ###
        return self.agents, self.agent_positions
    

    def update_edges(self):
        self.land_edges = cv2.Canny(self.env_layer,1,1)

    # setter functions
    def set_parameters(self, parameters):
        self.env_parameters = parameters
    
    # getter functions
    def get_layers(self):
        return (self.env_layer, self.agent_layer)
    def get_tile_layers(self, pos):
        return (self.env_layer[pos], self.agent_layer[pos])
    def get_agent_layer(self):
        return self.agent_layer
    def get_environment_layer(self):
        return self.env_layer
    def get_agents(self):
        return self.agents