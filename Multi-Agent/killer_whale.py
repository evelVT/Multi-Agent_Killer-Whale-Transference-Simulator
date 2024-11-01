'''
The class for the Killer Whale agent and its functions.
'''
import numpy as np
import random
from time import perf_counter
import scipy.stats as stats

from utils import manhattan, euclid

class KillerWhale:
    def __init__(self, position, dialect, age=0, death_age=None, following=None, follow_range=95, comm_range=190):
        self.id = str(perf_counter())
        self.position = np.array(position, dtype=np.int16)                                              # position of the kw (x, y) coords (int array)
        self.dialect = dialect                                                          # the dialect of the kw (binary string)
        self.age = age                                                                  # current age (int or float)
        self.death_age = np.random.normal(70, 10) if death_age == None else death_age   # age of death (int or float)
        self.following = following                                                      # following (KillerWhale) self or other
        self.follow_range = follow_range                                                # valid follow range for movement (int)
        self.comm_range = comm_range                                                    # range of communication (int)
        self.max_distance = 3
        self.size = 8
        self.target = np.zeros(2, dtype=np.float32)
        self.target_angle = 0
        self.target_distance = 0
        self.move_distance = 0
        self.transference = 0.005
        self.mutation = 0.001
        self.grouping_dialect = 0.005
        self.comm_dialect_difference = 0.020
        
        a, mu, sigma = 3.2, 8.5, 1.5
        s = stats.skewnorm(a, mu, sigma).rvs(1)
        self.reproduce_age = s[0] # random sample (7.5 - 14, with mu 8.5 and sigma 1.5)
    
    def compute_target(self):
        self.target_angle = np.random.uniform(0, 2*np.pi)
        self.target_distance = np.sqrt(np.random.uniform(0, 1)) * self.follow_range
        self.target[0] = self.following.position[0] + self.target_distance * np.cos(self.target_angle)
        self.target[1] = self.following.position[1] + self.target_distance * np.sin(self.target_angle)
        return self.target

    def move(self, move_vector, observed_agents, bounds):
        saved_pos = self.position

        move = np.zeros(2)
        # move agent to a new position
        counter = 0
        max_tries = 5

        while( counter <= max_tries ):
            self.move_distance = self.max_distance * random.uniform(0.35, 1)
            if self.following:
                vector = -(self.position - self.compute_target())
                vector_norm = np.linalg.norm(vector)
                if vector_norm != 0: self.position += np.rint( (self.move_distance*1.5) * vector/vector_norm ).astype(np.int16)
            else:
                if move_vector[0]==0:
                    move[0] = np.random.uniform(-1, 1, 1)
                else:
                    move[0] = move_vector[0]
                if move_vector[1]==0:
                    move[1] = np.random.uniform(-1, 1, 1)
                else:
                    move[1] = move_vector[1]
                move = move/np.linalg.norm(move)
                move = move + np.random.uniform(-.5, .5, 2)
                move = move/np.linalg.norm(move)
                move = (move* self.move_distance)
                newPos = np.rint(self.position + move).astype(np.int16)
                
                if newPos[0] >= bounds[0] or newPos[0] < 0:
                    move[0] = -move[0]
                    newPos = np.rint(self.position + move)
                if newPos[1] >= bounds[1] or newPos[1] < 0:
                    move[1] = -move[1]
                    newPos = (self.position + move)
                self.position = np.rint(newPos).astype(np.int16)

            # get new position
            new_vec_pos = self.position

            if (new_vec_pos in observed_agents):
                self.position = saved_pos
                counter += 1
                continue
            
            # if position is valid, break the while loop. Otherwise try again (max 5 times)
            break

    def communicate(self, message):
        if np.sum(message != self.dialect) >= len(self.dialect)*self.comm_dialect_difference:
            return
        # Get the bitwise difference
        difference_array = np.bitwise_xor(self.dialect, message)
        
        # Find the indices where there are differences (non-zero entries)
        diff_bits = np.flatnonzero(difference_array)
        
        if diff_bits.size >= np.rint(self.transference*len(self.dialect)):
            # Randomly select indices to flip
            flip_idx = np.random.choice(diff_bits, size=np.rint(self.transference*len(self.dialect)).astype(np.int32), replace=False)
        else:
            flip_idx = np.random.choice(diff_bits, size=diff_bits.size, replace=False)
        
        # add randomness
        idx = np.random.randint(0, len(message))
        count = 0
        while idx not in flip_idx:
            if count >= np.rint(self.mutation*len(self.dialect)): break
            flip_idx = np.append(flip_idx, idx)
            idx = np.random.randint(0, len(message))
            count += 1
            
        # Flip the selected bits in the dialect
        self.dialect[flip_idx] ^= 1  # Efficient bitwise flip
    
    def mutate_dialect(self):
        n_mutate = int((self.grouping_dialect*len(self.dialect))/2)
        flip_idxs = np.random.randint(0, len(self.dialect), size=n_mutate)
        self.dialect[flip_idxs] ^= 1

    def die(self):
        if self.age >= self.death_age:
            return 1
        else:
            return 0

    def grow(self):
        self.age += 0.01
        return

    def reproduce(self, environment):
        if self.age < 40 and self.age >= self.reproduce_age and self.following is not None:
            if environment.add_child(self):
                a, mu, sigma = 3, 4.5, 2.2
                s = stats.skewnorm(a, mu, sigma).rvs(1)
                self.reproduce_age = self.age + s[0] # random sample (2.5 - 12, with mu 4.5 and sigma 2.2)

    def find_leader(self, kw_list):
        self.following = None
        
        best_lead = None

        for kw in kw_list:
            if kw == self: continue

            #if manhattan(self.position, kw.position) <= self.follow_range:
            if best_lead is None or kw.age > best_lead.age:
                if np.sum(kw.dialect != self.dialect) <= len(self.dialect)*self.grouping_dialect:
                    best_lead = kw
        
        if best_lead and self.age < best_lead.age:
            self.following = best_lead

    def closest(self, kw_list):
        pod = kw_list
        if len(pod) <= 1: return None
        pod.sort(key=lambda kw: manhattan(self.position, kw.position))
        return pod[1]
    
    # setter functions:
    def set_position(self, pos):
        self.position = np.array(pos)
    def set_follow_range(self, d):
        self.follow_range = d
    def set_size(self, s):
        self.size = s
    def set_max_distance(self, d):
        self.max_distance = d
    def set_transference(self, n):
        self.transference = n
    def set_grouping_dialect(self, n):
        self.grouping_dialect = n
    def set_mutation(self, n):
        self.mutation = n
    def set_comm_range(self, r):
        self.comm_range = r
    def set_comm_dialect_difference(self, n):
        self.comm_dialect_difference = n
    def reset_id(self):
        self.id = str(perf_counter())
    
    # getter functions:
    def get_position(self):
        return tuple(self.position)
    # def get_vector_position(self):
    #     return self.position
    # def get_dialect(self):
    #     return self.dialect
    # def get_age(self):
    #     return self.age
    # def get_death_age(self):
    #     return self.death_age
    # def get_following(self):
    #     return self.following
    # def get_follow_range(self):
    #     return self.follow_range
    # def get_comm_range(self):
    #     return self.comm_range
    # def get_size(self):
    #     return self.size
    # def get_max_distance(self):
    #     return self.max_distance