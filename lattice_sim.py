import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class LatticePop:
    def __init__(self, mut_matrix, birth_rates, death_rates, geometry="cylinder"):
        # mutation rate per unit time from i -> j
        self.mu = np.copy(mut_matrix)
        self.total_mut_rates = np.sum(self.mu, axis=1).flatten()
        self.birth = np.copy(birth_rates)
        self.death = np.copy(death_rates)
        self.geom = geometry
        self.locs = None#np.zeros((1,1), dtype=np.int64)
        self.types = None#np.zeros((1,1))
        self.total_cells = None#0
        self.total_rate = None#0
        self.num_types = len(self.death)
        self.max_neighbors = 9
        self.early_term = False
        self.stop_time = 0.
        
        assert len(self.birth) == len(self.death)
        assert np.shape(self.mu)[0] == np.shape(self.mu)[1]
        assert np.shape(self.mu)[0] == len(self.birth)
        assert self.geom in ["cylinder", "square"]
    
    def get_adj_idx(self, loc):
        new_x = np.arange(-1,2) + loc[1]
        new_y = np.arange(-1,2) + loc[0]
        if self.geom == "cylinder":
            if new_x[0] < 0:
                new_x[0] = self.xmax - 1
            if new_x[2] == self.xmax:
                new_x[2] = 0
        elif self.geom == "square":
            if new_x[0] < 0:
                new_x = new_x[1:]
            if new_x[-1] == self.xmax:
                new_x = new_x[:-1]
        if new_y[0] < 0:
            new_y = new_y[1:]
        if new_y[-1] == self.ymax:
            new_y = new_y[:-1]
        to_return = []
        for j in new_x:
            for i in new_y:
                if not (i == loc[0] and j == loc[1]):
                    to_return.append((i,j))
        return(to_return)
    
    def get_adj_cells(self, loc):
        return([x for x in filter(lambda x: self.locs[x[0], x[1]] >= 0, self.get_adj_idx(loc))])
    
    def get_adj_empties(self, loc):
        return([x for x in filter(lambda x: self.locs[x[0], x[1]] < 0, self.get_adj_idx(loc))])
    
    def get_num_neighbors(self, loc):
        return(len(self.get_adj_cells(loc)))
    
    def get_num_empties(self, loc):
        return(len(self.get_adj_empties(loc)))
    
    def calc_mut_mat(self):
        return(np.multiply(self.types, self.total_mut_rates.reshape((-1,1))))
    
    def calc_birth_mat(self):
        neighbor_adjusted = np.multiply(self.types, np.arange(self.max_neighbors).reshape((1,-1)))
        return(np.multiply(neighbor_adjusted, self.birth.reshape((-1,1))))
    
    def calc_death_mat(self):
        return(np.multiply(self.types, self.death.reshape((-1,1))))
    
    def update_total_rate(self):
        self.total_rate = np.sum(self.calc_birth_mat() + self.calc_death_mat() + self.calc_mut_mat())
    
    def add_cell(self, loc, new_type):
        occ = self.get_num_empties(loc)
        self.types[new_type, occ] += 1
        self.locs[loc[0], loc[1]] = new_type
        self.type_to_loc[new_type][occ].append(loc)
        for neighbor in self.get_adj_cells(loc):
            new_occ = self.get_num_empties(neighbor)
            neighbor_type = self.locs[neighbor[0], neighbor[1]]
            self.types[neighbor_type, new_occ] += 1
            self.type_to_loc[neighbor_type][new_occ].append(neighbor)
            self.types[neighbor_type, new_occ+1] -= 1
            self.type_to_loc[neighbor_type][new_occ+1].remove(neighbor)
            assert self.types[neighbor_type, new_occ+1] >= 0
        self.update_total_rate()
        self.total_cells += 1
    
    def remove_cell(self, loc):
        occ = self.get_num_empties(loc)
        old_type = self.locs[loc[0], loc[1]]
        self.types[old_type, occ] -= 1
        self.locs[loc[0], loc[1]] = -1
        self.type_to_loc[old_type][occ].remove(loc)
        for neighbor in self.get_adj_cells(loc):
            new_occ = self.get_num_empties(neighbor)
            neighbor_type = self.locs[neighbor[0], neighbor[1]]
            self.types[neighbor_type, new_occ] += 1
            self.type_to_loc[neighbor_type][new_occ].append(neighbor)
            self.types[neighbor_type, new_occ-1] -= 1
            self.type_to_loc[neighbor_type][new_occ-1].remove(neighbor)
            assert self.types[neighbor_type, new_occ-1] >= 0
        self.update_total_rate()
        self.total_cells -= 1
    
    def set_pop_state(self, pop_mat):
        self.type_to_loc = {}
        for type_idx in range(self.num_types):
            self.type_to_loc[type_idx] = {}
            for num_neighbors in range(self.max_neighbors):
                self.type_to_loc[type_idx][num_neighbors] = []
        
        # -1 entry indicates no cell
        self.locs = np.copy(pop_mat)
        self.xmax = np.shape(pop_mat)[1]
        self.ymax = np.shape(pop_mat)[0]
        self.types = np.zeros((self.num_types, self.max_neighbors))
        self.total_cells = np.sum(self.locs >= 0)
        for i in range(self.ymax):
            for j in range(self.xmax):
                assert self.locs[i,j] < self.num_types, "input type index too high"
                if self.locs[i,j] >= 0:
                    occ = self.get_num_empties((i,j))
                    self.types[self.locs[i,j], occ] += 1
                    self.type_to_loc[self.locs[i,j]][occ].append((i,j))
        self.update_total_rate()
        self.early_term = False
    
    def get_next_event_time(self):
        # DOES NOT MUTATE SELF
        return(np.random.exponential(1/self.total_rate))
    
    def advance(self):
        # DOES MUTATE SELF
        # NEEDS TO UPDATE: total_cells, total_rate, locs, types, type_to_loc
        birth_mat = self.calc_birth_mat()
        death_mat = self.calc_death_mat()
        mut_mat = self.calc_mut_mat()
        denom = np.sum(birth_mat) + np.sum(death_mat) + np.sum(mut_mat)
        p_birth = np.sum(birth_mat)/denom
        p_death = p_birth + np.sum(death_mat)/denom
        unif_sampled = np.random.uniform()
        if unif_sampled < p_birth:
            reproducer = np.random.choice(birth_mat.size, p=birth_mat.flatten()/np.sum(birth_mat))
            rep_type = reproducer // self.max_neighbors
            rep_occ = reproducer % self.max_neighbors
            to_choose = self.type_to_loc[rep_type][rep_occ]
            rep_loc = to_choose[np.random.choice(len(to_choose))]
            to_choose = self.get_adj_empties(rep_loc)
            new_loc = to_choose[np.random.choice(len(to_choose))]
            self.add_cell(new_loc, rep_type)
        elif unif_sampled < p_death:
            dead = np.random.choice(death_mat.size, p=death_mat.flatten()/np.sum(death_mat))
            dead_type = dead // self.max_neighbors
            dead_occ = dead % self.max_neighbors
            to_choose = self.type_to_loc[dead_type][dead_occ]
            dead_loc = to_choose[np.random.choice(len(to_choose))]
            self.remove_cell(dead_loc)
        else:
            mutated = np.random.choice(mut_mat.size, p=mut_mat.flatten()/np.sum(mut_mat))
            mut_type = mutated // self.max_neighbors
            mut_occ = mutated % self.max_neighbors
            to_choose = self.type_to_loc[mut_type][mut_occ]
            mut_loc = to_choose[np.random.choice(len(to_choose))]
            
            new_type = np.random.choice(self.num_types, p=self.mu[mut_type,:].flatten()/np.sum(self.mu[mut_type,:]))
            self.remove_cell(mut_loc)
            self.add_cell(mut_loc, new_type)
    
    def simulate(self, timespan, stop_conditions={}):
        curr_time = 0
        while True:
            for condition in stop_conditions.keys():
                if np.sum(self.types, axis=1)[condition] >= stop_conditions[condition]:
                    self.stop_time = curr_time
                    self.early_term = True
                    return()
            next_time = curr_time + self.get_next_event_time()
            if next_time > timespan or self.total_cells == 0:
                self.stop_time = curr_time
                return()
            self.advance()
            curr_time = next_time
    
    def get_pop_state(self):
        return(np.copy(self.locs))
    
    def plot_pop_state(self):
        to_plot = np.copy(self.locs)
        #to_plot[to_plot==-1] = np.nan
        sns.heatmap(to_plot)
        plt.show()