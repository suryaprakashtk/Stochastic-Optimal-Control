from time import time
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import multivariate_normal

#  Given functions
# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time


# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1


# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]


class GPI(object):
    def __init__(self, obstacles):
        self.time_period = 100
        self.current_position = np.array([x_init, y_init, theta_init])
        self.dis_resolution = 7
        self.ang_resolution = 5
        self.linvel_resolution = 10
        self.angvel_resolution = 10
        self.state_space = None
        self.control_space = None
        self.obstacles = obstacles
        self.value_function = None
        self.policy = None
        self.stage_cost = None
        self.discount = 0.99  
        self.KD = None
        self.set_ini_control = 50
        self.counter = 0


    # Performs value iteration with only num_iter itertaions
    def iteration(self, curr_position, curr_ref_pos, curr_time, num_iter = 10):
        ind = curr_time%self.time_period
        curr_error = curr_position - curr_ref_pos
        distances, indices = self.KD.query(curr_error, k=1)
        curr_state = self.state_space[indices]
        
        # motion_model_mat = self.transition[:,:,:,ind]
        motion_model_mat = self.get_stm(ind)
        pi = np.zeros((num_iter+1,self.state_space.shape[0]),dtype='int')
        pi[0,:] = self.set_ini_control
        V = np.zeros((num_iter+1,self.state_space.shape[0]))
        t1 = time()
        for k in range(num_iter):
            Q = self.stage_cost[:,:] + self.discount * np.sum(motion_model_mat[:,:,:] * V[k,None,None,:], axis=2) # num_ntrm x nA
            pi[k+1,:] = np.argmin(Q, axis=1)    
            V[k+1,:] = np.min(Q,axis=1)
        t2 = time() - t1
        print("VI Computation completed    " + str(curr_time) + "  took  " + str(t2))
        result = self.control_space[pi[-1,indices]]
        
        return result
    
    # State transition matrix construction for a given time index.
    def get_stm(self,curr_iter):
        t1 = time()
        transition = np.zeros(shape=(self.state_space.shape[0],self.control_space.shape[0],self.state_space.shape[0]),dtype=float)
        next_state_exact = np.zeros(shape=(self.state_space.shape[0],self.state_space.shape[1]),dtype=float)
        for j in range(self.control_space.shape[0]):
            for k in range(self.state_space.shape[0]):
                next_state_exact[k,:] = self.car_error_next_state(curr_iter,self.state_space[k],self.control_space[j])
            next_state_exact_valid_ind = self.collisionfree(next_state_exact,curr_iter)
            next_state_exact_valid = next_state_exact[next_state_exact_valid_ind]
            next_state_prob = self.apply_guass_dist(next_state_exact_valid)
            transition[:,j,next_state_exact_valid_ind] = next_state_prob
        t2 = time() - t1
        print("state transition matrix retrival completed     " + str(curr_iter) + "  took  " + str(t2))
        return transition

    # error state motion model
    def car_error_next_state(self, curr_iter, curr_state, control):
        r_t = np.array(lissajous(curr_iter))
        r_t_1 = np.array(lissajous(curr_iter + 1))
        r_t_error = r_t - r_t_1
        theta = curr_state[2] + r_t[2]
        rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
        f = rot_3d_z @ control
        next_error_state = curr_state + time_step*f + r_t_error
        # angle wrap
        next_error_state[2] = (next_error_state[2] + np.pi)%(2 * np.pi) - np.pi
        return next_error_state
    
    # Precomputing all state tranistion matrix. But this is taking a lot of time and system is hanging.
    def precompute_stm(self):
        t1 = time()
        print("precomputing state transition matrix")
        self.transition = np.zeros(shape=(self.state_space.shape[0],self.control_space.shape[0],self.state_space.shape[0],self.time_period),dtype=float)
        next_state_exact = np.zeros(shape=(self.state_space.shape[0],self.state_space.shape[1]),dtype=float)
        for i in range(self.time_period):
            for j in range(self.control_space.shape[0]):
                for k in range(self.state_space.shape[0]):
                    next_state_exact[k,:] = self.car_error_next_state(i,self.state_space[k],self.control_space[j])
                next_state_exact_valid_ind = self.collisionfree(next_state_exact,i)
                next_state_exact_valid = next_state_exact[next_state_exact_valid_ind]
                next_state_prob = self.apply_guass_dist(next_state_exact_valid)
                self.transition[:,j,next_state_exact_valid_ind,i] = next_state_prob
            print(i)
        t2 = time() - t1
        print("precomputing state transition matrix completed took   " + str(t2))
        np.save('transition.npy', self.transition)
        print("transition saved")
        return
    
    # Takes all the next states and palces a guassian distrubution with those mean and evaluates the likelihood of all the states
    def apply_guass_dist(self,next_state):
        probability = np.zeros(shape=(next_state.shape[0],self.state_space.shape[0]),dtype=float)
        covariance = np.eye(2)
        for i in range(next_state.shape[0]):
            probability[i,:] = multivariate_normal.pdf(self.state_space[:,0:2], mean=next_state[i,0:2], cov=covariance)
        
        probability = probability.T
        row_sums = np.sum(probability, axis=1)
        normalized_prob = probability / row_sums[:, np.newaxis]
        return normalized_prob

    # Cheks for Collison
    def collisionfree(self,next_state,time_ind):
        # returns true if its collision free
        r_t_1 = np.array(lissajous(time_ind + 1))
        point = next_state + r_t_1
        dist_obs1 = np.linalg.norm(self.obstacles[0,0:2] - point[:,0:2], axis=1)
        dist_obs2 = np.linalg.norm(self.obstacles[1,0:2] - point[:,0:2], axis=1)
        radius1 = self.obstacles[0,2]
        radius2 = self.obstacles[1,2]
        valid_obs1 = dist_obs1 > radius1
        valid_obs2 = dist_obs2 > radius2
        within_range_disp_x = np.logical_and(point[:,0] >= -3, point[:,0] <= 3)
        within_range_disp_y = np.logical_and(point[:,1] >= -3, point[:,1] <= 3)
        within_range_theta = np.logical_and(point[:,2] >= -np.pi, point[:,2] <= np.pi)
        final = [valid_obs1, valid_obs2, within_range_disp_x, within_range_disp_y, within_range_theta]
        result = np.all(final, axis=0)
        return result
        
    def create_space(self):
        # state space
        result = []
        dis_x = np.linspace(-1, 1, self.dis_resolution)
        dis_y = np.linspace(-1, 1, self.dis_resolution)
        dis_theta = np.linspace(np.pi, -np.pi, self.ang_resolution)
        for j in range(dis_theta.shape[0]):
                for k in range(dis_y.shape[0]):
                    for l in range(dis_x.shape[0]):
                        result.append([dis_x[l],dis_y[k],dis_theta[j]])
        self.state_space = np.array(result)
        
        # Control space
        result = []
        lin_v = np.linspace(v_min, v_max, self.linvel_resolution)
        ang_v = np.linspace(w_min, w_max, self.angvel_resolution)
        for i in range(ang_v.shape[0]):
            for j in range(lin_v.shape[0]):
                result.append([lin_v[j],ang_v[i]])

        self.control_space = np.array(result)

        # Precompute stage cost
        self.stage_cost = np.zeros(shape=(self.state_space.shape[0],self.control_space.shape[0]),dtype=float)
        for j in range(self.control_space.shape[0]):
            for i in range(self.state_space.shape[0]):
                self.stage_cost[i,j] = self.state_space[i,0]**2 + self.state_space[i,1]**2 + self.control_space[j,0]**2 + self.control_space[j,1]**2 + (1-np.cos(self.state_space[i,2]))**2
        self.KD = KDTree(self.state_space)

        return
    
    # Generates all reference trajectory for the entire time period.
    def generate_ref_traj(self):
        result = []
        for i in range(self.time_period):
            result.append(lissajous(i))    
        return np.array(result)
