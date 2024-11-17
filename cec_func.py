import numpy as np
import casadi as ca

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

# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()

# Class for CECC 
class CEC_problem():
    def __init__(self, time_horizon = 10, obs = False):
        self.time_horizon = time_horizon
        self.Q = 1.1*ca.MX.eye(2)
        self.R = 0.1*ca.MX.eye(2)
        self.q = 1
        self.gamma = 1
        self.obstacle = obs
        if(self.obstacle):
            self.time_horizon = time_horizon
            self.Q = 0.7*ca.MX.eye(2)
            self.R = 0.6*ca.MX.eye(2)
            self.q = 1
            self.gamma = 0.85
  
    def car_error_next_state(self, curr_iter, curr_state, control):
        r_t = np.array(lissajous(curr_iter))
        r_t_1 = np.array(lissajous(curr_iter + 1))
        r_t_error = r_t - r_t_1
        theta = curr_state[2] + r_t[2]
        rot_3d_z = ca.vertcat( ca.hcat([ca.cos(theta), 0]), ca.hcat([ca.sin(theta), 0]), ca.hcat([0,1]))
        f = rot_3d_z @ control
        next_error_state = curr_state + time_step*f + r_t_error
        next_error_state[2] = self.wrap_angle(next_error_state[2])
        return next_error_state
    
    def wrap_angle(self,angle):
        wrapped_angle = ca.mod((angle + ca.pi), (2 * ca.pi)) - ca.pi
        return wrapped_angle

    def cec_optim(self, curr_state, curr_ref, curr_iter):
        error = curr_state - curr_ref

        opti = ca.Opti()
        e = opti.variable(3,self.time_horizon + 1)
        u = opti.variable(2,self.time_horizon)

        obj = 0
        for i in range(self.time_horizon):
            r_t = lissajous(curr_iter + i)
            state_x = e[0,i] + r_t[0]
            state_y = e[1,i] + r_t[1]
            state_z = e[2,i] + r_t[2]

            opti.subject_to(state_x<=3)
            opti.subject_to(state_x>=-3)
            opti.subject_to(state_y<=3)
            opti.subject_to(state_y>=-3)
            # opti.subject_to(state_z<=ca.pi)
            # opti.subject_to(state_z>=-ca.pi)

            opti.subject_to(e[:,i+1]==self.car_error_next_state(curr_iter + i, e[:,i], u[:,i]))
            
            obj = obj + self.gamma**i * (e[0:2,i].T@self.Q@e[0:2,i] + self.q*(1-ca.cos(e[2,i]))**2 + u[:,i].T@self.R@u[:,i])

            if(self.obstacle):
                opti.subject_to((state_x + 2)**2 + (state_y + 2)**2 >= 0.51**2)
                opti.subject_to((state_x - 1)**2 + (state_y - 2)**2 >= 0.51**2)

        obj = obj + self.gamma**self.time_horizon * (e[0:2,-1].T@self.Q@e[0:2,-1] + self.q*(1-ca.cos(e[2,-1]))**2)# + u[:,-1].T@self.R@u[:,-1])


        
        opti.subject_to(u[0,:]<=1)
        opti.subject_to(u[0,:]>=0)
        opti.subject_to(u[1,:]<=1)
        opti.subject_to(u[1,:]>=-1)
        opti.subject_to(e[:,0] == error)

        opti.minimize(obj)
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt', opts)
        sol = opti.solve()

        u_results = sol.value(u)
        return u_results[:,0]
    
    # First attempt dummy ignore
    # def cec_optim_try(self, curr_state, curr_ref, curr_iter):
    #     opti = ca.Opti()
    #     e = opti.variable(3,self.time_horizon + 1)
    #     u = opti.variable(2,self.time_horizon)
    #     Obs1 = ca.MX([-2, -2])
    #     Obs2 = ca.MX([1, 2])

    #     error = curr_state - curr_ref
    #     opti.subject_to(e[:,0]==error)
    #     opti.subject_to(u[0,:]<=1)
    #     opti.subject_to(u[0,:]>=0)
    #     opti.subject_to(u[1,:]<=1)
    #     opti.subject_to(u[1,:]>=-1)

    #     obj = 0
    #     for i in range(self.time_horizon-1):
    #         # using Q, q and R as identity 
    #         obj = obj + e[0,i]**2 + e[1,i]**2 + (1 - ca.cos(e[2,i]))**2 + u[0,i]**2 + u[1,i]**2

    #         state = e[:,i] + lissajous(curr_iter + i)
    #         # opti.subject_to(state[0]<=3)
    #         # opti.subject_to(state[0]>=-3)
    #         # opti.subject_to(state[1]<=3)
    #         # opti.subject_to(state[1]>=-3)
    #         # opti.subject_to(state[2]<=ca.pi)
    #         # opti.subject_to(state[2]>=-ca.pi)

    #         r_t = ca.MX(lissajous(curr_iter + i))
    #         r_t_1 = ca.MX(lissajous(curr_iter + i + 1))
    #         next_e_x = e[0,i] + r_t[0] - r_t_1[0] + u[0,i]*ca.cos(e[2,i] + r_t[2])*time_step
    #         next_e_y = e[1,i] + r_t[1] - r_t_1[1] + u[0,i]*ca.sin(e[2,i] + r_t[2])*time_step
    #         next_e_t = e[2,i] + r_t[2] - r_t_1[2] + u[1,i]*time_step

    #         opti.subject_to(next_e_x + r_t_1[0]<=3)
    #         opti.subject_to(next_e_x + r_t_1[0]>=-3)
    #         opti.subject_to(next_e_y + r_t_1[1]<=3)
    #         opti.subject_to(next_e_y + r_t_1[1]>=-3)
    #         # opti.subject_to(next_e_t + r_t_1[2]<=ca.pi)
    #         # opti.subject_to(next_e_t + r_t_1[2]>=-ca.pi)

            
    #         opti.subject_to(e[0,i+1]==next_e_x)
    #         opti.subject_to(e[1,i+1]==next_e_y)
    #         opti.subject_to(e[2,i+1]==next_e_t)

    #         # distance1 = ca.norm_2(state[0:2,:] - Obs1)
    #         # distance2 = ca.norm_2(state[0:2,:] - Obs2)
    #         # opti.subject_to(distance1 <= 0.5)
    #         # opti.subject_to(distance2 <= 0.5)

    #     opti.minimize(obj)
    #     opts = {"ipopt.print_level": 0, "print_time": 0}
    #     opti.solver('ipopt', opts)
    #     sol = opti.solve()

    #     # # Print the solution
    #     # print("Solution:")
    #     # print("x = ", sol.value(u))
    #     # print("Objective = ", sol.value(obj))
    #     u_results = sol.value(u)
    #     return u_results[:,0]
