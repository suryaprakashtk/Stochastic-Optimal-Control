from time import time
import numpy as np
from utils import visualize
from cec_func import *
from gpi import *


if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0

    # Uncomment the line to run CEC horizon or GPI
    cec_horizon = CEC_problem(time_horizon = 15, obs=True)
    # GPI is incomplete
    # gpi_contrroller = GPI(obstacles)
    # gpi_contrroller.create_space()
    # gpi_contrroller.precompute_stm()

    # Main loop
    while (cur_iter * time_step < sim_time):
        
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        # Uncomment the controller corresponding to CEC or GPI
        # control = simple_controller(cur_state, cur_ref)
        control = cec_horizon.cec_optim(cur_state, cur_ref, cur_iter)
        # control = gpi_contrroller.iteration(cur_state, cur_ref, cur_iter,100)
        print(str(cur_iter) + "  [v,w]", control)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)

        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        # print(cur_iter)
        # print(t2-t1)
        times.append(t2-t1)
        error = error + np.linalg.norm(cur_state - cur_ref)
        temp = cur_state - cur_ref
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)

