import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator
import matplotlib.ticker as mtick
import scipy #.optimize
import scipy.interpolate
import scipy.integrate

from temperature_profile import brick_profile, Bjorken_hydro_profile
from parton_emission_rates import energy_loss_rates
from solver_euler import parton_evolution_solver_euler
from solver_rk import parton_evolution_solver_rk

import time
from joblib import Parallel, delayed

import signal

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

hbarc=0.1973

# Initial conditions
def P_g_tau0(p):
    p0=1.75
    return np.power(p0*p0+p*p,-5.)

def simulate(ii, params, p_min=1, p_max=20, num_p_bins=20):
    print(f'Working on design {ii+1}')
    start_time = time.time()

    signal.alarm(600) # time out after 10 minutes
    try:
        params=params.flatten()

        param_dict={
        'T0_in_GeV':0.3,
        'tau0':0.2,
        'T_final_in_GeV':0.15,
        'alpha_s':params[0],
        'N_f':0,
        'mD_factor':1.0,
        'exponent_inel':params[1],
        'exponent_el':params[2],
        'RAA_pT_binnings':np.linspace(p_min, p_max, num_p_bins),
        'scale_inel':params[3],
        'scale_el':params[4]
        }

        T0_in_GeV=param_dict['T0_in_GeV']
        tau0=param_dict['tau0']
        T_final_in_GeV=param_dict['T_final_in_GeV']
        alpha_s=param_dict['alpha_s']
        N_f=param_dict['N_f']
        mD_factor=param_dict['mD_factor']
        exponent_inel=param_dict['exponent_inel']
        exponent_el=param_dict['exponent_el']
        RAA_pT_binnings=param_dict['RAA_pT_binnings']
        # The new parameters
        scale_inel=param_dict['scale_inel']
        scale_el=param_dict['scale_el']

        #################################################
        ############## Temperature profile ##############
        ##################################################

        #T_profile=brick_profile(T0_in_GeV=T0_in_GeV)
        T_profile=Bjorken_hydro_profile(T0_in_GeV=T0_in_GeV, tau0=tau0)

        ######################################################
        ############## Parton energy loss rates ##############
        ######################################################


        K_factor_fct_inel=lambda T, scale_inel=scale_inel, exponent_inel=exponent_inel: (1.+np.power(T/scale_inel,exponent_inel))
        K_factor_fct_elastic=lambda T, scale_el=scale_el, exponent_el=exponent_el: (1.+np.power(T/scale_el,exponent_el))

        energy_loss_rate=energy_loss_rates(alpha_s = alpha_s, N_f=N_f, mD_factor=mD_factor, K_factor_fct_inel=K_factor_fct_inel, K_factor_fct_elastic=K_factor_fct_elastic)

        #######################################################
        ############## Parton energy loss solver ##############
        #######################################################


        # Initialize and use the solver
        num_p_solver=num_p_bins
        pmin_solver=p_min
        pmax_solver=p_max

        #parton_evolution_solver=parton_evolution_solver_euler(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=num_p_solver, pmin=pmin_solver, pmax=pmax_solver)
        #P_final_fct=parton_evolution_solver.evolve_to_min_temperature(dtau=dtau_adaptive, T_min_in_GeV=T_final_in_GeV, use_adaptive_timestep=True)

        parton_evolution_solver=parton_evolution_solver_rk(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=num_p_solver, pmin=pmin_solver, pmax=pmax_solver)
        P_final_fct=parton_evolution_solver.evolve_to_min_temperature(T_min_in_GeV=T_final_in_GeV)

        # Compute some "RAA"-equivalent
        P_initial=P_g_tau0(RAA_pT_binnings)
        P_final=P_final_fct(RAA_pT_binnings)
        result=P_final/P_initial
        end_time = time.time()
        delta_t = end_time - start_time

        if  delta_t > 60 :
            print(f'For model parameters {params} takes {delta_t} S')

        return result

    except TimeoutException: # what to do if it takes longer than 10 minutes
        print(f'For model parameters {params} takes more than 10 mins')
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        # Reset the alarm
        signal.alarm(0)


def run_simulation(design_matrix, p_min=1, p_max=20, num_p_bins=20):
    """Simulate energy loss of a jet for each design point

    parameters
    ----------
    see definition of "param_dict"
    p_min, p_max, num_p_bins : Fix the momentem range and number of bins for R_AA calculations

    Returns
    -------
    numpy array
        R_AA binned according to the momentem range speceifed at the input

    """
    par = True # Run in parallel
    # Evaluate simulation at validation points
    simulate_run = lambda ii, params: simulate(ii, params, p_min=p_min, p_max=p_max, num_p_bins=num_p_bins)
    if par:
        cores = 5
        observations = Parallel(n_jobs=cores)(delayed(simulate_run)(ii,params) for ii, params in enumerate(design_matrix))
    else:
        observations = [simulate_run(ii,params) for ii, params in enumerate(design_matrix)]

    observations=np.array(observations)
    return observations
