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
from solver_rk import parton_evolution_solver_rk45

hbarc=0.1973


def run_simulation(design_matrix):
    """Simulate energy loss of a jet for each design point

    parameters
    ----------
    see definition of "param_dict"

    Returns
    -------
    numpy array
        R_AA binned according to 'RAA_pT_binnings'

    """

    observations = []
    for ii, params in enumerate(design_matrix):
        print(f'Working on {ii}/{design_matrix.shape[0]} designs')
        params=params.flatten()

        param_dict={
        'T0_in_GeV':params[0],
        'tau0':params[1],
        'T_final_in_GeV':params[2],
        'alpha_s':params[3],
        'N_f':0, #params[4],
        'RAA_pT_binnings':np.linspace(1,20,20)
        }

        T0_in_GeV=param_dict['T0_in_GeV']
        tau0=param_dict['tau0']
        T_final_in_GeV=param_dict['T_final_in_GeV']
        alpha_s=param_dict['alpha_s']
        N_f=param_dict['N_f']
        RAA_pT_binnings=param_dict['RAA_pT_binnings']

        #################################################
        ############## Temperature profile ##############
        ##################################################

        #T_profile=brick_profile(T0_in_GeV=T0_in_GeV)
        T_profile=Bjorken_hydro_profile(T0_in_GeV=T0_in_GeV, tau0=tau0)

        ######################################################
        ############## Parton energy loss rates ##############
        ######################################################

        energy_loss_rate=energy_loss_rates(alpha_s = alpha_s, N_f=N_f)

        #######################################################
        ############## Parton energy loss solver ##############
        #######################################################

        # Initial conditions 
        def P_g_tau0(p):

            p0=1.75
            
            return np.power(p0*p0+p*p,-5.)


        # Initialize and use the solver
        num_p_solver=20
        pmin_solver=1
        pmax_solver=20
        #parton_evolution_solver=parton_evolution_solver_euler(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=num_p_solver, pmin=pmin_solver, pmax=pmax_solver)
        #P_final_fct=parton_evolution_solver.evolve_to_min_temperature(dtau=dtau_adaptive, T_min_in_GeV=T_final_in_GeV, use_adaptive_timestep=True)
        parton_evolution_solver=parton_evolution_solver_rk45(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=num_p_solver, pmin=pmin_solver, pmax=pmax_solver)
        P_final_fct=parton_evolution_solver.evolve_to_min_temperature(T_min_in_GeV=T_final_in_GeV)

        # Compute some "RAA"-equivalent
        P_initial=P_g_tau0(RAA_pT_binnings)
        P_final=P_final_fct(RAA_pT_binnings)
        result=P_final/P_initial
        observations.append(result)

    observations=np.array(observations)
    print(f'Shape of the result array is {observations.shape}')
    return observations

