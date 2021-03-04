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
from solver_rk8 import parton_evolution_solver_rk8

import time

hbarc=0.1973


#################################################
############## Temperature profile ##############
##################################################

T0_in_GeV=.300
tau0=0.4

#T_profile=brick_profile(T0_in_GeV=T0_in_GeV)
T_profile=Bjorken_hydro_profile(T0_in_GeV=T0_in_GeV, tau0=tau0)

######################################################
############## Parton energy loss rates ##############
######################################################

#g_s=2
#alpha_s=g_s**2/(4.*numpy.pi)
alpha_s=0.2
g_s=np.sqrt(4*np.pi*alpha_s)
N_f=0

scale_inel=.2 #np.inf
exponent_inel=-1
scale_el=.2
exponent_el=1

K_factor_fct_inel=lambda T, scale_inel=scale_inel, exponent_inel=exponent_inel : (1.+np.power(T/scale_inel,exponent_inel))
K_factor_fct_elastic=lambda T, scale_el=scale_el, exponent_el=exponent_el : (1.+np.power(T/scale_el,exponent_el))
energy_loss_rate=energy_loss_rates(alpha_s = alpha_s, N_f=N_f, K_factor_fct_inel=K_factor_fct_inel, K_factor_fct_elastic=K_factor_fct_elastic)

#######################################################
############## Parton energy loss solver ##############
#######################################################

# Initial conditions 
def P_g_tau0(p):

    p0=1.75
    
    return np.power(p0*p0+p*p,-5.)

# Initialize and use the solver
#parton_evolution_solver=parton_evolution_solver_euler(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=20, pmin=1, pmax=20)
#parton_evolution_solver2=parton_evolution_solver_euler(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=40, pmin=1, pmax=20)
#T_min_in_GeV=.25
#dtau=0.005
#P_final_fct1=parton_evolution_solver.evolve_to_min_temperature(dtau=dtau, T_min_in_GeV=T_min_in_GeV, use_adaptive_timestep=False)
#P_final_fct2=parton_evolution_solver2.evolve_to_min_temperature(dtau=dtau, T_min_in_GeV=T_min_in_GeV, use_adaptive_timestep=False)
##P_final2_fct=parton_evolution_solver.evolve_to_min_temperature(dtau=dtau/2, T_min_in_GeV=T_min_in_GeV, use_adaptive_timestep=False)
#
## Print and plot
#
## Compute some "RAA"-equivalent
#p_list=np.linspace(1.,20,10) # Define some p_T bins
#P_initial=P_g_tau0(p_list)
#P_final1=P_final_fct1(p_list)
#P_final2=P_final_fct2(p_list)
#print('mock R_AA')
#print('using 20 points in momentum to solve the parton evolution')
#print(P_final1/P_initial)
#print('using 40 points in momentum to solve the parton evolution')
#print(P_final2/P_initial)

T_min_in_GeV=.15
dtau=0.05
dtau_adaptive=0.01

# Initialize and use the solver
#parton_evolution_solver=parton_evolution_solver_euler(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=20, pmin=1, pmax=20)
parton_evolution_solver_rk=parton_evolution_solver_rk(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=20, pmin=1, pmax=20)
parton_evolution_solver_rk8=parton_evolution_solver_rk8(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=20, pmin=1, pmax=20)

tic = time.perf_counter()
P_final_fct1=parton_evolution_solver_rk.evolve_to_min_temperature(T_min_in_GeV=T_min_in_GeV)
toc = time.perf_counter()
print(f"RK4/5: {toc - tic:0.4f} seconds")
#tic = time.perf_counter()
#P_final_fct2=parton_evolution_solver_rk8.evolve_to_min_temperature(T_min_in_GeV=T_min_in_GeV)
#toc = time.perf_counter()
#print(f"RK8: {toc - tic:0.4f} seconds")

# Compute some "RAA"-equivalent
p_list=np.linspace(1.,20,10) # Define some p_T bins
P_initial=P_g_tau0(p_list)
P_final1=P_final_fct1(p_list)
#P_final2=P_final_fct2(p_list)
print('mock R_AA')
print('using RK4/5 to solve the parton evolution')
print(P_final1/P_initial)
#print('using RK8 to solve the parton evolution')
#print(P_final2/P_initial)


##print(zip(p_list,P_final,P_final/P_initial))
##print([(p, P, R) for (p,P,R) in zip(p_list,P_final,P_final/P_initial)])
#
#p_list=np.linspace(1.,20,20) # Define some p_T bins
#P_initial=P_g_tau0(p_list)
#P_final=P_final_fct(p_list)
#print('mock R_AA again')
##print([(p, P, R) for (p,P,R) in zip(p_list,P_final,P_final/P_initial)])
#print(P_final/P_initial)
##print(P_final)
