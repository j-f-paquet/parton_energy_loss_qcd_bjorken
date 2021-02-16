import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator
import matplotlib.ticker as mtick
import scipy #.optimize
import scipy.interpolate
import scipy.integrate

from temperature_profile import init_brick_profile, init_Bjorken_hydro_profile
from parton_emission_rates import init_energy_loss_rates


hbarc=0.1973


#################################################
############## Temperature profile ##############
##################################################

T0_in_GeV=.300
tau0=0.4

#T_brick_fct=init_brick_profile(T0_in_GeV)
T_ns_fct=init_Bjorken_hydro_profile(tau0, T0_in_GeV)


######################################################
############## Parton energy loss rates ##############
######################################################

#g_s=2
#alpha_s=g_s**2/(4.*numpy.pi)

alpha_s=0.1
g_s=np.sqrt(4*np.pi*alpha_s)

N_f=0
N_c=3

# Rate are functions of: p, omega, T
energy_loss_rate=init_energy_loss_rates(alpha_s, N_f, N_c)


##############################################
############## Parton evolution ##############
##############################################

# Initial conditions 
def P_g_t0(p):

    p0=1.75
    
    return np.power(p0*p0+p*p,-5.)

# Momentum grid
n_p=20
pmin=1
pmax=20
p_list=np.linspace(pmin,pmax,n_p)

log_P_g_init=np.log(P_g_t0(p_list))

def Pg_update(log_P_g_prev,T,dt):

    P_g=np.zeros(n_p)

    for ip, p in enumerate(p_list):

        # Get interpolator for log_P_g_prev(p)
        interp_log_P_g_prev=scipy.interpolate.interp1d(p_list,log_P_g_prev,  kind='linear', fill_value='extrapolate') #fill_value=-1000, bounds_error=False)

        def P_g_prev(p):
            return np.exp(interp_log_P_g_prev(p))

        def integrand(omega):

            return P_g_prev(p+omega)*energy_loss_rate(p+omega, omega,T)-P_g_prev(p)*energy_loss_rate(p, omega,T)

        def integrand_middle(p,u):
            return integrand(p*(1-u))+integrand(p*(1+u))

        vec_integrand = np.vectorize(integrand)

        npts=1000
        epsabs=1e-30
        epsrel=1e-3

        delta=0.2
        if (p*(1-delta)<pmin)or(p*(1+delta)>pmax):
            res_quad2a=scipy.integrate.quad(vec_integrand, pmin, p, limit=npts, epsabs=epsabs, epsrel=epsrel)
            res_quad2c=scipy.integrate.quad(vec_integrand, p, pmax, limit=npts, epsabs=epsabs, epsrel=epsrel)
            res=res_quad2a[0]+res_quad2c[0]
        else:
            res_quad2a=scipy.integrate.quad(vec_integrand, pmin, p*(1-delta), limit=npts, epsabs=epsabs, epsrel=epsrel)
            res_quad2b=scipy.integrate.quad(lambda u, p=p: p*integrand_middle(p,u), 0, delta, limit=npts, epsabs=epsabs, epsrel=epsrel)
            res_quad2c=scipy.integrate.quad(vec_integrand, p*(1+delta), pmax, limit=npts, epsabs=epsabs, epsrel=epsrel)
            res=res_quad2a[0]+res_quad2b[0]+res_quad2c[0]

        P_g[ip]=P_g_prev(p)+dt*res

    return P_g


#print(Pg_update(log_P_g_init,.3,.1))
#exit(1)

# Solve until out of the medium
T_min_in_GeV=.15
taumin=.4
tau=taumin
dtau=0.005
T_in_GeV=T_ns_fct(tau)
log_P_g_prev=log_P_g_init
while (T_in_GeV>T_min_in_GeV):

    # Compute spectra at next timestep
    log_P_g=np.log(Pg_update(log_P_g_prev,T_in_GeV,dtau))

    # Adaptive dtau, such that d(temperature) remains roughly the same
    dtau*=(tau+dtau)/tau
    #dtau*=np.power((tau+dtau)/tau,1+1./3.)
    # Next timestep
    tau+=dtau 
    T_in_GeV=T_ns_fct(tau)
    log_P_g_prev=log_P_g

    print("RAA at tau=",tau," fm (T=",T_in_GeV," GeV)")  
    print(np.exp(log_P_g)/np.exp(log_P_g_init))
