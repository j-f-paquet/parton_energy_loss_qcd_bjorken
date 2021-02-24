import numpy as np
import scipy
import scipy.interpolate
import scipy.integrate
import scipy.optimize

from temperature_profile import brick_profile, Bjorken_hydro_profile
from parton_emission_rates import energy_loss_rates

hbarc=0.1973

##############################################
############## Parton evolution ##############
##############################################

# Input: 
# - initial conditions
# - initial time
# Functions:
# - evolve_to_max_time
# - evolve_to_min_temperature


class parton_evolution_solver_rk45:

    def __init__(self, initial_condition_fct, tau0, T_profile, energy_loss_rate, num_p=20, pmin=1, pmax=20):
        self.init_cond_fct = initial_condition_fct
        self.tau0=tau0
        # Momentum grid
        self.p_list=np.linspace(pmin,pmax,num_p)
        self.P_g_init=initial_condition_fct(self.p_list)
        self.T_profile=T_profile
        self.energy_loss_rate=energy_loss_rate
        self.num_p=num_p
        self.pmin=pmin
        self.pmax=pmax

    def rhs(self,tau,P_g):

        rhs_val=np.zeros_like(P_g)

        log_P_g_prev=np.log(P_g)

        T=self.T_profile.get_T(tau)

        dGamma_domega=self.energy_loss_rate.dGamma_domega
        pmin=self.pmin
        pmax=self.pmax

        for ip, p in enumerate(self.p_list):

            # Get interpolator for log_P_g_prev(p)
            interp_log_P_g_prev=scipy.interpolate.interp1d(self.p_list,log_P_g_prev,  kind='linear', fill_value='extrapolate') #fill_value=-1000, bounds_error=False)

            def P_g_prev(p):
                return np.exp(interp_log_P_g_prev(p))

            def integrand(omega):

                return P_g_prev(p+omega)*dGamma_domega(p+omega, omega,T)-P_g_prev(p)*dGamma_domega(p, omega,T)

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

            rhs_val[ip]=res

        return rhs_val

    # Returns an interpolation function with the results
    def evolve_to_max_time(self, tau_max):

            tau=self.tau0

            rk_obj=scipy.integrate.RK45(lambda t, y: self.rhs(t,y), self.tau0, self.P_g_init, tau_max)

            while (rk_obj.t < tau_max):
                rk_obj.step()

            res=rk_obj.y;

            # Make interpolation function with the results
            interp_log_P_g=scipy.interpolate.interp1d(self.p_list,np.log(res),  kind='cubic', bounds_error=True)
                
            return lambda p: np.exp(interp_log_P_g(p))


    # Returns an interpolation function with the results
    def evolve_to_min_temperature(self, T_min_in_GeV):

            # Find tau_max corresponding to T_min_in_GeV
            tau_max=scipy.optimize.brentq(lambda tau : self.T_profile.get_T(tau)-T_min_in_GeV,self.T_profile.tau0,100)

            return self.evolve_to_max_time(tau_max)

            #tau_max=1000.

            #rk_obj=scipy.integrate.RK45(lambda t, y: self.rhs(t,y), self.tau0, self.P_g_init, tau_max)

            #T_in_GeV=T_profile.get_T(self.tau0)

            #while (rk_obj.t < tau_max):
            #    rk_obj.step()
            #    if (T_profile.get_T(rk_obj.t)<T_min_in_GeV):
            #        break

            #res=rk_obj.y;

            ## Make interpolation function with the results
            #interp_log_P_g=scipy.interpolate.interp1d(self.p_list,np.log(res),  kind='cubic', bounds_error=True)
            #    
            #return lambda p: np.exp(interp_log_P_g(p))


## Temperature profile
#T0_in_GeV=.300
#tau0=0.4
#
##T_brick_fct=brick_profile(T0_in_GeV=T0_in_GeV)
#T_profile=Bjorken_hydro_profile(T0_in_GeV=T0_in_GeV, tau0=tau0)
#
#
## Parton energy loss rates
##g_s=2
##alpha_s=g_s**2/(4.*numpy.pi)
#alpha_s=0.1
#g_s=np.sqrt(4*np.pi*alpha_s)
#N_f=0
## Rate are functions of: p, omega, T
#energy_loss_rate=energy_loss_rates(alpha_s = alpha_s, N_f=N_f)
#
## Initial conditions 
#def P_g_tau0(p):
#
#    p0=1.75
#    
#    return np.power(p0*p0+p*p,-5.)
#
#parton_evolution_solver=parton_evolution_solver_euler(initial_condition_fct=P_g_tau0, tau0=tau0, T_profile=T_profile, energy_loss_rate=energy_loss_rate, num_p=20, pmin=1, pmax=20)
#
## Solve until out of the medium
#T_min_in_GeV=.29
#dtau=0.005
#res1=parton_evolution_solver.evolve_to_min_temperature(dtau=dtau, T_min_in_GeV=T_min_in_GeV, use_adaptive_timestep=False)
#res2=parton_evolution_solver.evolve_to_min_temperature(dtau=dtau/2, T_min_in_GeV=T_min_in_GeV, use_adaptive_timestep=False)
#
#p_list=np.linspace(5.,10,10)
#print(res1(p_list)/res2(p_list))
#
#
#
##T_in_GeV=T_ns_fct.get_T(tau)
##log_P_g_prev=log_P_g_init
##while (T_in_GeV>T_min_in_GeV):
##
##    # Compute spectra at next timestep
##    log_P_g=np.log(Pg_update(log_P_g_prev,T_in_GeV,dtau))
##
##    # Adaptive dtau, such that d(temperature) remains roughly the same
##    dtau*=(tau+dtau)/tau
##    #dtau*=np.power((tau+dtau)/tau,1+1./3.)
##    # Next timestep
##    tau+=dtau 
##    T_in_GeV=T_ns_fct.get_T(tau)
##    log_P_g_prev=log_P_g
##
##    print("RAA at tau=",tau," fm (T=",T_in_GeV," GeV)")  
##    print(np.exp(log_P_g)/np.exp(log_P_g_init))
