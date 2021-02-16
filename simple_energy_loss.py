import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator
import matplotlib.ticker as mtick
import scipy #.optimize
import scipy.interpolate
import scipy.integrate
#import scipy.integrate.simpson

from eos import cs2_qcd_fct
from style import font_choice, linewidth_choice
from bjorken_ns_solver import init_bjorken_ns_solver, approx_effective_viscosity_ns, better_effective_viscosity_ns

hbarc=0.1973

# Speed of sound (EOS)
def cs2_fct(T_in_fm):
    return cs2_qcd_fct(T_in_fm)
    #return 1./3

# Viscosities
def eta_over_s_fct(T_in_fm):

    return 0.1

#    T_in_GeV=T_in_fm*hbarc;
#
#    T_kink_in_GeV =0.17
#    low_T_slope=5
#    high_T_slope=0.5
#    eta_over_s_at_kink=0.04
#
#    if (T_in_GeV<T_kink_in_GeV):
#        eta_over_s=eta_over_s_at_kink + low_T_slope*(T_kink_in_GeV - T_in_GeV);
#    else:
#        eta_over_s=eta_over_s_at_kink + high_T_slope*(T_in_GeV - T_kink_in_GeV);
#
#    return eta_over_s;


def combined_visc_fct(T_in_fm):
    eta_over_s=eta_over_s_fct(T_in_fm)
    zeta_over_s=0.0 #zeta_over_s_fct(T_in_fm)
    return (4./3.*eta_over_s+zeta_over_s)



###############################################################################
########## Curves are plotted for the following initial temperatures ##########
###############################################################################

# Other parameters
T0_in_fm=0.3/hbarc
tau0=0.2
Tf_in_fm=0.15/hbarc

# Get Navier-Stokes solution
ns_sol=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct, combined_visc_fct_param=combined_visc_fct)



#################################################
############## Temperature profile ##############
##################################################

#tau_plot=10.
#
#plt.figure()
#plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
##plt.yticks([])
##plt.axes().yaxis.set_minor_formatter(NullFormatter())
##plt.axes().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
#plt.gca().yaxis.set_ticks_position('both')
#plt.axes().yaxis.set_minor_locator(AutoMinorLocator())
#
#plt.xlim(tau0,tau_plot)
#plt.ylim(.1,1.0*T0_in_fm*hbarc)
##plt.xscale('log')
##plt.yscale('log')
##plt.yticks(np.arange(0.,0.3,.05))
#
#plt.xlabel(r'$\tau$ (fm)')
#plt.ylabel(r"$T$ (GeV)")
#
## Plot the exact solution
#tau_range=np.arange(tau0,tau_plot,(tau_plot-tau0)/100)
#T_from_exact= hbarc*np.array(list(map(ns_sol.integrate,tau_range)))
#plt.plot(tau_range, T_from_exact,"-",color='red',label="", linewidth=3)
#
#plt.legend()
#plt.tight_layout()
#plt.savefig("T_profile.pdf")
#plt.show()
#
#exit(1)


######################################################
############## Parton energy loss rates ##############
######################################################

#g_s=2
#alpha_s=g_s**2/(4.*numpy.pi)

alpha_s=0.1
g_s=np.sqrt(4*np.pi*alpha_s)

N_f=0

N_c=3
C_A=N_c

def m_D(T):

    return g_s*T*np.sqrt((N_c/3.+N_f/6))

# For gluons
def qhat_eff(omega,T):

    # Solution of qhat_eff = q_soft^22(mu_T) with mu_T^2=C0 Sqrt(2 omega qhat_eff)
    # with C_0=2 e^{2-\gamma +\frac{\pi }{4}}
    # and q_soft^22(mu_T)=norm Log(mu_T^2/m_D^2)
    # with norm=alpha_s*C_A*T*m_D(T)**2

    norm=alpha_s*C_A*T*m_D(T)**2

    C_0=18.1983

    return -0.5*norm*scipy.special.lambertw(-m_D(T)**4/(norm*C_0**2*omega),k=-1)


#print(qhat_eff(1e-2*1000,.3))
# should be about 0.052 with N_f=0 and alpha_s=0.1

def dGamma_domega_inel(p, omega,T):

    if (p == 0)or(omega>=p):
        res=0.0

    else:

        z=omega/p

        #qhat_eff=qhat_eff(omega,T)
        qhat_eff_val=0.04 #qhat_eff(1e-2*p,T)
        #print(qhat_eff_val)

        res=alpha_s*N_c/(np.pi*p)*np.power(z*(1-z),-3./2.)*np.sqrt(qhat_eff_val/p)

    return res


#plt.figure()
#plt.axes().xaxis.set_minor_locator(AutoMinorLocator())
#plt.gca().yaxis.set_ticks_position('both')
#plt.axes().yaxis.set_minor_locator(AutoMinorLocator())
#
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(3e-4,5e-1)
#plt.ylim(1e-6,1e-1)
#
#plt.xlabel(r'$\omega/p$')
#plt.ylabel(r"$d\Gamma/d\omega$")
#
## Plot the exact solution
#z_range=np.array([10.**x for x in np.arange(-4,0,.1)])
#p=1000.
#T=0.3
#omega_range=z_range*p
#dGamma_list=[dGamma_domega_inel(p,omega,T) for omega in omega_range]
#
#plt.plot(z_range, dGamma_list,"-",color='red',label="", linewidth=3)
#
#plt.legend()
#plt.tight_layout()
##plt.savefig("T_profile.pdf")
#plt.show()



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

            return P_g_prev(p+omega)*dGamma_domega_inel(p+omega, omega,T)-P_g_prev(p)*dGamma_domega_inel(p, omega,T)

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
T_in_fm=ns_sol.integrate(tau)
T_in_GeV=T_in_fm*hbarc
log_P_g_prev=log_P_g_init
while (T_in_GeV>T_min_in_GeV):

    # Compute spectra at next timestep
    log_P_g=np.log(Pg_update(log_P_g_prev,T_in_GeV,dtau))

    # Adaptive dtau, such that d(temperature) remains roughly the same
    dtau*=(tau+dtau)/tau
    #dtau*=np.power((tau+dtau)/tau,1+1./3.)
    # Next timestep
    tau+=dtau 
    T_in_fm=ns_sol.integrate(tau)
    T_in_GeV=T_in_fm*hbarc
    log_P_g_prev=log_P_g

    print("RAA at tau=",tau," fm (T=",T_in_GeV," GeV)")  
    print(np.exp(log_P_g)/np.exp(log_P_g_init))
