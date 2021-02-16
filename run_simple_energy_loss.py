import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, AutoMinorLocator
import matplotlib.ticker as mtick
import scipy #.optimize
import scipy.interpolate
import scipy.integrate

from eos import cs2_qcd_fct
#from style import font_choice, linewidth_choice
from bjorken_ns_solver import init_bjorken_ns_solver, approx_effective_viscosity_ns, better_effective_viscosity_ns

hbarc=0.1973

default_design = np.array([[0.3/hbarc, 0.2, 0.15/hbarc,0.1, 0, 3]])
def run_simulation(design_matrix = default_design):
    """Simulate energy loss of a jet for each design point

    parameters
    ----------
    design_matrix : numpy array
        M x N dimensional array with M design points and
        N number of model parameters for each design.
        N model parameters is as follows
        --------------------------------
        Initialization
            T0_in_fm (deafult value 0.3/hbarc)
            tau0 (deafult value 0.2)
            Tf_in_fm (deafult value 0.15/hbarc)

        Parton Energy loss rates
            alpha_s (deafult value 0.1)
            N_f (deafult value 0)
            N_c (deafult value 3)


    Returns
    -------
    numpy array
        M x P array of R_AA obserables for M design points
        and for P number of P_T bins.
        -----------------------------
        Currently
        number of p_T bins = 50
        p_min = 1
        p_max = 30

    """
    print(f'The shape of design matrix is {design_matrix.shape}\n')

    observations = []
    for ii, params in enumerate(design_matrix):
        print(f'Working on {ii}/{design_matrix.shape[0]} designs')
        #print(f'The shape of first parmeter row is {params.shape}\n')
        params=params.flatten()
        # Other parameters
        T0_in_fm=params[0]
        tau0=params[1]
        Tf_in_fm=params[2]

    # Get Navier-Stokes solution
        ns_sol=init_bjorken_ns_solver(T0_in_fm, tau0, cs2_fct, combined_visc_fct_param=combined_visc_fct)
        energy_loss_rate = energy_loss_rates(alpha_s = params[3], N_f = params[4], N_c = params[5])
        dGamma_domega_inel= energy_loss_rate.dGamma_domega_inel

    # Solve until out of the medium
        T_min_in_GeV=.150
        taumin=.4
        tau=taumin
        dtau=0.025
        T_in_fm=ns_sol.integrate(tau)
        T_in_GeV=T_in_fm*hbarc
        log_P_g_prev=log_P_g_init
        while (T_in_GeV>T_min_in_GeV):

            # Compute spectra at next timestep
            log_P_g=np.log(Pg_update(log_P_g_prev,T_in_GeV,dtau,dGamma_domega_inel))

            # Next timestep
            tau+=dtau
            T_in_fm=ns_sol.integrate(tau)
            T_in_GeV=T_in_fm*hbarc
            log_P_g_prev=log_P_g

            #print("RAA at tau=",tau," fm (T=",T_in_GeV," GeV)")
            #print(np.exp(log_P_g)/np.exp(log_P_g_init))# Solve until out of the medium
        observations.append(np.exp(log_P_g)/np.exp(log_P_g_init))

    result = np.array(observations)
    print(f'Shape of the result array is {result.shape}')
    return result



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




#################################################
############## Temperature profile ##############
##################################################
#
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


######################################################
############## Parton energy loss rates ##############
######################################################

#g_s=2
#alpha_s=g_s**2/(4.*numpy.pi)

class energy_loss_rates:

    def __init__(self, alpha_s, N_f, N_c):
        self.alpha_s = alpha_s
        self.g_s = np.sqrt(4*np.pi*alpha_s)
        self.N_f = N_f
        self.N_c = N_c
        self.C_A = N_c

    def m_D(self, T):
        return self.g_s*T*np.sqrt((self.N_c/3.+self.N_f/6))

    def qhat_eff(self, omega, T):

        # Solution of qhat_eff = q_soft^22(mu_T) with mu_T^2=C0 Sqrt(2 omega qhat_eff)
        # with C_0=2 e^{2-\gamma +\frac{\pi }{4}}
        # and q_soft^22(mu_T)=norm Log(mu_T^2/m_D^2)
        # with norm=alpha_s*C_A*T*m_D(T)**2

        norm=self.alpha_s*self.C_A*T*self.m_D(T)**2
        C_0=18.1983
        return -0.5*norm*scipy.special.lambertw(-self.m_D(T)**4/(norm*self.C_0**2*omega),k=-1)


    def dGamma_domega_inel(self, p, omega, T):

        if (p == 0)or(omega>=p):
            res=0.0

        else:

            z=omega/p

            #qhat_eff=self.qhat_eff(omega,T)
            qhat_eff_val=0.04 #qhat_eff(1e-2*p,T)
            #print(qhat_eff_val)

            res=self.alpha_s*self.N_c/(np.pi*p)*np.power(z*(1-z),-3./2.)*np.sqrt(qhat_eff_val/p)

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
n_p=50
pmin=1
pmax=30
p_list=np.linspace(pmin,pmax,n_p)

log_P_g_init=np.log(P_g_t0(p_list))


print(f'Momentem range is ({pmin},{pmax} GeV).\nNumber of bins {n_p}.')
#print(log_P_g_prev)
#print(np.exp(log_P_g_init))

def Pg_update(log_P_g_prev,T,dt,dGamma_domega_inel):

    P_g=np.zeros(n_p)

    for ip, p in enumerate(p_list):

        # Get interpolator for log_P_g_prev(p)
        interp_log_P_g_prev=scipy.interpolate.interp1d(p_list,log_P_g_prev, fill_value=-1000, bounds_error=False)

        def P_g_prev(p):
            return np.exp(interp_log_P_g_prev(p))

        def integrand(omega):

            return P_g_prev(p+omega)*dGamma_domega_inel(p+omega, omega,T)-P_g_prev(p)*dGamma_domega_inel(p, omega,T)
            #return -P_g_prev(p)*dGamma_domega_inel(p, omega,T)
            #return dGamma_domega_inel(p, omega,T)

        #print([integrand(omega) for omega in np.linspace(pmin,pmax,100)])

        res=scipy.integrate.quad(integrand, pmin, pmax, limit=10000, epsabs=1e-10, epsrel=1e-3)

        #print(res)

        #exit(1)

        P_g[ip]=P_g_prev(p)+dt*res[0]

    return P_g


#print("updated!")
#print(Pg_update(log_P_g_init,.3,.1))
