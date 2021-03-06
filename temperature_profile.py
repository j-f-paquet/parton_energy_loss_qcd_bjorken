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
from bjorken_ns_solver import init_bjorken_ns_solver, approx_effective_viscosity_ns, better_effective_viscosity_ns

hbarc=0.1973

class brick_profile:

    def __init__(self, T0_in_GeV):
        self.T0_in_GeV = T0_in_GeV

    def get_T(self, tau):
        return self.T0_in_GeV

class Bjorken_hydro_profile:

    def __init__(self, T0_in_GeV, tau0):
        self.T0_in_GeV = T0_in_GeV
        self.tau0 = tau0
        T0_in_fm=T0_in_GeV/hbarc
        self.ns_sol = init_bjorken_ns_solver(T0_in_fm, self.tau0, self.cs2_fct, self.combined_visc_fct)
    # Speed of sound (EOS)
    def cs2_fct(self, T_in_fm):
        return cs2_qcd_fct(T_in_fm)
        #return 1./3

    # Viscosities
    def eta_over_s_fct(self, T_in_fm):

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


    def combined_visc_fct(self, T_in_fm):
        eta_over_s=self.eta_over_s_fct(T_in_fm)
        zeta_over_s=0.0 #zeta_over_s_fct(T_in_fm)
        return (4./3.*eta_over_s+zeta_over_s)

    def get_T(self, tau):
        return hbarc*self.ns_sol.integrate(tau)


###################################################
########### Plot the temperature profile ##########
###################################################
#
#T0_in_GeV=.400
#tau0=0.5
#
#T_brick_fct=brick_profile(T0_in_GeV=T0_in_GeV)
#T_ns_fct=Bjorken_hydro_profile(T0_in_GeV=T0_in_GeV, tau0=tau0)
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
#plt.ylim(.1,1.5*T0_in_GeV)
##plt.xscale('log')
##plt.yscale('log')
##plt.yticks(np.arange(0.,0.3,.05))
#
#plt.xlabel(r'$\tau$ (fm)')
#plt.ylabel(r"$T$ (GeV)")
#
## Plot the exact solution
#tau_range=np.arange(tau0,tau_plot,(tau_plot-tau0)/100)
#T_from_brick= np.array(list(map(T_brick_fct.get_T,tau_range)))
#plt.plot(tau_range, T_from_brick,"-",color='red',label="", linewidth=3)
#T_from_ns= np.array(list(map(T_ns_fct.get_T,tau_range)))
#plt.plot(tau_range, T_from_ns,"--",color='blue',label="", linewidth=3)
#
#plt.legend()
#plt.tight_layout()
#plt.savefig("T_profile.pdf")
#plt.show()
#
