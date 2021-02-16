import numpy as np
import scipy #.optimize
import scipy.interpolate
import scipy.integrate

hbarc=0.1973


######################################################
############## Parton energy loss rates ##############
######################################################

def init_energy_loss_rates(alpha_s, N_f, N_c):

    #g_s=2
    #alpha_s=g_s**2/(4.*numpy.pi)

    #alpha_s=0.1
    g_s=np.sqrt(4*np.pi*alpha_s)

    #N_f=0

    #N_c=3
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

        return -0.5*norm*np.real(scipy.special.lambertw(-m_D(T)**4/(norm*C_0**2*omega),k=-1))
    #print(qhat_eff(1e-2*1000,.3))
    # should be about 0.052 with N_f=0 and alpha_s=0.1

    def dGamma_domega_inel(p, omega,T):

        if (p == 0)or(omega>=p):
            res=0.0

        else:

            z=omega/p

            qhat_eff_val=qhat_eff(omega,T)
            #qhat_eff_val=0.04 #qhat_eff(1e-2*p,T)
            #print(qhat_eff_val)

            res=alpha_s*N_c/(np.pi*p)*np.power(z*(1-z),-3./2.)*np.sqrt(qhat_eff_val/p)

        return res


    return lambda p, omega, T : dGamma_domega_inel(p,omega,T)


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



