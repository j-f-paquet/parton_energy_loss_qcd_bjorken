from scipy.stats import norm
from scipy import integrate
import numpy as np

from ...core.acquisition import Acquisition

def EI_1Obs(m, s2, a, alpha):
    s = np.sqrt(s2)
    f = lambda y: (y-m)**2 * norm.pdf((y-m)/s) # integrand for the third term
    f_acqu_x_temp = ((alpha**2 * s2) - (m-a)**2) * (norm.cdf((a-m)/s+alpha) - norm.cdf((a-m)/s-alpha)) \
            + 2*(m-a)*s2 * (norm.pdf((a-m)/s+alpha) - norm.pdf((a-m)/s-alpha)) \
            - integrate.quad(f, a-alpha*s, a+alpha*s)[0]
    return f_acqu_x_temp


class contourMinEI(Acquisition):

    """
    This acquisition is extended from Sequential Experimental Design for
    Contour Estimation (Ranjan et al. 2008). It works for multiple output problem. It has not taken the noise of experimental data into account yet.
    """
    def __init__(self, modelwrapper, a_list, alpha_list, w_list) -> None:

        """
        :param modelwrapper: One model for multiple inputs and multiple outputs, combining multiple independent models
        :param a_list : Target observable evalues, one for each observable
        :param alpha_list: Control the sampling of the regions with highest variance, one for each observable
        :param w_list: Weights of EI put on each observable
        """
        self.modelwrapper = modelwrapper
        self.a_list = a_list
        self.alpha_list = alpha_list
        self.w_list = w_list
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        
        '''
        Idea: have a model wrapper for a set of independent GPs, one model corresponding to one observable, calculate EI(x) for each model, then combine them for final EI to be optimized (in this acquisition function we use the weighted sum of all EI's)
        '''
        
        # save predictive means and variances of all observables in 2 dictionaries
        m_dict = dict(); s2_dict = dict()
        for i in range(len(self.a_list)):
            new_column = [[i]]*len(x) # predict for ith emulator
            pred_temp = self.modelwrapper.predict(np.append(x,new_column,axis=1))
            m_vec = pred_temp[0] # vector of predictive means
            s2_vec = pred_temp[1] # vector of predictive variances
            m_dict[i] = m_vec; s2_dict[i] = s2_vec

        f_acqu_x = []
        for index in range(len(x)): # for each x, calculate EI of each observable
            EI_eachObs = []
            for i in range(len(self.a_list)):
                a = self.a_list[i]
                alpha = self.alpha_list[i]
                m = m_dict[i][index]; s2 = s2_dict[i][index]
                EI_eachObs.append(EI_1Obs(m, s2, a, alpha))
            f_acqu_x.append(min(EI_eachObs)) # Approach 1, return minimum EI

        f_acqu_x = np.array(f_acqu_x); f_acqu_x = f_acqu_x.reshape(-1,1)
        
        return f_acqu_x


    @property
    def has_gradients(self):
        return False
