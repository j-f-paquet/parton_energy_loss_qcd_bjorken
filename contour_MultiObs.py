from scipy.stats import norm
from scipy import integrate
import numpy as np

from ...core.acquisition import Acquisition

def EI_1Obs(m, s, a, alpha):
    f = lambda y: (y-m)**2 * norm.pdf((y-m)/s) # integrand for the third term
    f_acqu_x_temp = ((alpha**2 * s2) - (m-a)**2) * (norm.cdf((a-m)/s+alpha) - norm.cdf((a-m)/s-alpha)) \
            + 2*(m-a)*s2 * (norm.pdf((a-m)/s+alpha) - norm.pdf((a-m)/s-alpha)) \
            - integrate.quad(f, a-alpha*s, a+alpha*s)[0]
    return f_acqu_x_temp


class contourWeightedEI(Acquisition):

    """
    This acquisition is extended from Sequential Experimental Design for
    Contour Estimation (Ranjan et al. 2008). It works for multiple output problem. It has not taken the noise of experimental data into account yet.
    """
    def __init__(self, model_list, a_list, alpha_list, w_list) -> None:

        """
        :param model_list: The list of emulation models, one for each observable
        :param a_list : Target observable evalues, one for each observable
        :param alpha_list: Control the sampling of the regions with highest variance, one for each observable
        :param w_list: Weights of EI put on each observable
        """
        self.model_list = model_list
        self.a_list = a_list
        self.alpha_list = alpha_list
        self.w_list = w_list
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        
        '''
        Idea: have a list of models as inputs, one model corresponding to one observable, calculate EI(x) for each model, then combine them for final EI to be optimized (in this acquisition function we use the weighted sum of all EI's)
        '''
        
        # save predictive means and variances of all observables in 2 dictionaries
        m_dict = dict(); s2_dict = dict()
        for i in range(len(self.model_list)):
            model = self.model_list[i]
            m_vec, s2_vec = model.predict(x)
            m_dict[i] = m_vec; s2_dict[i] = s2_vec

        f_acqu_x = []
        for index in range(len(m_dict[0])): # same number of x for each observable
            EI_eachObs = [] # for each x, calculate EI of each observable
            for i in range(len(self.model_list)):
                a = self.a_list[i]
                alpha = self.alpha_list[i]
                m = m_dict[i][index]; s2 = s2_dict[i][index]; s = np.sqrt(s2)
                EI_eachObs.append(EI_1Obs(m, s, a, alpha))
            f_acqu_x.append(sum(map(lambda wi, EIi: wi * EIi, w_list, EI_eachObs))) # Approach 2, return weighted sum

        f_acqu_x = np.array(f_acqu_x); f_acqu_x = f_acqu_x.reshape(-1,1)

        return f_acqu_x


    @property
    def has_gradients(self):
        return False
