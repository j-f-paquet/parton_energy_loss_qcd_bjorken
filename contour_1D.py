from scipy.stats import norm
from scipy import integrate
import numpy as np

from ...core.acquisition import Acquisition
class contour1D(Acquisition):

    """
    This acquisition is for Sequential Experimental Design for
    Contour Estimation (Ranjan et al. 2008). This method is only suitable for simulations
    with single output
    """
    def __init__(self, model, a = 0, alpha = 0.1) -> None:

        """
        :param model: The emulation model
        :param a : Target observable evalue
        :param alpha: Control the sampling of the regions with highest variance

        """
        self.model = model
        self.a = a
        self.alpha = alpha
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        m_vec, s2_vec = self.model.predict(x)
        # target a, alpha should be inputs (to move to __init__)
        a = self.a
        alpha = self.alpha

        f_acqu_x = []
        for index in range(len(m_vec)):
            m = m_vec[index]; s2 = s2_vec[index]; s = np.sqrt(s2)
            f = lambda y: (y-m)**2 * norm.pdf((y-m)/s) # integrand for the third term
            f_acqu_x_temp = ((alpha**2 * s2) - (m-a)**2) * (norm.cdf((a-m)/s+alpha) - norm.cdf((a-m)/s-alpha)) \
                    + 2*(m-a)*s2 * (norm.pdf((a-m)/s+alpha) - norm.pdf((a-m)/s-alpha)) \
                    - integrate.quad(f, a-alpha*s, a+alpha*s)[0]
            f_acqu_x.append(f_acqu_x_temp)

        f_acqu_x = np.array(f_acqu_x); f_acqu_x = f_acqu_x.reshape(-1,1)

        return f_acqu_x


    @property
    def has_gradients(self):
        return False
