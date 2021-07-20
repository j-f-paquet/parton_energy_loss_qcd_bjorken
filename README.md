# Contour estimation with active learning sampling

We will be using emukit to do the active learning sampling for new aquisition functions.

To make any aqusition function available in emukit, copy the aqusition function script to the following folder of your local instalation (conda environment instalation) of the emukit.

> emukit/experimental_design/acquisitions

If you are using conda env, your local instalation of emukit will be somewhere like here.


/Users/dananjayaliyanage/miniconda3/envs/parton_loss/lib/python3.6/site-packages/emukit/experimental_design/acquisitions.

To get the contour_1D aqusition function to my local instalation of emukit I did the following. 

> cp contour_1D.py /Users/dananjayaliyanage/miniconda3/envs/parton_loss/lib/python3.6/site-packages/emukit/experimental_design/acquisitions


# Some references for the physics behind the package

Evolution of the medium: 0+1D Bjorken QCD hydrodynamics
Eq. 2 of https://arxiv.org/pdf/1912.06287.pdf

Parton energy loss and parton evolution:
https://arxiv.org/pdf/1006.2379.pdf
- Inelastic rate approximation: section III-D
- Elastic rate approximation: Eq. 21
- Evolution equation: Eq. 17

Parton distribution initial conditions (and also parton evolution):
Eq. 9 of https://arxiv.org/pdf/hep-ph/0309332.pdf
