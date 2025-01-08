from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ...model import Model
from ...wc import WCModel as WCModel_numba


class WCModel(WCModel_numba):
    """
    The two-population Wilson-Cowan model
    """

    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None):

        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)

        # Initialize base class Model
        Model.__init__(self, integration=integration, params=params)


WCModel.args_names = [
    "startind",
    "t",
    "dt",
    "sqrt_dt",
    "N",
    "Cmat",
    "K_gl",
    "Dmat_ndt",
    "exc_init",
    "inh_init",
    "exc_ext_baseline",
    "inh_ext_baseline",
    "exc_ext",
    "inh_ext",
    "tau_exc",
    "tau_inh",
    "a_exc",
    "a_inh",
    "mu_exc",
    "mu_inh",
    "c_excexc",
    "c_excinh",
    "c_inhexc",
    "c_inhinh",
    "exc_ou_init",
    "inh_ou_init",
    "exc_ou_mean",
    "inh_ou_mean",
    "tau_ou",
    "sigma_ou",
    "key",
]
