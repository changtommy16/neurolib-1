import numpy as np
from ...utils.collections import dotdict


def loadDefaultParams(Cmat=None, Dmat=None, seed=None, sigmoid_type="wendling2002"):
    """Load default parameters for the Wendling Neural Mass Model.
    
    This implements the Wendling-Chauvel model (Wendling et al., 2002)
    with 10 ODEs representing pyramidal cells, excitatory interneurons,
    slow inhibitory interneurons, and fast inhibitory interneurons.
    
    References:
    - Wendling, F., Bartolomei, F., Bellanger, J. J., & Chauvel, P. (2002). 
      Epileptic fast activity can be explained by a model of impaired GABAergic 
      dendritic inhibition. European Journal of Neuroscience, 15(9), 1499-1508.
    - Köksal Ersöz, E., et al. (2020). Neural mass modeling of slow-fast dynamics 
      of seizure initiation and abortion. PLoS Computational Biology, 16(11), e1008430.

    :param Cmat: Structural connectivity matrix (adjacency matrix), defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Fiber length matrix for delay computation, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional
    :param sigmoid_type: Sigmoid variant ("wendling2002" or "pcbi2020"), defaults to "wendling2002"
    :type sigmoid_type: str, optional
    :return: Dictionary with default parameters
    :rtype: dict
    """
    
    params = dotdict({})
    
    ### Runtime parameters (MATCHED TO WORKING CODE)
    params.dt = 0.1  # Time step (ms) = 0.0001 s (10 kHz sampling)
    params.duration = 10000  # Simulation duration (ms) - 10s
    np.random.seed(seed)
    params.seed = seed
    
    # Sigmoid type
    params.sigmoid_type = sigmoid_type
    
    # ------------------------------------------------------------------------
    # Global whole-brain network parameters
    # ------------------------------------------------------------------------
    
    params.signalV = 20.0  # Signal transmission speed (m/s)
    params.K_gl = 0.5  # Global coupling strength
    
    if Cmat is None:
        params.N = 1
        params.Cmat = np.zeros((1, 1))
        params.lengthMat = np.zeros((1, 1))
    else:
        params.Cmat = Cmat.copy()
        np.fill_diagonal(params.Cmat, 0)
        params.N = len(params.Cmat)
        params.lengthMat = Dmat if Dmat is not None else np.zeros_like(Cmat)
    
    # ------------------------------------------------------------------------
    # Local node parameters (Wendling 2002 defaults)
    # ------------------------------------------------------------------------
    
    # Synaptic gains (mV) - using working validation params
    params.A = 5.0  # Excitatory gain (validated)
    params.B = 25.0  # Slow inhibitory gain (validated)
    params.G = 15.0  # Fast inhibitory gain (validated)
    
    # Time constants (1/ms) - CORRECTED to match Wendling 2002 paper
    params.a = 100.0 / 1000.0  # 0.1 (1/ms) = 100 s^-1 (tau_a = 10 ms)
    params.b = 50.0 / 1000.0   # 0.05 (1/ms) = 50 s^-1 (tau_b = 20 ms) - STANDARD VALUE
    params.g = 500.0 / 1000.0  # 0.5 (1/ms) = 500 s^-1 (tau_g = 2 ms)
    
    # Connectivity constants
    params.C = 135.0  # Base connectivity constant
    params.C1 = 1.0 * 135.0   # C1 = C
    params.C2 = 0.8 * 135.0   # C2 = 0.8*C
    params.C3 = 0.25 * 135.0  # C3 = 0.25*C
    params.C4 = 0.25 * 135.0  # C4 = 0.25*C
    params.C5 = 0.3 * 135.0   # C5 = 0.3*C
    params.C6 = 0.1 * 135.0   # C6 = 0.1*C
    params.C7 = 0.8 * 135.0   # C7 = 0.8*C
    
    # External input - matched to working code
    params.p_mean = 90.0      # Mean input (Hz)
    params.p_sigma = 2.0      # Input noise std (Hz) - for Type 3 SWD
    
    # Sigmoid parameters (Wendling 2002 form)
    params.e0 = 2.5   # Half of maximum firing rate (Hz)
    params.v0 = 6.0   # Firing threshold (mV)
    params.r = 0.56   # Sigmoid slope (1/mV)
    
    # Integration method
    params.integration_method = "euler"  # "rk4" or "euler" - using Euler to match original author's code
    
    # ------------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------------
    
    # Zero initial conditions (matching original author's implementation)
    params.y0_init = np.zeros((params.N, 1))
    params.y1_init = np.zeros((params.N, 1))
    params.y2_init = np.zeros((params.N, 1))
    params.y3_init = np.zeros((params.N, 1))
    params.y4_init = np.zeros((params.N, 1))
    params.y5_init = np.zeros((params.N, 1))
    params.y6_init = np.zeros((params.N, 1))
    params.y7_init = np.zeros((params.N, 1))
    params.y8_init = np.zeros((params.N, 1))
    params.y9_init = np.zeros((params.N, 1))
    
    return params
