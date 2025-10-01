import numpy as np
from numba import njit
from neurolib.models.model import Model

class Wendling2005(Model):
    """
    Wendling 2005 model of interictal to ictal transition in temporal lobe epilepsy.

    This model is based on the paper:
    Wendling F, Hernandez A, Bellanger JJ, Chauvel P, Bartolomei F.
    Interictal to ictal transition in human temporal lobe epilepsy:
    insights from a computational model of intracerebral EEG.
    J Clin Neurophysiol. 2005 Oct;22(5):343-56.
    """

    name = "Wendling2005"
    description = "Wendling 2005 model of temporal lobe epilepsy"

    # Default parameters from the C code
    default_parameters = {
        "A": 3.5,       # EXC Parameter (excitation)
        "B": 20.0,      # SDI Parameter (slow dendritic inhibition)
        "G": 15.0,      # FSI Parameter (fast somatic inhibition)
        "a": 100.0,     # time constants of EPSPs and IPSPs
        "b": 50.0,
        "g": 500.0,
        "v0": 6.0,      # Sigmoid parameters
        "e0": 2.5,
        "r": 0.56,
        "C": 135.0,     # Global connectivity constant
        "C1_frac": 1.0,   # Connectivity fractions to be multiplied by C
        "C2_frac": 0.8,
        "C3_frac": 0.25,
        "C4_frac": 0.25,
        "C5_frac": 0.3,
        "C6_frac": 0.1,
        "C7_frac": 0.8,
        "meanP": 3.0,     # Input noise parameters: mean
        "sigmaP": 1.0,    # Input noise parameters: standard deviation
        "coefMultP": 30.0 # Input noise parameters: coefficient multiplier
    }

    state_vars = [
        "y0", "y1", "y2", "y3", "y4", # Potentials
        "y5", "y6", "y7", "y8", "y9"  # Derivatives
    ]
    
    init_vars = [
        "y0_init", "y1_init", "y2_init", "y3_init", "y4_init",
        "y5_init", "y6_init", "y7_init", "y8_init", "y9_init",
    ]
    
    output_vars = ["y0", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8", "y9"]
    default_output = "y1"  # Can compute v = y1 - y2 - y3 from outputs

    def __init__(self, params=None, **kwargs):
        # Start with default parameters
        merged_params = self.default_parameters.copy()
        
        # Update with user-provided parameters
        if params is not None:
            merged_params.update(params)
        
        # Call parent constructor with integration function and merged parameters
        super().__init__(integration=timeIntegration, params=merged_params, **kwargs)
        
        # Initialize state variables
        for var in self.state_vars:
            setattr(self, f"{var}_init", self.params.get(f"{var}_init", 0.0))

        # Initialize state variables
        for i, var in enumerate(self.state_vars):
            setattr(self, f"{var}_init", self.params.get(f"{var}_init", 0.0))


# Numba-accelerated integration function
@njit(cache=True, fastmath=True)
def _sigm(v, e0, v0, r):
    """Sigmoid function"""
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@njit(cache=True, fastmath=True)
def _integrate_wendling(y0, n_steps, dt,
                        A, a, B, b, G, g,
                        C1, C2, C3, C4, C5, C6, C7,
                        e0, v0, r, p_mean, p_sigma):
    """Core integration loop"""
    ys = np.zeros((10, n_steps), dtype=np.float64)
    y = y0.copy()

    for k in range(n_steps):
        y0_, y1, y2, y3, y4, y5, y6, y7, y8, y9 = y
        
        # Background input with noise
        # p(t) ~ N(p_mean, p_sigma) where both are already scaled by coefMultP
        p_t = np.random.normal(p_mean, p_sigma)
        
        # Derivatives
        dy0 = y5
        dy5 = A * a * _sigm(y1 - y2 - y3, e0, v0, r) - 2.0 * a * y5 - a * a * y0_
        
        dy1 = y6
        dy6 = A * a * (p_t + C2 * _sigm(C1 * y0_, e0, v0, r)) - 2.0 * a * y6 - a * a * y1
        
        dy2 = y7
        dy7 = B * b * C4 * _sigm(C3 * y0_, e0, v0, r) - 2.0 * b * y7 - b * b * y2
        
        dy3 = y8
        dy8 = G * g * C7 * _sigm(C5 * y0_ - C6 * y4, e0, v0, r) - 2.0 * g * y8 - g * g * y3
        
        dy4 = y9
        dy9 = B * b * _sigm(C3 * y0_, e0, v0, r) - 2.0 * b * y9 - b * b * y4
        
        # Euler integration
        y5 += dt * dy5
        y6 += dt * dy6
        y7 += dt * dy7
        y8 += dt * dy8
        y9 += dt * dy9
        
        y0_ += dt * dy0
        y1  += dt * dy1
        y2  += dt * dy2
        y3  += dt * dy3
        y4  += dt * dy4
        
        # Update state vector
        y[0] = y0_; y[1] = y1; y[2] = y2; y[3] = y3; y[4] = y4
        y[5] = y5;  y[6] = y6; y[7] = y7; y[8] = y8; y[9] = y9
        
        # Store
        for i in range(10):
            ys[i, k] = y[i]

    return ys


def timeIntegration(params):
    """Integration function for neurolib Model"""
    dt = params["dt"]
    duration = params["duration"]
    n_steps = int(duration / dt)
    
    # Get initial state
    y = params.get("init_state", np.zeros(10, dtype=np.float64))
    
    # Set random seed if provided
    if params.get("seed") is not None:
        np.random.seed(params["seed"])
    
    # Scale connectivity constants by C
    C = params["C"]
    C1 = params["C1_frac"] * C
    C2 = params["C2_frac"] * C
    C3 = params["C3_frac"] * C
    C4 = params["C4_frac"] * C
    C5 = params["C5_frac"] * C
    C6 = params["C6_frac"] * C
    C7 = params["C7_frac"] * C
    
    # Background noise (both mean and std are scaled by coefMultP)
    p_mean = params["meanP"] * params["coefMultP"]  # 3.0 * 30.0 = 90.0
    p_sigma = params["sigmaP"] * params["coefMultP"]  # 1.0 * 30.0 = 30.0
    
    # Call numba-accelerated integration
    ys = _integrate_wendling(
        y0=y,
        n_steps=n_steps, dt=dt,
        A=params["A"], a=params["a"],
        B=params["B"], b=params["b"],
        G=params["G"], g=params["g"],
        C1=C1, C2=C2, C3=C3, C4=C4, C5=C5, C6=C6, C7=C7,
        e0=params["e0"], v0=params["v0"], r=params["r"],
        p_mean=p_mean, p_sigma=p_sigma,
    )

    # Return time array and state variables
    t = np.arange(0, duration, dt)[:n_steps]
    return (t, *[ys[i] for i in range(10)])
