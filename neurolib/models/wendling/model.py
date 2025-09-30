# -*- coding: utf-8 -*-
"""
Single-node Wendling–Chauvel neural mass (10D) as a neurolib.Model
- Populations: PY (pyramidal), EX (excitatory), SI (slow inh), FI (fast inh)
- Second-order PSP filters per pathway; sigmoid firing-rate
- Pure Wendling 2002 model
"""
import numpy as np
from numba import njit
from neurolib.models.model import Model


class WendlingModel(Model):
    """
    Minimal single-node Wendling model in neurolib 'Model' style.
    Outputs:
      - t: time vector
      - state: (T, 10) full state trajectory
      - v: pyramidal dendritic potential v_pyr = y1 - y2 - y3
      - rate: pyramidal firing rate S(v_pyr)
    """
    
    name = "wendling"
    description = "Wendling neural mass model"
    
    # ---- Required metadata for neurolib.Model ----
    state_vars = ["y0","y1","y2","y3","y4", "y5","y6","y7","y8","y9"]  # 10D
    init_vars  = ["init_state"]
    output_vars = ["y0","y1","y2","y3","y4", "y5","y6","y7","y8","y9"]  # Only state variables
    default_output = "v"

    def __init__(self, params=None):
        """Initialize WendlingModel, ensuring user parameters are merged with defaults."""
        
        # Get a copy of the default parameters from the class
        default_params = self.__class__.params.copy()
        
        # Update the defaults with any user-provided parameters
        if params is not None:
            default_params.update(params)
            
        # Pass the fully merged parameter set to the parent constructor
        super().__init__(integration=timeIntegration, params=default_params)

        # Provide a safe default init_state if the user didn't set one later
        if not hasattr(self, "init_state"):
            self.init_state = np.zeros(10, dtype=np.float64)

    # ---- Default parameters ----
    params = dict(
        # Synaptic gains & time constants (Arrais 2019 TABLE I)
        A=5.0,    a=100.0,   # Excitatory: A ∈ [0.1, 10] mV
        B=25.0,   b=30.0,    # Slow inhibitory: B ∈ [0.1, 50] mV, b = 30 s⁻¹
        G=20.0,   g=350.0,   # Fast inhibitory: G ∈ [20, 50] mV, g = 350 s⁻¹

        # Connectivity
        C=135.0,
        C1=1.0, C2=0.8, C3=0.25, C4=0.25, C5=0.3, C6=0.1, C7=0.8,

        # Sigmoid (Arrais 2019 formulation)
        e0=2.5, v0=6.0, r=0.56,
        S=0.0,  # Sigmoid offset parameter (S in Arrais 2019)

        # Background input p(t) = p_mean + p_std * N(0,1)
        # Arrais 2019 uses p_mean=90, p_std=30
        p_mean=90.0,   # Hz
        p_sigma=30.0,  # Hz (standard deviation of noise)

        # Integration
        dt=0.0001,     # 10 kHz
        duration=50.0,
        seed=None,
    )

    def run(self, **kwargs):
        """Override run method to compute derived outputs after integration."""
        # Call parent run method
        super().run(**kwargs)
        
        return self.outputs
    
    def storeOutputsAndStates(self, t, variables, append=False):
        """Override to fix time-output length mismatch by ensuring time vector alignment."""
        # Store time array with proper IC removal to match state variables
        if self.startindt > 0:
            # Remove initial conditions from time vector to match state variables
            t_trimmed = t[self.startindt:]
            self.setOutput("t", t_trimmed, append=append, removeICs=False)
        else:
            self.setOutput("t", t, append=append, removeICs=False)
        
        self.setStateVariables("t", t)
        
        # Store state variables with IC removal as usual
        for svn, sv in zip(self.state_vars, variables):
            if svn in self.output_vars:
                self.setOutput(svn, sv, append=append, removeICs=True)
            self.setStateVariables(svn, sv)

    def integrate(self, append_outputs=False, simulate_bold=False):
        """Override integrate to compute derived outputs after integration."""
        # Call parent integrate method
        super().integrate(append_outputs=append_outputs, simulate_bold=simulate_bold)
        
        # Compute derived outputs v and rate after integration
        self._compute_derived_outputs()

    def _compute_derived_outputs(self):
        """Compute v (pyramidal potential) and rate (firing rate) from state variables."""
        if "y1" in self.outputs and "y2" in self.outputs and "y3" in self.outputs:
            # Wendling 2002: pyramidal membrane potential = y1 - y2 - y3
            # (excitatory PSP - slow inhibitory PSP - fast inhibitory PSP)
            v = self.outputs["y1"] - self.outputs["y2"] - self.outputs["y3"]
            
            # Don't remove ICs since y1,y2,y3 already had them removed and now match time vector
            self.setOutput("v", v, removeICs=False)
            
            # Compute firing rate S(v) using Wendling sigmoid
            rate = 2.0 * self.params["e0"] / (1.0 + np.exp(self.params["r"] * (self.params["v0"] - v)))
            self.setOutput("rate", rate, removeICs=False)

    # convenience: computed outputs (removed properties to avoid conflict with setOutput)


# ---------- Numba-accelerated core ----------

@njit(cache=True, fastmath=True)
def _sigm(v, e0, v0, r, S):
    # Arrais 2019 sigmoid with offset S: Sig(S + v)
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - (S + v))))


@njit(cache=True, fastmath=True)
def _integrate_wendling(y0, n_steps, dt,
                        A,a, B,b, G,g,
                        C,C1,C2,C3,C4,C5,C6,C7,
                        e0,v0,r,S, p_mean, p_sigma):
    ys = np.zeros((10, n_steps), dtype=np.float64)
    y = y0.copy()

    for k in range(n_steps):
        # State variables: y0-y4 are potentials, dy0-dy4 are velocities (first derivatives)
        y0_, y1, y2, y3, y4, dy0, dy1, dy2, dy3, dy4 = y
        
        # Background input: p(t) in Hz
        # Gaussian white noise input: p(t) = p_mean + p_sigma * ξ(t) where ξ(t) ~ N(0,1)
        # For Euler-Maruyama integration of SDEs, the noise term is scaled by sqrt(dt)
        xi_t = np.random.normal(0.0, 1.0)  # Standard Gaussian random variable ξ(t) ~ N(0,1)
        p_t = p_mean + p_sigma * xi_t #* np.sqrt(dt)  # Proper Gaussian white noise scaling
        
        # Derivatives following Arrais 2019 formulation:
        # ÿ0(t) = Aa*Sig(S + y1 − y2 − y3) − 2a*ẏ0(t) − a²*y0(t)
        ddy0 = A * a * _sigm(y1-y2-y3, e0, v0, r, S) - 2.0 * a * dy0 - a * a * y0_  # Second derivative (acceleration)
        
        # ÿ1(t) = Aa{p(t) + Sig(S + C1*y0)} − 2a*ẏ1(t) − a²*y1(t)
        # Note: No C2 coefficient in Arrais 2019
        ddy1 = A * a * (p_t + C2*_sigm(C1 * y0_, e0, v0, r, S)) - 2.0 * a * dy1 - a * a * y1
        
        # ÿ2(t) = Bb*C4*Sig(S + C3*y0) − 2b*ẏ2(t) − b²*y2(t)
        ddy2 = B * b * C4 * _sigm(C3 * y0_, e0, v0, r, S) - 2.0 * b * dy2 - b * b * y2
        
        # ÿ3(t) = Gg*C7*Sig(S + C5*y0 − y4) − 2g*ẏ3(t) − g²*y3(t)
        ddy3 = G * g * C7 * _sigm(C5 * y0_ - C6*y4, e0, v0, r, S) - 2.0 * g * dy3 - g * g * y3
        
        # ÿ4(t) = Bb*C6*Sig(S + C3*y0) − 2b*ẏ4(t) − b²*y4(t)
        ddy4 = B * b * _sigm(C3 * y0_, e0, v0, r, S) - 2.0 * b * dy4 - b * b * y4
        
        # Euler integration (following Arrais 2019 guide)
        # Step 1: Update velocities (dy_i) using accelerations (ddy_i)
        dy0 += dt * ddy0  # ẏ0 += ÿ0 * dt
        dy1 += dt * ddy1  # ẏ1 += ÿ1 * dt
        dy2 += dt * ddy2  # ẏ2 += ÿ2 * dt
        dy3 += dt * ddy3  # ẏ3 += ÿ3 * dt
        dy4 += dt * ddy4  # ẏ4 += ÿ4 * dt
        
        # Step 2: Update potentials (y_i) using velocities (dy_i)
        y0_ += dt * dy0  # y0 += ẏ0 * dt
        y1  += dt * dy1  # y1 += ẏ1 * dt
        y2  += dt * dy2  # y2 += ẏ2 * dt
        y3  += dt * dy3  # y3 += ẏ3 * dt
        y4  += dt * dy4  # y4 += ẏ4 * dt
        
        # Update state vector
        y[0] = y0_; y[1] = y1;  y[2] = y2;  y[3] = y3;  y[4] = y4
        y[5] = dy0; y[6] = dy1; y[7] = dy2; y[8] = dy3; y[9] = dy4
        
        
        # Store all state variables
        for i in range(10):
            ys[i, k] = y[i]

    return ys


def timeIntegration(params):
    """
    Integrates the Wendling model using Euler-Maruyama method for stochastic differential equations.
    
    :param params: Parameter dictionary for the model
    :type params: dict
    :return: Integrated state variables and derived outputs
    :rtype: tuple
    """
    dt = params["dt"]
    duration = params["duration"]
    n_steps = int(duration / dt)
    
    # Get initial state
    y = params.get("init_state", np.zeros(10, dtype=np.float64))
    
    # Set random seed if provided
    if params.get("seed") is not None:
        np.random.seed(params["seed"])
    
    # Scale connectivity constants by C (like github version)
    C1_scaled = params["C1"] * params["C"]
    C2_scaled = params["C2"] * params["C"]
    C3_scaled = params["C3"] * params["C"]
    C4_scaled = params["C4"] * params["C"]
    C5_scaled = params["C5"] * params["C"]
    C6_scaled = params["C6"] * params["C"]
    C7_scaled = params["C7"] * params["C"]
    
    # Call the numba-accelerated integration function (Arrais 2019 formulation)
    ys = _integrate_wendling(
        y0=y,
        n_steps=n_steps, dt=params["dt"],
        A=params["A"], a=params["a"], B=params["B"], b=params["b"], G=params["G"], g=params["g"],
        C=params["C"], C1=C1_scaled, C2=C2_scaled, C3=C3_scaled, C4=C4_scaled, C5=C5_scaled, C6=C6_scaled, C7=C7_scaled,
        e0=params["e0"], v0=params["v0"], r=params["r"], S=params["S"],
        p_mean=params["p_mean"], p_sigma=params["p_sigma"],
    )

    # Return time array and state variables
    t = np.arange(0, duration, dt)[:n_steps]
    return (t, *[ys[i] for i in range(10)])
