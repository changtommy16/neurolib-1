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
        """Initialize WendlingModel with proper parent class initialization."""
        
        # Start with default parameters
        full_params = self.params.copy()
        
        # Update with user-provided parameters if any
        if params is not None:
            full_params.update(params)
            
        # Pass the integration function and params to the base class
        super().__init__(integration=timeIntegration, params=full_params)

        # Provide a safe default init_state if the user didn't set one later
        if not hasattr(self, "init_state"):
            self.init_state = np.zeros(10, dtype=np.float64)

    # ---- Default parameters ----
    params = dict(
        # Synaptic gains & time constants (Original Wendling et al. 2002)
        A=4.0,    a=100.0,     # excitatory postsynaptic potential
        B=40.0,   b=50.0,      # slow inhibitory postsynaptic potential (GABA_B)
        G=20.0,   g=350.0,     # fast inhibitory postsynaptic potential (GABA_A)

        # Connectivity constants (Wendling et al. 2002)
        C=135.0,               # average number of synaptic contacts
        C1=1.0,                # pyramidal cells → excitatory interneurons
        C2=0.8,                # excitatory interneurons → pyramidal cells  
        C3=0.25,               # pyramidal cells → slow inhibitory interneurons
        C4=0.25,               # slow inhibitory interneurons → pyramidal cells
        C5=0.3,                # pyramidal cells → fast inhibitory interneurons
        C6=0.1,                # slow inhibitory → fast inhibitory interneurons
        C7=0.8,                # fast inhibitory interneurons → pyramidal cells

        # Sigmoid function parameters (Wendling et al. 2002)
        e0=2.5,                # maximum firing rate (spikes/s)
        v0=6.0,                # postsynaptic potential for which sigmoid = e0/2 (mV)
        r=0.56,                # steepness of the sigmoid (mV^-1)

        # Background input from neighboring neural areas (simple noise like github version)
        p_mean=90.0,           # mean background input (spikes/s)
        p_sigma=30.0,          # noise amplitude (spikes/s)
        
        # Stimulus gain (from github version)
        SG=1.0,                # stimulus (synaptic) gain
        
        # Integration control (matching github version)
        dt=0.0001,          # integration step (s) - same as github Fs=512
        duration=15.0,         # simulation duration (s) - same as github
        seed=None,             # random seed
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
def _sigm(v, e0, v0, r):
    # Standard JR/Wendling sigmoid -> firing rate (Hz)
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@njit(cache=True, fastmath=True)
def _integrate_wendling(y0, n_steps, dt,
                        A,a, B,b, G,g,
                        C,C1,C2,C3,C4,C5,C6,C7,
                        e0,v0,r, p_mean, p_sigma, SG):
    ys = np.zeros((10, n_steps), dtype=np.float64)
    v_out = np.zeros(n_steps, dtype=np.float64)
    rate_out = np.zeros(n_steps, dtype=np.float64)
    y = y0.copy()

    for k in range(n_steps):
        # State variables following github_wendling.py structure
        y0_,y1,y2,y3,y4, y5,y6,y7,y8,y9 = y
        
        # Simple noise implementation like github version (no sqrt(dt) scaling)
        noise = np.random.normal(p_mean, p_sigma)
        
        # Derivatives following github_wendling.py exactly
        dy0 = y5
        dy5 = A * a * _sigm(y1-y2-y3, e0, v0, r) - 2.0 * a * y5 - a * a * y0_
        
        dy1 = y6
        dy6 = A * a * (noise + C2 * _sigm(C1 * y0_ + SG, e0, v0, r)) - 2.0 * a * y6 - a * a * y1
        
        dy2 = y7
        dy7 = B * b * (C4 * _sigm(C3 * y0_, e0, v0, r)) - 2.0 * b * y7 - b * b * y2
        
        dy3 = y8
        dy8 = G * g * (C7 * _sigm((C5 * y0_ - C6 * y4), e0, v0, r)) - 2.0 * g * y8 - g * g * y3
        
        dy4 = y9
        dy9 = B * b * (_sigm(C3 * y0_, e0, v0, r)) - 2.0 * b * y9 - b * b * y4
        
        # Euler integration
        y0_ += dt*dy0; y1 += dt*dy1; y2 += dt*dy2; y3 += dt*dy3; y4 += dt*dy4
        y5  += dt*dy5; y6 += dt*dy6; y7 += dt*dy7; y8 += dt*dy8; y9 += dt*dy9
        y[0]=y0_; y[1]=y1; y[2]=y2; y[3]=y3; y[4]=y4; y[5]=y5; y[6]=y6; y[7]=y7; y[8]=y8; y[9]=y9
        
        # Output: pyramidal membrane potential and firing rate (same as github)
        v_out[k] = y1 - y2 - y3  # LFP output like github version
        rate_out[k] = _sigm(y1-y2-y3, e0, v0, r)  # pyramidal firing rate
        
        # Store all state variables
        for i in range(10):
            ys[i, k] = y[i]

    return ys, v_out, rate_out


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
    
    # Call the numba-accelerated integration function
    ys, v_out, rate_out = _integrate_wendling(
        y0=y,
        n_steps=n_steps, dt=params["dt"],
        A=params["A"], a=params["a"], B=params["B"], b=params["b"], G=params["G"], g=params["g"],
        C=params["C"], C1=C1_scaled, C2=C2_scaled, C3=C3_scaled, C4=C4_scaled, C5=C5_scaled, C6=C6_scaled, C7=C7_scaled,
        e0=params["e0"], v0=params["v0"], r=params["r"],
        p_mean=params["p_mean"], p_sigma=params["p_sigma"], SG=params["SG"],
    )

    # Return only the state variables in the order of state_vars: y0, y1, y2, ..., y9
    t = np.arange(0, duration, dt)[:n_steps]
    return (t, *[ys[i] for i in range(10)])
