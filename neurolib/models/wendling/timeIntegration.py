import numpy as np
import numba
from numba import njit

from ...utils import model_utils as mu


def timeIntegration(params):
    """Time integration for Wendling Neural Mass Model.
    
    Implements the 10-ODE Wendling-Chauvel model with RK4 or Euler integration.
    Supports both single node and whole-brain network simulations.
    
    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables (t, y0, y1, ..., y9)
    :rtype: tuple of numpy.ndarray
    """
    
    dt = params["dt"]  # Time step (ms)
    duration = params["duration"]  # Simulation duration (ms)
    RNGseed = params["seed"]  # Random seed
    
    # Set random seed if provided
    if RNGseed is not None:
        np.random.seed(RNGseed)
    
    # ------------------------------------------------------------------------
    # Local parameters
    # ------------------------------------------------------------------------
    A = params["A"]
    B = params["B"]
    G = params["G"]
    a = params["a"]
    b = params["b"]
    g = params["g"]
    C1 = params["C1"]
    C2 = params["C2"]
    C3 = params["C3"]
    C4 = params["C4"]
    C5 = params["C5"]
    C6 = params["C6"]
    C7 = params["C7"]
    p_mean = params["p_mean"]
    p_sigma = params["p_sigma"]
    e0 = params["e0"]
    v0 = params["v0"]
    r = params["r"]
    sigmoid_type = params.get("sigmoid_type", "wendling2002")
    integration_method = params.get("integration_method", "rk4")
    
    # ------------------------------------------------------------------------
    # Global coupling parameters
    # ------------------------------------------------------------------------
    Cmat = params["Cmat"]
    N = len(Cmat)  # Number of nodes
    K_gl = params["K_gl"]  # Global coupling strength
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]
    
    if N == 1:
        Dmat = np.zeros((N, N))
    else:
        # Compute delay matrix
        Dmat = mu.computeDelayMatrix(lengthMat, signalV)
        Dmat[np.eye(len(Dmat)) == 1] = np.zeros(len(Dmat))
    
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # Delay matrix in multiples of dt
    
    # ------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time vector (ms)
    
    max_global_delay = int(np.max(Dmat_ndt))
    startind = max_global_delay + 1  # Start index after initial conditions
    
    # State variable arrays
    y0_arr = np.zeros((N, startind + len(t)))
    y1_arr = np.zeros((N, startind + len(t)))
    y2_arr = np.zeros((N, startind + len(t)))
    y3_arr = np.zeros((N, startind + len(t)))
    y4_arr = np.zeros((N, startind + len(t)))
    y5_arr = np.zeros((N, startind + len(t)))
    y6_arr = np.zeros((N, startind + len(t)))
    y7_arr = np.zeros((N, startind + len(t)))
    y8_arr = np.zeros((N, startind + len(t)))
    y9_arr = np.zeros((N, startind + len(t)))
    
    # Set initial conditions
    if np.shape(params["y0_init"])[1] == 1:
        y0_arr[:, :startind] = np.dot(params["y0_init"], np.ones((1, startind)))
        y1_arr[:, :startind] = np.dot(params["y1_init"], np.ones((1, startind)))
        y2_arr[:, :startind] = np.dot(params["y2_init"], np.ones((1, startind)))
        y3_arr[:, :startind] = np.dot(params["y3_init"], np.ones((1, startind)))
        y4_arr[:, :startind] = np.dot(params["y4_init"], np.ones((1, startind)))
        y5_arr[:, :startind] = np.dot(params["y5_init"], np.ones((1, startind)))
        y6_arr[:, :startind] = np.dot(params["y6_init"], np.ones((1, startind)))
        y7_arr[:, :startind] = np.dot(params["y7_init"], np.ones((1, startind)))
        y8_arr[:, :startind] = np.dot(params["y8_init"], np.ones((1, startind)))
        y9_arr[:, :startind] = np.dot(params["y9_init"], np.ones((1, startind)))
    else:
        y0_arr[:, :startind] = params["y0_init"][:, -startind:]
        y1_arr[:, :startind] = params["y1_init"][:, -startind:]
        y2_arr[:, :startind] = params["y2_init"][:, -startind:]
        y3_arr[:, :startind] = params["y3_init"][:, -startind:]
        y4_arr[:, :startind] = params["y4_init"][:, -startind:]
        y5_arr[:, :startind] = params["y5_init"][:, -startind:]
        y6_arr[:, :startind] = params["y6_init"][:, -startind:]
        y7_arr[:, :startind] = params["y7_init"][:, -startind:]
        y8_arr[:, :startind] = params["y8_init"][:, -startind:]
        y9_arr[:, :startind] = params["y9_init"][:, -startind:]
    
    # Normalize connectivity matrix
    if N > 1:
        Cmat_normalized = Cmat / np.max(Cmat) if np.max(Cmat) > 0 else Cmat
    else:
        Cmat_normalized = Cmat
    
    # ------------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------------
    if integration_method == "rk4":
        # Use RK4 integration
        return_arrays = rk4_integration(
            y0_arr, y1_arr, y2_arr, y3_arr, y4_arr,
            y5_arr, y6_arr, y7_arr, y8_arr, y9_arr,
            t, dt, startind, N,
            A, B, G, a, b, g,
            C1, C2, C3, C4, C5, C6, C7,
            p_mean, p_sigma,
            e0, v0, r, sigmoid_type,
            Cmat_normalized, K_gl, Dmat_ndt
        )
    else:
        # Use Euler integration
        return_arrays = euler_integration(
            y0_arr, y1_arr, y2_arr, y3_arr, y4_arr,
            y5_arr, y6_arr, y7_arr, y8_arr, y9_arr,
            t, dt, startind, N,
            A, B, G, a, b, g,
            C1, C2, C3, C4, C5, C6, C7,
            p_mean, p_sigma,
            e0, v0, r, sigmoid_type,
            Cmat_normalized, K_gl, Dmat_ndt
        )
    
    # Return time vector and all state variables (including initial conditions)
    return (t,) + return_arrays


@numba.njit
def sigmoid_wendling2002(v, e0, v0, r):
    """Wendling 2002 sigmoid: S(v) = 2*e0 / (1 + exp(r*(v0 - v)))"""
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@numba.njit
def sigmoid_pcbi2020(v):
    """PCBI 2020 sigmoid: S(v) = 5 / (1 + exp(0.56*(6 - v)))"""
    return 5.0 / (1.0 + np.exp(0.56 * (6.0 - v)))


@numba.njit
def compute_sigmoid(v, e0, v0, r, sigmoid_type):
    """Compute sigmoid based on type."""
    if sigmoid_type == 0:  # wendling2002
        return sigmoid_wendling2002(v, e0, v0, r)
    else:  # pcbi2020
        return sigmoid_pcbi2020(v)


@numba.njit
def derivatives_wendling(
    y0, y1, y2, y3, y4, y5, y6, y7, y8, y9,
    A, B, G, a, b, g,
    C1, C2, C3, C4, C5, C6, C7,
    p_t, e0, v0, r, sigmoid_type, coupling_input
):
    """
    Compute derivatives for the Wendling 10-ODE system.
    
    State variables:
    y0 - Pyramidal cell membrane potential
    y1 - Excitatory input to pyramidal cells
    y2 - Slow inhibitory input
    y3 - Fast inhibitory input
    y4 - Interneuron population potential
    y5-y9 - Derivatives of y0-y4
    """
    
    # Compute sigmoid inputs
    S_y1_y2_y3 = compute_sigmoid(y1 - y2 - y3, e0, v0, r, sigmoid_type)
    S_C1_y0 = compute_sigmoid(C1 * y0, e0, v0, r, sigmoid_type)
    S_C3_y0 = compute_sigmoid(C3 * y0, e0, v0, r, sigmoid_type)
    S_C5_y0_C6_y4 = compute_sigmoid(C5 * y0 - C6 * y4, e0, v0, r, sigmoid_type)
    
    # 10 ODEs
    dy0 = y5
    dy5 = A * a * (S_y1_y2_y3 + coupling_input) - 2.0 * a * y5 - a * a * y0
    
    dy1 = y6
    dy6 = A * a * (p_t + C2 * S_C1_y0) - 2.0 * a * y6 - a * a * y1
    
    dy2 = y7
    dy7 = B * b * C4 * S_C3_y0 - 2.0 * b * y7 - b * b * y2
    
    dy3 = y8
    dy8 = G * g * C7 * S_C5_y0_C6_y4 - 2.0 * g * y8 - g * g * y3
    
    dy4 = y9
    dy9 = B * b * S_C3_y0 - 2.0 * b * y9 - b * b * y4
    
    return dy0, dy1, dy2, dy3, dy4, dy5, dy6, dy7, dy8, dy9


@numba.njit
def rk4_integration(
    y0_arr, y1_arr, y2_arr, y3_arr, y4_arr,
    y5_arr, y6_arr, y7_arr, y8_arr, y9_arr,
    t, dt, startind, N,
    A, B, G, a, b, g,
    C1, C2, C3, C4, C5, C6, C7,
    p_mean, p_sigma,
    e0, v0, r, sigmoid_type_str,
    Cmat, K_gl, Dmat_ndt
):
    """RK4 integration for Wendling model."""
    
    # Convert sigmoid type to integer for numba
    sigmoid_type = 0 if sigmoid_type_str == "wendling2002" else 1
    
    for i in range(len(t)):
        idx = startind + i
        
        for node in range(N):
            # Get current state
            y0 = y0_arr[node, idx - 1]
            y1 = y1_arr[node, idx - 1]
            y2 = y2_arr[node, idx - 1]
            y3 = y3_arr[node, idx - 1]
            y4 = y4_arr[node, idx - 1]
            y5 = y5_arr[node, idx - 1]
            y6 = y6_arr[node, idx - 1]
            y7 = y7_arr[node, idx - 1]
            y8 = y8_arr[node, idx - 1]
            y9 = y9_arr[node, idx - 1]
            
            # External input with noise (Euler-Maruyama scaling)
            p_t = p_mean + p_sigma * np.random.randn() * np.sqrt(dt)
            
            # Compute coupling input from other nodes
            coupling_input = 0.0
            if N > 1:
                for j in range(N):
                    if Cmat[node, j] > 0:
                        delay_idx = idx - 1 - Dmat_ndt[node, j]
                        if delay_idx >= 0:
                            # Couple via output: S(y1 - y2 - y3)
                            delayed_output = compute_sigmoid(
                                y1_arr[j, delay_idx] - y2_arr[j, delay_idx] - y3_arr[j, delay_idx],
                                e0, v0, r, sigmoid_type
                            )
                            coupling_input += K_gl * Cmat[node, j] * delayed_output
            
            # RK4 stages
            # k1
            k1 = derivatives_wendling(
                y0, y1, y2, y3, y4, y5, y6, y7, y8, y9,
                A, B, G, a, b, g, C1, C2, C3, C4, C5, C6, C7,
                p_t, e0, v0, r, sigmoid_type, coupling_input
            )
            
            # k2
            k2 = derivatives_wendling(
                y0 + 0.5 * dt * k1[0], y1 + 0.5 * dt * k1[1],
                y2 + 0.5 * dt * k1[2], y3 + 0.5 * dt * k1[3],
                y4 + 0.5 * dt * k1[4], y5 + 0.5 * dt * k1[5],
                y6 + 0.5 * dt * k1[6], y7 + 0.5 * dt * k1[7],
                y8 + 0.5 * dt * k1[8], y9 + 0.5 * dt * k1[9],
                A, B, G, a, b, g, C1, C2, C3, C4, C5, C6, C7,
                p_t, e0, v0, r, sigmoid_type, coupling_input
            )
            
            # k3
            k3 = derivatives_wendling(
                y0 + 0.5 * dt * k2[0], y1 + 0.5 * dt * k2[1],
                y2 + 0.5 * dt * k2[2], y3 + 0.5 * dt * k2[3],
                y4 + 0.5 * dt * k2[4], y5 + 0.5 * dt * k2[5],
                y6 + 0.5 * dt * k2[6], y7 + 0.5 * dt * k2[7],
                y8 + 0.5 * dt * k2[8], y9 + 0.5 * dt * k2[9],
                A, B, G, a, b, g, C1, C2, C3, C4, C5, C6, C7,
                p_t, e0, v0, r, sigmoid_type, coupling_input
            )
            
            # k4
            k4 = derivatives_wendling(
                y0 + dt * k3[0], y1 + dt * k3[1],
                y2 + dt * k3[2], y3 + dt * k3[3],
                y4 + dt * k3[4], y5 + dt * k3[5],
                y6 + dt * k3[6], y7 + dt * k3[7],
                y8 + dt * k3[8], y9 + dt * k3[9],
                A, B, G, a, b, g, C1, C2, C3, C4, C5, C6, C7,
                p_t, e0, v0, r, sigmoid_type, coupling_input
            )
            
            # Update state
            y0_arr[node, idx] = y0 + (dt / 6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            y1_arr[node, idx] = y1 + (dt / 6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            y2_arr[node, idx] = y2 + (dt / 6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
            y3_arr[node, idx] = y3 + (dt / 6.0) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
            y4_arr[node, idx] = y4 + (dt / 6.0) * (k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
            y5_arr[node, idx] = y5 + (dt / 6.0) * (k1[5] + 2*k2[5] + 2*k3[5] + k4[5])
            y6_arr[node, idx] = y6 + (dt / 6.0) * (k1[6] + 2*k2[6] + 2*k3[6] + k4[6])
            y7_arr[node, idx] = y7 + (dt / 6.0) * (k1[7] + 2*k2[7] + 2*k3[7] + k4[7])
            y8_arr[node, idx] = y8 + (dt / 6.0) * (k1[8] + 2*k2[8] + 2*k3[8] + k4[8])
            y9_arr[node, idx] = y9 + (dt / 6.0) * (k1[9] + 2*k2[9] + 2*k3[9] + k4[9])
    
    return (y0_arr, y1_arr, y2_arr, y3_arr, y4_arr,
            y5_arr, y6_arr, y7_arr, y8_arr, y9_arr)


@numba.njit
def euler_integration(
    y0_arr, y1_arr, y2_arr, y3_arr, y4_arr,
    y5_arr, y6_arr, y7_arr, y8_arr, y9_arr,
    t, dt, startind, N,
    A, B, G, a, b, g,
    C1, C2, C3, C4, C5, C6, C7,
    p_mean, p_sigma,
    e0, v0, r, sigmoid_type_str,
    Cmat, K_gl, Dmat_ndt
):
    """Euler integration for Wendling model (faster but less accurate)."""
    
    sigmoid_type = 0 if sigmoid_type_str == "wendling2002" else 1
    
    for i in range(len(t)):
        idx = startind + i
        
        for node in range(N):
            y0 = y0_arr[node, idx - 1]
            y1 = y1_arr[node, idx - 1]
            y2 = y2_arr[node, idx - 1]
            y3 = y3_arr[node, idx - 1]
            y4 = y4_arr[node, idx - 1]
            y5 = y5_arr[node, idx - 1]
            y6 = y6_arr[node, idx - 1]
            y7 = y7_arr[node, idx - 1]
            y8 = y8_arr[node, idx - 1]
            y9 = y9_arr[node, idx - 1]
            
            # Euler-Maruyama: white noise term must be scaled by sqrt(dt)
            p_t = p_mean + p_sigma * np.random.randn() * np.sqrt(dt)
            
            coupling_input = 0.0
            if N > 1:
                for j in range(N):
                    if Cmat[node, j] > 0:
                        delay_idx = idx - 1 - Dmat_ndt[node, j]
                        if delay_idx >= 0:
                            delayed_output = compute_sigmoid(
                                y1_arr[j, delay_idx] - y2_arr[j, delay_idx] - y3_arr[j, delay_idx],
                                e0, v0, r, sigmoid_type
                            )
                            coupling_input += K_gl * Cmat[node, j] * delayed_output
            
            dy = derivatives_wendling(
                y0, y1, y2, y3, y4, y5, y6, y7, y8, y9,
                A, B, G, a, b, g, C1, C2, C3, C4, C5, C6, C7,
                p_t, e0, v0, r, sigmoid_type, coupling_input
            )
            
            y0_arr[node, idx] = y0 + dt * dy[0]
            y1_arr[node, idx] = y1 + dt * dy[1]
            y2_arr[node, idx] = y2 + dt * dy[2]
            y3_arr[node, idx] = y3 + dt * dy[3]
            y4_arr[node, idx] = y4 + dt * dy[4]
            y5_arr[node, idx] = y5 + dt * dy[5]
            y6_arr[node, idx] = y6 + dt * dy[6]
            y7_arr[node, idx] = y7 + dt * dy[7]
            y8_arr[node, idx] = y8 + dt * dy[8]
            y9_arr[node, idx] = y9 + dt * dy[9]
    
    return (y0_arr, y1_arr, y2_arr, y3_arr, y4_arr,
            y5_arr, y6_arr, y7_arr, y8_arr, y9_arr)


# ==================== SUCCESS VERSION: Direct from working code ====================
# This is the proven-stable Euler-Maruyama implementation

@njit(cache=True, fastmath=True)
def _sigm_fast(v, e0, v0, r):
    """Fast sigmoid for numba."""
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


@njit(cache=True, fastmath=True)
def _integrate_wendling_simple(y0, n_steps, dt,
                                A, a, B, b, G, g,
                                C, C1, C2, C3, C4, C5, C6, C7,
                                e0, v0, r, p_mean, p_sigma):
    """
    Proven-stable Euler-Maruyama integration from your successful code.
    Units: dt in seconds, a/b/g in 1/s (not 1/ms).
    """
    ys = np.zeros((10, n_steps), dtype=np.float64)
    y = y0.copy()

    for k in range(n_steps):
        y0_, y1, y2, y3, y4, y5, y6, y7, y8, y9 = y
        
        # Gaussian white noise with sqrt(dt) scaling (Euler-Maruyama)
        xi_t = np.random.normal(0.0, 1.0)
        p_t = p_mean + p_sigma * xi_t * np.sqrt(dt)
        
        # Derivatives (exactly as in working code)
        dy0 = y5
        dy5 = A * a * _sigm_fast(y1 - y2 - y3, e0, v0, r) - 2.0 * a * y5 - a * a * y0_
        
        dy1 = y6
        dy6 = A * a * (C2 * _sigm_fast(C1 * y0_, e0, v0, r) + p_t) - 2.0 * a * y6 - a * a * y1
        
        dy2 = y7
        dy7 = B * b * (C4 * _sigm_fast(C3 * y0_, e0, v0, r)) - 2.0 * b * y7 - b * b * y2
        
        dy3 = y8
        dy8 = G * g * (C7 * _sigm_fast((C5 * y0_ - C6 * y4), e0, v0, r)) - 2.0 * g * y8 - g * g * y3
        
        dy4 = y9
        dy9 = B * b * (_sigm_fast(C3 * y0_, e0, v0, r)) - 2.0 * b * y9 - b * b * y4
        
        # Euler update
        y0_ += dt * dy0; y1 += dt * dy1; y2 += dt * dy2; y3 += dt * dy3; y4 += dt * dy4
        y5 += dt * dy5; y6 += dt * dy6; y7 += dt * dy7; y8 += dt * dy8; y9 += dt * dy9
        y[0]=y0_; y[1]=y1; y[2]=y2; y[3]=y3; y[4]=y4; y[5]=y5; y[6]=y6; y[7]=y7; y[8]=y8; y[9]=y9
        
        for i in range(10):
            ys[i, k] = y[i]

    return ys


def timeIntegration_simple(params):
    """
    Single-node integration using proven-stable Euler-Maruyama.
    Use this for validation tests.
    """
    dt_ms = params["dt"]
    duration_ms = params["duration"]
    dt = dt_ms / 1000.0  # Convert to seconds
    duration = duration_ms / 1000.0
    n_steps = int(duration / dt)
    
    y = np.zeros(10, dtype=np.float64)
    
    if params.get("seed") is not None:
        np.random.seed(params["seed"])
    
    # Parameters in 1/s
    a = params["a"] * 1000.0
    b = params["b"] * 1000.0
    g = params["g"] * 1000.0
    
    ys = _integrate_wendling_simple(
        y0=y, n_steps=n_steps, dt=dt,
        A=params["A"], a=a, B=params["B"], b=b, G=params["G"], g=g,
        C=params["C"], C1=params["C1"], C2=params["C2"], C3=params["C3"], C4=params["C4"],
        C5=params["C5"], C6=params["C6"], C7=params["C7"],
        e0=params["e0"], v0=params["v0"], r=params["r"],
        p_mean=params["p_mean"], p_sigma=params["p_sigma"],
    )

    t = np.arange(0, duration_ms, dt_ms)[:n_steps]
    return (t, *[ys[i][np.newaxis, :] for i in range(10)])
