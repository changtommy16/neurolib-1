# Wendling Model Implementation Notes (Arrais 2019)

## Implementation Guide Compliance ✅

This implementation follows the detailed recipe from Arrais et al. (2019) paper.

### State Vector Structure (10 variables)

```python
state = [y₀, y₁, y₂, y₃, y₄, z₀, z₁, z₂, z₃, z₄]
       = [y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]]
```

Where:
- `y₀-y₄`: Potentials (activity levels) of 5 neuron populations
- `z₀-z₄`: Velocities (rates of change) = `ẏ₀-ẏ₄`

### Parameters (Default Values)

```python
# Synaptic gains & time constants
A=5.0,    a=100.0   # Excitatory
B=25.0,   b=25.0    # Slow inhibitory
G=15.0,   g=500.0   # Fast inhibitory

# Connectivity (scaled by C=135.0)
C1=1.0, C2=0.8, C3=0.25, C4=0.25, C5=0.3, C6=0.1, C7=0.8

# Sigmoid parameters
e0=2.5, v0=6.0, r=0.56

# Stimulation offset
S=0.0  # Set to non-zero to apply stimulation

# Noise input
p_mean=90.0   # Mean firing rate (Hz)
p_sigma=30.0  # Standard deviation (Hz) - Arrais 2019 uses 30

# Simulation
dt=0.0001      # Time step (0.1 ms)
duration=50.0  # Total simulation time (s)
```

### The Five Second-Order ODEs (Arrais 2019 Formulation)

#### 1. Pyramidal Output (y₀)
```
ÿ₀(t) = Aa·Sig(S + y₁ − y₂ − y₃) − 2a·ẏ₀(t) − a²·y₀(t)
```
- Input: Combined signal from excitatory (y₁) and inhibitory (y₂, y₃) populations
- Output: Main EEG-like signal

#### 2. Excitatory Input (y₁) - **KEY: No C2 coefficient**
```
ÿ₁(t) = Aa·{p(t) + Sig(S + C₁·y₀)} − 2a·ẏ₁(t) − a²·y₁(t)
```
- Input: Background noise p(t) + feedback from pyramidal cells
- **Note**: Direct addition `p(t) + Sig(...)`, no C2 multiplier

#### 3. Slow Inhibition (y₂)
```
ÿ₂(t) = Bb·C₄·Sig(S + C₃·y₀) − 2b·ẏ₂(t) − b²·y₂(t)
```
- Feedback from pyramidal cells via C₃, weighted by C₄

#### 4. Fast Inhibition (y₃)
```
ÿ₃(t) = Gg·C₇·Sig(S + C₅·y₀ − y₄) − 2g·ẏ₃(t) − g²·y₃(t)
```
- Feedback from pyramidal cells (C₅·y₀) and inhibitory interneurons (y₄)

#### 5. Inhibitory Interneurons (y₄)
```
ÿ₄(t) = Bb·C₆·Sig(S + C₃·y₀) − 2b·ẏ₄(t) − b²·y₄(t)
```
- Drives fast inhibition, feedback from pyramidal cells

### Sigmoid Function

```python
Sig(v) = 2·e₀ / (1 + exp(r·(v₀ - (S + v))))
```

Where:
- `S + v`: Arrais 2019 adds stimulation offset S inside sigmoid
- Returns firing rate (Hz)

### Noise Generation

```python
p(t) = p_mean + p_sigma · ξ(t) · √dt
```

Where:
- `ξ(t)`: Standard Gaussian random variable N(0,1)
- `√dt`: Scaling for Euler-Maruyama integration of SDEs

### Integration Loop (Euler Method)

For each time step `dt`:

```python
# A. Get current state
y₀, y₁, y₂, y₃, y₄ = state[0:5]  # Potentials
z₀, z₁, z₂, z₃, z₄ = state[5:10] # Velocities

# B. Calculate sigmoid outputs
sig_0 = Sig(S + y₁ - y₂ - y₃)
sig_1 = Sig(S + C₁·y₀)
sig_2 = Sig(S + C₃·y₀)
sig_3 = Sig(S + C₅·y₀ - y₄)
sig_4 = sig_2  # Same input as sig_2

# C. Calculate accelerations (dz/dt)
dz0_dt = A·a·sig_0 - 2·a·z₀ - a²·y₀
dz1_dt = A·a·(p_t + sig_1) - 2·a·z₁ - a²·y₁
dz2_dt = B·b·C₄·sig_2 - 2·b·z₂ - b²·y₂
dz3_dt = G·g·C₇·sig_3 - 2·g·z₃ - g²·y₃
dz4_dt = B·b·C₆·sig_4 - 2·b·z₄ - b²·y₄

# D. Update state (IMPORTANT: Order matters!)
# Step 1: Update velocities first
z₀ += dz0_dt · dt
z₁ += dz1_dt · dt
z₂ += dz2_dt · dt
z₃ += dz3_dt · dt
z₄ += dz4_dt · dt

# Step 2: Update potentials using NEW velocities
y₀ += z₀ · dt
y₁ += z₁ · dt
y₂ += z₂ · dt
y₃ += z₃ · dt
y₄ += z₄ · dt

# E. Store y₀ for output (EEG-like signal)
output[time_index] = y₀
```

## Key Differences from Other Formulations

1. **Sigmoid Offset S**: Arrais 2019 adds `S` inside sigmoid: `Sig(S + v)` not `Sig(v)`
2. **No C2 in y₁ equation**: Direct sum `p(t) + Sig(...)`, not `C2·Sig(...) + p(t)`
3. **Update Order**: Velocities first, then potentials (as per Arrais guide)
4. **Noise Scaling**: Uses `√dt` for proper SDE integration
5. **Higher Noise**: p_sigma=30 (Arrais 2019) vs 2 (other implementations)

## Outputs

Main output: `v = y₁ - y₂ - y₃` (pyramidal membrane potential)
- This represents the EEG-like signal
- Alternative: Use `y₀` directly for pyramidal output

Firing rate: `rate = Sig(v)` using the sigmoid function

## Testing Different Epileptic Patterns

Adjust parameters A, B, G to generate different patterns:
- Type 1 (Rhythmic spikes): A=2, B=24, G=20
- Type 2 (Spike-wave): A=5, B=23, G=20
- Type 3 (Fast activity): A=5.5, B=20, G=20
- Type 4 (Polyspikes): A=6, B=9, G=20
- Type 5 (Slow waves): A=5, B=1, G=20

## References

Arrais, D. M. C., Laurin, N., Rocha, J. A., Aguiar, J. A., & Rosa, R. S. (2019). 
Identification of effective stimulation parameters to abort epileptic seizures in 
a neural mass model. Journal of Computational Neuroscience.
