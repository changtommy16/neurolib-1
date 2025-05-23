# ECT Simulation with Wilson-Cowan Model

This module provides functionality to simulate Electroconvulsive Therapy (ECT) sessions using the Wilson-Cowan neural mass model in Neurolib. The implementation allows for time-varying parameter changes to simulate different phases of ECT:

1. **Pre-ECT Baseline Phase**: Normal brain activity before the intervention.
2. **Stimulus Phase**: Brief, strong excitatory input representing the electrical stimulus.
3. **Ictal Phase**: Seizure activity characterized by increased excitation and neural synchronization.
4. **Post-ictal Suppression**: Period of increased inhibition after the seizure.

## Usage

There are two ways to use this functionality:

### 1. Using the ECTSimulation Class (Higher-level API)

```python
from neurolib.models.wc import WCModel
from neurolib.utils.stimulus import ECTSimulation

# Create a Wilson-Cowan model
model = WCModel()
model.params['duration'] = 6000  # ms

# Create an ECT simulation
ect = ECTSimulation(duration=model.params['duration'], dt=model.params['dt'])

# Add the stimulus phase
ect.add_stimulus_phase(
    start=500,       # ms
    duration=100,    # ms
    amplitude=2.0    # strength of stimulus
)

# Add the ictal (seizure) phase
ect.add_ictal_phase(
    start=600,           # ms
    duration=2000,       # ms
    exc_increase=1.5,    # increase excitatory baseline
    excexc_increase=16.0 # increase E-E connectivity
)

# Add the post-ictal suppression phase
ect.add_postictal_phase(
    start=2600,          # ms
    duration=3000,       # ms
    inh_increase=3.0,    # increase inhibitory baseline
    inhexc_increase=20.0 # increase I-E connectivity
)

# Run the model with the ECT simulation
model.run(time_varying_params=ect.get_time_varying_parameters())
```

### 2. Using TimeVaryingParameters Class (Lower-level API)

For more detailed control over parameter changes:

```python
from neurolib.models.wc import WCModel
from neurolib.utils.stimulus import TimeVaryingParameters

# Create a model
model = WCModel()
model.params['duration'] = 6000  # ms

# Create a TimeVaryingParameters object
tvp = TimeVaryingParameters(duration=model.params['duration'], dt=model.params['dt'])

# Add parameter changes with different timing and shapes
# Stimulus phase - brief excitatory input
tvp.add_parameter_change(
    param_name='exc_ext',
    start_time=500,
    end_time=600,
    target_value=2.0,
    mode='step'
)

# Ictal phase - increased excitatory connectivity
tvp.add_parameter_change(
    param_name='c_excexc',
    start_time=600,
    end_time=2600,
    target_value=16.0,
    mode='step'
)

# Post-ictal phase - increased inhibition with gradual onset
tvp.add_parameter_change(
    param_name='inh_ext_baseline',
    start_time=2600,
    end_time=5600,
    target_value=3.0,
    mode='ramp',
    shape_param=0.5
)

# Run the model with the time-varying parameters
model.run(time_varying_params=tvp)
```

## Available Parameter Change Modes

The `TimeVaryingParameters` class supports several modes for changing parameters:

- `step`: Instantaneous change to target value
- `ramp`: Linear change from initial to target value
- `exponential`: Exponential approach to target value
- `gaussian`: Gaussian-shaped change (useful for temporary increases/decreases)
- `sine`: Sinusoidal change (useful for oscillatory patterns)

## Parameters That Can Be Modified

The following Wilson-Cowan model parameters can be modified during a simulation:

- `exc_ext_baseline`: Baseline excitatory external input
- `inh_ext_baseline`: Baseline inhibitory external input
- `exc_ext`: Time-varying excitatory external input
- `inh_ext`: Time-varying inhibitory external input
- `c_excexc`: Excitatory to excitatory connection strength
- `c_excinh`: Excitatory to inhibitory connection strength
- `c_inhexc`: Inhibitory to excitatory connection strength
- `c_inhinh`: Inhibitory to inhibitory connection strength

## Example

See the full example in `example_ect_simulation.py`.

## References

- Wilson, H.R., & Cowan, J.D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. Biophysical Journal, 12(1), 1-24.
- Papadopoulos, L., et al. (2020). Relations between large-scale brain connectivity and effects of regional stimulation depend on collective dynamical state. arXiv. 