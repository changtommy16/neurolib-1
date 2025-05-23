"""
Example script for simulating ECT (Electroconvulsive Therapy) phases using the WC model
with the new time-varying parameter functionality.

This example shows how to simulate:
1. Baseline activity (pre-ECT)
2. Stimulus phase (brief excitatory input)
3. Ictal phase (seizure with increased excitation)
4. Post-ictal suppression phase (increased inhibition)
"""

import numpy as np
import matplotlib.pyplot as plt
from neurolib.models.wc import WCModel
from neurolib.utils.stimulus import TimeVaryingParameters, ECTSimulation

# Create a Wilson-Cowan model
model = WCModel()

# Set basic model parameters
model.params['duration'] = 6000  # 6 seconds simulation
model.params['dt'] = 0.1  # integration step
model.params['sigma_ou'] = 0.02  # noise level

# Create an ECT simulation
ect = ECTSimulation(duration=model.params['duration'], dt=model.params['dt'])

# Define the phases of ECT
# Stimulus phase: brief, strong excitatory input
ect.add_stimulus_phase(
    start=500,         # start at 500ms
    duration=100,      # 100ms duration
    amplitude=2.0,     # amplitude of the stimulus
    parameter='exc_ext'  # apply to the excitatory external input
)

# Ictal phase: increased excitatory activity
ect.add_ictal_phase(
    start=600,           # start right after stimulus
    duration=2000,       # 2 seconds of seizure
    exc_increase=1.5,    # increase excitatory baseline
    excexc_increase=16.0,  # increase excitatory-to-excitatory connectivity
    mode='step'          # step change (sudden onset)
)

# Post-ictal phase: increased inhibition
ect.add_postictal_phase(
    start=2600,          # start after ictal phase
    duration=3000,       # 3 seconds of post-ictal suppression
    inh_increase=3.0,    # increase inhibitory baseline
    inhexc_increase=20.0,  # increase inhibitory-to-excitatory connectivity
    mode='ramp',         # gradual onset
    shape_param=0.5      # shape parameter for the ramp
)

# Run the model with the ECT simulation
model.run(time_varying_params=ect.get_time_varying_parameters())

# Plot the results
plt.figure(figsize=(12, 8))

# Plot excitatory activity
plt.subplot(3, 1, 1)
plt.plot(model.t, model.exc.T)
plt.title('Excitatory Activity during ECT Simulation')
plt.ylabel('Activity')
plt.axvspan(500, 600, color='yellow', alpha=0.3, label='Stimulus')
plt.axvspan(600, 2600, color='red', alpha=0.3, label='Ictal')
plt.axvspan(2600, 5600, color='blue', alpha=0.3, label='Post-ictal')
plt.legend()

# Plot inhibitory activity
plt.subplot(3, 1, 2)
plt.plot(model.t, model.inh.T)
plt.title('Inhibitory Activity during ECT Simulation')
plt.ylabel('Activity')
plt.axvspan(500, 600, color='yellow', alpha=0.3)
plt.axvspan(600, 2600, color='red', alpha=0.3)
plt.axvspan(2600, 5600, color='blue', alpha=0.3)

# Plot phase portrait (exc vs inh)
plt.subplot(3, 1, 3)
plt.plot(model.exc.T, model.inh.T)
plt.title('Phase Portrait (Excitatory vs Inhibitory Activity)')
plt.xlabel('Excitatory Activity')
plt.ylabel('Inhibitory Activity')

plt.tight_layout()
plt.savefig('ect_simulation.png')
plt.show()

# Demonstrate manual parameter changes using TimeVaryingParameters directly
# This provides more flexibility for complex parameter changes

print("\nDemonstrating manual parameter changes with TimeVaryingParameters:\n")

# Create a TimeVaryingParameters object
tvp = TimeVaryingParameters(duration=3000, dt=0.1)

# Add multiple parameter changes with different timing and shapes
tvp.add_parameter_change(
    param_name='exc_ext_baseline',
    start_time=500,
    end_time=1000,
    target_value=1.5,
    mode='exponential',
    shape_param=0.05
)

tvp.add_parameter_change(
    param_name='c_inhexc',
    start_time=1500,
    end_time=2500,
    target_value=20.0,
    mode='sine',
    shape_param=0.5
)

# Create a second model for this demonstration
model2 = WCModel()
model2.params['duration'] = 3000
model2.params['sigma_ou'] = 0.02
model2.run(time_varying_params=tvp)

# Plot the results of the second demonstration
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(model2.t, model2.exc.T)
plt.title('Excitatory Activity with Complex Parameter Changes')
plt.ylabel('Activity')
plt.axvspan(500, 1000, color='green', alpha=0.3, label='Exponential excitatory baseline')
plt.axvspan(1500, 2500, color='purple', alpha=0.3, label='Sinusoidal inhibitory-to-excitatory')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(model2.t, model2.inh.T)
plt.title('Inhibitory Activity with Complex Parameter Changes')
plt.xlabel('Time (ms)')
plt.ylabel('Activity')
plt.axvspan(500, 1000, color='green', alpha=0.3)
plt.axvspan(1500, 2500, color='purple', alpha=0.3)

plt.tight_layout()
plt.savefig('complex_parameter_changes.png')
plt.show()

print("\nSimulations completed. Results saved as 'ect_simulation.png' and 'complex_parameter_changes.png'.") 