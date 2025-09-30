# -*- coding: utf-8 -*-
"""
Signal Classification for Wendling Model
Based on Arrais et al. (2019) classification criteria

Five rhythm types:
1. Background activity
2. Spikes
3. Slow rhythmic activity
4. Alpha rhythm
5. Fast activity
"""
import numpy as np
from scipy.signal import find_peaks, welch


def classify_wendling_signal(t, v, fs, verbose=False):
    """
    Classify Wendling model output signal according to Arrais 2019 criteria.
    
    Classification Rules:
    1. Compute dominant frequency (freq of max power in PSD)
    2. If dominant_freq > 8 Hz → "alpha" or "fast" (rejected for seizure studies)
    3. If dominant_freq <= 8 Hz:
       - If PSD magnitude at dominant freq < 0.1 mV²/Hz → "background"
       - Else, count peaks in 10s window:
         * If peaks < 10 → "spikes"
         * If peaks >= 10 → "slow_rhythmic"
    
    Parameters
    ----------
    t : array
        Time vector (seconds)
    v : array
        Signal (EEG-like output from Wendling model)
    fs : float
        Sampling frequency (Hz)
    verbose : bool
        If True, print classification details
        
    Returns
    -------
    classification : str
        One of: "background", "spikes", "slow_rhythmic", "alpha", "fast", "other"
    info : dict
        Dictionary with classification details
    """
    
    # Compute Power Spectral Density using Welch's method
    freqs, psd = welch(v, fs=fs, nperseg=min(len(v), int(4*fs)), 
                      noverlap=int(2*fs))
    
    # Find dominant frequency (frequency with maximum power)
    dominant_idx = np.argmax(psd)
    dominant_freq = freqs[dominant_idx]
    dominant_power = psd[dominant_idx]
    
    # Initialize classification info
    info = {
        'dominant_frequency': dominant_freq,
        'dominant_power': dominant_power,
        'sampling_frequency': fs,
        'signal_duration': t[-1] - t[0],
    }
    
    # Rule 1: Check if dominant frequency > 8 Hz (alpha or fast activity)
    if dominant_freq > 8.0:
        if dominant_freq <= 13.0:
            classification = "alpha"
        else:
            classification = "fast"
        
        if verbose:
            print(f"Dominant frequency {dominant_freq:.2f} Hz > 8 Hz")
            print(f"Classification: {classification}")
        
        info['classification'] = classification
        return classification, info
    
    # Rule 2: For signals with dominant_freq <= 8 Hz
    # Check if background (low power)
    if dominant_power < 0.1:
        classification = "background"
        
        if verbose:
            print(f"Dominant frequency {dominant_freq:.2f} Hz <= 8 Hz")
            print(f"Dominant power {dominant_power:.6f} mV²/Hz < 0.1")
            print(f"Classification: background")
        
        info['classification'] = classification
        return classification, info
    
    # Rule 3: Distinguish between spikes and slow rhythmic
    # Count peaks in a 10-second portion
    analysis_duration = min(10.0, t[-1] - t[0])
    analysis_samples = int(analysis_duration * fs)
    
    # Take middle portion for stability
    start_idx = (len(v) - analysis_samples) // 2
    end_idx = start_idx + analysis_samples
    v_segment = v[start_idx:end_idx]
    
    # Detect peaks (use height threshold based on std)
    threshold = np.mean(v_segment) + 0.5 * np.std(v_segment)
    peaks, _ = find_peaks(v_segment, height=threshold, distance=int(0.1*fs))
    num_peaks = len(peaks)
    
    info['num_peaks_in_10s'] = num_peaks
    info['analysis_duration'] = analysis_duration
    
    if num_peaks < 10:
        classification = "spikes"
        if verbose:
            print(f"Dominant frequency {dominant_freq:.2f} Hz <= 8 Hz")
            print(f"Dominant power {dominant_power:.6f} mV²/Hz >= 0.1")
            print(f"Number of peaks in {analysis_duration:.1f}s: {num_peaks} < 10")
            print(f"Classification: spikes")
    else:
        classification = "slow_rhythmic"
        if verbose:
            print(f"Dominant frequency {dominant_freq:.2f} Hz <= 8 Hz")
            print(f"Dominant power {dominant_power:.6f} mV²/Hz >= 0.1")
            print(f"Number of peaks in {analysis_duration:.1f}s: {num_peaks} >= 10")
            print(f"Classification: slow_rhythmic")
    
    info['classification'] = classification
    return classification, info


# Known parameter sets from Arrais 2019 Fig. 2
KNOWN_PATTERNS = {
    'background': {'A': 2.0, 'B': 24.0, 'G': 20.0},
    'spikes': {'A': 5.0, 'B': 23.0, 'G': 20.0},
    'slow_rhythmic': {'A': 5.5, 'B': 20.0, 'G': 20.0},
    'alpha': {'A': 6.0, 'B': 9.0, 'G': 20.0},
    'fast': {'A': 5.0, 'B': 1.0, 'G': 20.0},
}
