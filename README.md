
# Empirical Wavelet Transform for Seizure Detection

**Sanju Kumar**  
**2001EE62**  
[sanjuphogat8238@gmail.com](mailto:sanjuphogat8238@gmail.com)  
Electrical and Electronics Department  
Indian Institute of Technology Patna  

**Supervised by:**  
**Dr. Maheshkumar H. Kolekar**  
Electrical and Electronics Department  
Indian Institute of Technology Patna  

---

## Project Overview

This project applies the Empirical Wavelet Transform (EWT) method to EEG signal data to detect and classify epileptic seizures. The EWT method is an adaptive signal decomposition technique that is particularly effective for analyzing nonlinear and non-stationary signals, such as EEG data. The project utilizes two publicly available EEG datasets, decomposes the signals using EWT, and evaluates the resulting features for seizure detection.

---

## Methodology

### Dataset

The following EEG datasets are used:
1. **University of Bonn EEG Dataset** - Frequently used as a benchmark for seizure detection algorithms.
2. **Neurology and Sleep Centre, Hauz Khas, New Delhi (NSC-ND) EEG Dataset** - EEG data from epilepsy patients.

### EWT Decomposition

The `ewtpy` Python package is used to perform Empirical Wavelet Transform (EWT) on the EEG signals. The EWT method is fully adaptive, defining filter supports based on the spectral distribution of the input signal. The number of modes \(N\) and the detection mode are set by the user, and various regularization options are available for spectrum smoothing.

#### Key Parameters:
- `N`: Number of supports (modes) to decompose the signal into.
- `detect`: Detection mode (`locmax`, `locmaxmin`, `locmaxminf`).
- `reg`: Spectrum regularization method (`none`, `average`, `gaussian`).
- `Fs`: Sampling frequency in Hz (set to 1 if unknown).

### Signal Decomposition and Plotting

The provided Python script loads an EEG signal from a CSV file, applies the EWT, and plots both the original signal and the decomposed modes. Additionally, the script plots the signal's frequency spectrum and the boundaries of the EWT modes.

---

## Python Script

```python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import ewtpy

signal = "sig3.csv" # Replace with your signal file (e.g., sig1.csv, sig2.csv, eeg.csv)
data = np.loadtxt(signal, delimiter=",")
if len(data.shape) > 1:
    f = data[:,0]
else:
    f = data
    
#f = f - np.mean(f)

N = 3 # Number of supports (modes)
detect = "locmax" # Detection mode: locmax, locmaxmin, locmaxminf
reg = 'none' # Spectrum regularization: none, average, gaussian
lengthFilter = 0 # Length of average or gaussian filter (set to 0 if not used)
sigmaFilter = 0 # Sigma of gaussian filter (set to 0 if not used)
Fs = 1 # Sampling frequency in Hz (set to 1 if unknown)

ewt, mfb, boundaries = ewtpy.EWT1D(f, 
                                    N=N,
                                    log=0,
                                    detect=detect, 
                                    completion=0, 
                                    reg=reg, 
                                    lengthFilter=lengthFilter,
                                    sigmaFilter=sigmaFilter)

# Plot original signal and decomposed modes
plt.figure()
plt.subplot(211)
plt.plot(f)
plt.title('Original signal %s' % signal)
plt.subplot(212)
plt.plot(ewt)
plt.title('EWT modes')

# Show boundaries in frequency domain
ff = np.fft.fft(f)
freq = 2 * np.pi * np.arange(0, len(ff)) / len(ff)

if Fs != -1:
    freq = freq * Fs / (2 * np.pi)
    boundariesPLT = boundaries * Fs / (2 * np.pi)
else:
    boundariesPLT = boundaries

ff = abs(ff[:ff.size//2]) # One-sided magnitude
freq = freq[:freq.size//2]

plt.figure()
plt.plot(freq, ff)
for bb in boundariesPLT:
    plt.plot([bb, bb], [0, max(ff)], 'r--')
plt.title('Spectrum partitioning')
plt.xlabel('Hz')
plt.show()
```

### Script Functionality:
- **Signal Loading:** Loads an EEG signal from a specified CSV file.
- **EWT Decomposition:** Decomposes the signal into a specified number of modes using EWT.
- **Plotting:** Visualizes the original signal, the decomposed modes, and the frequency spectrum with marked boundaries.

---

## Conclusion

The Empirical Wavelet Transform (EWT) offers an adaptive and effective way to decompose EEG signals for the purpose of seizure detection. This project highlights the utility of EWT in processing and analyzing EEG data, with potential applications in real-time seizure detection systems.

---

## References

1. Carvalho, Vinícius Rezende et al. “Evaluating five different adaptive decomposition methods for EEG signal seizure detection and classification.” *Biomed. Signal Process. Control.* 62 (2020): 102073.
```
