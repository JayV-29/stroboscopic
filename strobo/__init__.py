"""Stroboscopic sensing: tiny latent-oscillator world models that decide *when* to
sample an expensive sensor, driven by an always-on cheap stream.

Sub-packages
------------
data    loaders for PPG-DaLiA / WESAD / UCI-HAR / MIT-BIH plus a synthetic generator
sim     burst masking and energy accounting
models  oscillator bank, sampling head, decoder, and the baseline policies
export  int8 fixed-point export and op counting
"""

__version__ = "0.1.0"
