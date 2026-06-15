"""
preprocess package
------------------
EEG preprocessing pipeline for the PhysioNet Motor Movement/Imagery dataset.

Modules
-------
config      : all tunable parameters
utils       : logging setup, directory helpers, timing
loader      : recursive EDF discovery and loading
filters     : bandpass + notch filtering
ica         : ICA fitting and artefact removal
epochs      : bad-channel detection, epoching, rejection
visualizer  : raw / filtered / clean comparison plots, ICA plots
pipeline    : end-to-end per-subject orchestration
"""
