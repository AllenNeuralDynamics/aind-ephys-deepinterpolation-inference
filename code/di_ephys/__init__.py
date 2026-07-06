"""DeepInterpolation for Neuropixels ecephys (PyTorch, 1D probe-axis port).

Ported from the voltage-imaging DeepInterpolationPyTorch codebase
(AllenNeuralDynamics/DeepInterpolationPyTorch). The voltage model stacks context
frames as channels of a 2D U-Net that predicts a held-out center frame. For
extracellular ephys the natural spatial axis is the probe's channel layout, so
here the U-Net is 1D (Conv1d over the 384 channels) with the pre/post context
time samples as input channels.
"""
