# aind-ephys-deepinterpolation-inference

Code Ocean capsule that applies a trained **DeepInterpolation** model to a
Neuropixels recording and writes the denoised recording, for use as a step in the
[AIND ephys pipeline](https://github.com/AllenNeuralDynamics/aind-ephys-pipeline)
and [hybrid benchmark](https://github.com/AllenNeuralDynamics/aind-ephys-hybrid-benchmark).

It mirrors the IO contract of
[`aind-ephys-compress`](https://github.com/AllenNeuralDynamics/aind-ephys-compress):
it reads the `*job*.json` produced by `aind-ephys-job-dispatch` (or hybrid
generation), loads each recording with SpikeInterface, denoises it, saves a Zarr
recording to `../results`, and rewrites the job config to point at the denoised
recording.

## Placement in the pipeline

The bundled model is trained on **raw** AP-band traces (per-channel z-scored, no
CMR or software filter), so it operates in the raw domain and runs *before*
preprocessing — the same slot `compress` occupies:

```
hybrid-generation -> [ deepinterpolation ] -> preprocessing -> spike-sorting
```

## Parameters

| flag | default | meaning |
|------|---------|---------|
| `--checkpoint` | bundled `code/checkpoints/best_model.pt` | trained `di_ephys` checkpoint |
| `--device` | `cuda` | torch device |
| `--batch-size` | `256` | frames per forward pass |
| `--chunk-duration` | `1s` | SpikeInterface save chunk size |
| `--norm-sample-seconds` | `60` | seconds sampled to estimate per-channel z-score stats |

## Model

The default checkpoint is the "fold" geometry with the three-frame SUPPORT hole
(base width 32), the efficient champion from the architecture search
(held-out validation L1 0.2942, +34% over a linear baseline). Inference rebuilds
the probe grid from the recording's channel locations; `load_state_dict` fails
loudly if the probe geometry differs from training (NP1 384-channel).

`di_ephys/model.py` and `di_ephys/dataset.py` are copied verbatim from the
[training repo](https://github.com/AllenNeuralDynamics/aind-ephys-deepinterpolation).
