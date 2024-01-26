# AeroWrenchPlanner

## Introduction

This repository supports our research on enhancing quadrotor trajectory generation in the presence of aerodynamic wrenches, such as those from payloads. This method uses a tracking cost function, derived from simulation data, to guide the trajectory planning, resulting in improved tracking accuracy and safer maneuvering. Our findings, validated through simulations and Crazyflie hardware experiments, demonstrate reduced tracking costs and avoidance of controller saturation and catastrophic outcomes during aggressive maneuvers.

## Paper

[arXiv](https://arxiv.org/abs/2401.04960)

## Environment Setup

### Dependencies

- **JAX**: Follow the installation guide [here](https://jax.readthedocs.io/en/latest/installation.html).
- **RotorPy**: Installation instructions available on its GitHub page [here](https://github.com/spencerfolk/rotorpy).

## Usage Instructions

### Data Collection

- Data is uploaded via Box, including raw data for each trajectory in CSV format (state, flat thrust, etc.) saved in `/trial_data`.
- Cost and coefficients for each trajectory are stored in `data.csv`.
- To collect data with different penalty values, use `data_collection_diff_rho.py`. This script varies the penalty value (`robust_c`) to observe its impact on model performance.

### Training Models

- Models with different penalty values are also uploaded via Box, titled `rho-[penalty value]`, e.g., `rho-1`.
- Run `learning/training.py` to train your model.

## Evaluation

### Inference

- Use `learning/inference.py` to run inference on trained model
- Use `learning/inference_with_drag_comp.py` to perform inference incorporating drag compensation on the controller.

### Visualization

- Use `tools/plot_traj.py` to plot trajectories for single trial data.
- Use `tools/final_boxplot.py` for generating boxplots showing the Relative Cost Ratio of your approach compared to Minsnap.
- Use `tools/inference_once.py` to run inference and visualize the results.

## Citation

If this repo helps your research, please cite our paper at:

```
@misc{zhang2024change,
      title={Why Change Your Controller When You Can Change Your Planner: Drag-Aware Trajectory Generation for Quadrotor Systems},
      author={Hanli Zhang and Anusha Srikanthan and Spencer Folk and Vijay Kumar and Nikolai Matni},
      year={2024},
      eprint={2401.04960},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

If you use the RotorPy simulator, please cite their paper at:

```
@article{folk2023rotorpy,
  title={RotorPy: A Python-based Multirotor Simulator with Aerodynamics for Education and Research},
  author={Folk, Spencer and Paulos, James and Kumar, Vijay},
  journal={arXiv preprint arXiv:2306.04485},
  year={2023}
}
```

## video

TODO
