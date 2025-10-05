# **`README.md`**

# Continuous-Time Reinforcement Learning for Asset-Liability Management

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.23280-b31b1b.svg)](https://arxiv.org/abs/2509.23280)
[![Conference](https://img.shields.io/badge/Conference-ICAIF%20'25-9cf)](https://icaif.acm.org/2025/)
[![Year](https://img.shields.io/badge/Year-2025-purple)](https://github.com/chirindaopensource/continuous_time_reinforcement_learning_asset_liability_management)
[![Discipline](https://img.shields.io/badge/Discipline-Quantitative%20Finance%20%7C%20RL-00529B)](https://github.com/chirindaopensource/continuous_time_reinforcement_learning_asset_liability_management)
[![Primary Data](https://img.shields.io/badge/Data-Simulated%20SDE-lightgrey)](https://github.com/chirindaopensource/continuous_time_reinforcement_learning_asset_liability_management)
[![Core Method](https://img.shields.io/badge/Method-Continuous--Time%20RL-orange)](https://github.com/chirindaopensource/continuous_time_reinforcement_learning_asset_liability_management)
[![Key Concepts](https://img.shields.io/badge/Concepts-LQ%20Control%20%7C%20Soft%20Actor--Critic-red)](https://github.com/chirindaopensource/continuous_time_reinforcement_learning_asset_liability_management)
[![Baselines](https://img.shields.io/badge/Baselines-SAC%20%7C%20PPO%20%7C%20DDPG%20%7C%20CPPI-blueviolet)](https://github.com/chirindaopensource/continuous_time_reinforcement_learning_asset_liability_management)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0086D1?style=flat)](https://gymnasium.farama.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
--

**Repository:** `https://github.com/chirindaopensource/continuous_time_rl_for_alm`

**Owner:** 2025 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2025 paper entitled **"Continuous-Time Reinforcement Learning for Asset-Liability Management"** by:

*   Yilie Huang

The project provides a complete, end-to-end computational framework for replicating the paper's novel continuous-time reinforcement learning approach to ALM. It delivers a modular, auditable, and extensible pipeline that executes the entire research workflow: from rigorous, reproducible experimental setup and parallelized simulation to comprehensive statistical analysis and the generation of all publication-quality figures and tables.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callables](#key-callables)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a Python implementation of the methodologies presented in the 2025 paper "Continuous-Time Reinforcement Learning for Asset-Liability Management." The core of this repository is the iPython Notebook `continuous_time_reinforcement_learning_asset_liability_management_draft.ipynb`, which contains a comprehensive suite of functions to replicate the paper's findings, from initial data validation to the final generation of all analytical tables and figures.

The paper introduces a novel model-free, continuous-time reinforcement learning (RL) algorithm for the Asset-Liability Management (ALM) problem. It frames the problem as a Linear-Quadratic (LQ) control task and develops a soft actor-critic method with adaptive exploration to dynamically manage the surplus deviation between assets and liabilities. This codebase operationalizes this framework, allowing users to:
-   Rigorously validate and manage the entire experimental configuration.
-   Systematically generate reproducible, randomized market scenarios based on a stochastic differential equation (SDE) model.
-   Execute large-scale, parallelized simulations comparing the proposed ALM-RL agent against six distinct baselines.
-   Perform comprehensive statistical analysis using non-parametric tests to validate performance claims.
-   Conduct a full suite of robustness analyses, including hyperparameter sensitivity, market parameter stress tests, and discretization analysis.

## Theoretical Background

The implemented methods are grounded in stochastic optimal control, reinforcement learning, and numerical methods for SDEs.

**1. ALM as a Linear-Quadratic (LQ) Control Problem:**
The core of the problem is to control the surplus deviation, `x(t)`, from a target. Its dynamics are modeled by the SDE:
$$
dx(t) = (A x(t) + B u(t))dt + (C x(t) + D u(t))dW(t)
$$
where `u(t)` is the control action. The objective is to maximize the expected value of a quadratic functional that penalizes deviations over a finite horizon `[0, T]`:
$$
\max_{u} \mathbb{E}\left[ \int_{0}^{T} -\frac{1}{2}Qx(t)^2 dt - \frac{1}{2}Hx(T)^2 \right]
$$

**2. Continuous-Time Soft Actor-Critic:**
Since the market parameters `A, B, C, D` are unknown, a model-free RL approach is used. The paper develops a continuous-time soft actor-critic algorithm based on an entropy-regularized objective:
$$
J(t, x; \pi) = \mathbb{E}\left[ \int_{t}^{T} \left(-\frac{1}{2}Qx(s)^2 + \gamma p(s)\right) ds - \frac{1}{2}Hx(T)^2 \Big| x(t)=x \right]
$$
where `p(s)` is the entropy of the stochastic policy `π`.

**3. Key Algorithmic Features:**
-   **Parametric Forms:** Based on LQ theory, the value function `J` is parameterized as a quadratic function of `x`, and the policy `π` is a Gaussian distribution whose mean is linear in `x`.
-   **Adaptive Exploration:** The policy's variance (actor exploration) is learned via policy gradient.
-   **Scheduled Exploration:** The entropy temperature `γ` (critic exploration) follows a deterministic, decaying schedule.
-   **Update Rules:** The agent learns via discretized versions of continuous-time temporal difference and policy gradient updates (Eqs. 16, 17, 18 in the paper).

## Features

The provided iPython Notebook (`continuous_time_reinforcement_learning_asset_liability_management_draft.ipynb`) implements the full research pipeline, including:

-   **Modular, Multi-Task Architecture:** The entire pipeline is broken down into 13 distinct, modular tasks, each with its own orchestrator function, covering validation, setup, simulation, analysis, and reporting.
-   **Configuration-Driven Design:** All experimental parameters are managed in an external `config.yaml` file, allowing for easy customization and replication without code changes.
-   **Multi-Algorithm Support:** Complete, from-scratch implementations of the proposed **ALM-RL** agent and six baselines: **DCPPI**, **ACS**, **MBP**, **SAC**, **PPO**, and **DDPG**.
-   **Rigorous Reproducibility:** A multi-level seeding protocol ensures bitwise reproducibility of market scenarios and isolates stochastic streams for fair agent comparison.
-   **Parallelized Execution:** The main experimental pipeline is designed for parallel execution across multiple CPU cores, dramatically reducing the time required for the 200 independent runs.
-   **Comprehensive Analysis Suite:** Implements the full statistical analysis from the paper, including moving average smoothing, terminal performance extraction, and one-sided Wilcoxon signed-rank tests.
-   **Robustness Analysis Module:** Includes a full suite of post-hoc analyses to test hyperparameter sensitivity, robustness to extreme market conditions, and sensitivity to SDE discretization.
-   **Automated Reporting:** Programmatic generation of all key tables and figures from the paper.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Validation (Task 1):** Ingests and rigorously validates the `config.yaml` for structural, mathematical, and logical consistency.
2.  **Setup (Task 2):** Establishes the deterministic seeding hierarchy for the entire experiment.
3.  **Initialization (Task 3):** Generates the 200 randomized market scenarios and the corresponding initial parameters for all agents.
4.  **Agent & Environment Implementation (Tasks 4-7):** Provides complete, professional-grade implementations of all agents and the SDE environment.
5.  **Execution (Task 8):** Runs the main simulation pipeline in parallel, executing 20,000 episodes for each of the 7 agents across all 200 market scenarios.
6.  **Metrics & Analysis (Tasks 9-10):** Processes the raw simulation data to compute smoothed learning curves, terminal performance, and the final p-value matrix.
7.  **Visualization (Task 11):** Generates the final, publication-quality plots and summary tables.
8.  **Orchestration & Robustness (Tasks 12-13):** Provides top-level orchestrators to run the main pipeline and the additional robustness analyses.

## Core Components (Notebook Structure)

The `continuous_time_reinforcement_learning_asset_liability_management_draft.ipynb` notebook is structured as a logical pipeline with modular orchestrator functions for each of the major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callables

The project is designed around a single, top-level user-facing interface function:

-   **`main`:** This master orchestrator function, located in the final section of the notebook, runs the entire automated research pipeline from end-to-end. It can be configured to run the main reproduction experiment, the robustness analyses, or both. A single call to this function reproduces the entire computational portion of the project.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `numpy`, `pandas`, `scipy`, `pyyaml`, `torch`, `gymnasium`, `matplotlib`, `seaborn`, `tqdm`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/continuous_time_rl_for_alm.git
    cd continuous_time_reinforcement_learning_asset_liability_management
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install numpy pandas scipy pyyaml torch gymnasium matplotlib seaborn tqdm
    ```

## Input Data Structure

The pipeline is driven by a single `config.yaml` file. No external datasets are required, as the market scenarios are procedurally generated based on the parameters within this file.

## Usage

The `continuous_time_reinforcement_learning_asset_liability_management_draft.ipynb` notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell of the notebook, which calls the top-level `main` orchestrator:

```python
# Final cell of the notebook or in a main.py script

# Load the configuration from the YAML file.
STUDY_INPUTS = load_config('config.yaml')

# Run the entire study (reproduction and robustness analysis).
final_artifacts = main(
    study_params=STUDY_INPUTS,
    run_reproduction=True,
    run_robustness=True,
    num_workers=8  # Adjust based on available CPU cores
)

# The `final_artifacts` dictionary will contain the key results DataFrames.
```

## Output Structure

The `main` function creates one or two output directories (`alm_rl_reproduction_output/` and `alm_rl_robustness_output/`) with the following structure:

```
output_directory/
│
├── data/
│   ├── seed_table.csv
│   ├── market_params_table.csv
│   ├── alm_rl_initial_table.csv
│   ├── baselines_initial_table.csv
│   ├── raw_results.npy
│   ├── learning_curves.csv
│   ├── terminal_performance.csv
│   └── p_value_matrix.csv
│
├── figures/
│   ├── figure1_learning_curves.png
│   └── figure2_p_value_heatmap.png
│
└── tables/
    └── table1_summary_statistics.html
```

## Project Structure

```
continuous_time_rl_for_alm/
│
├── continuous_time_reinforcement_learning_asset_liability_management_draft.ipynb # Main implementation notebook
├── config.yaml                                                                   # Master configuration file
├── requirements.txt                                                              # Python package dependencies
├── LICENSE                                                                       # MIT license file
└── README.md                                                                     # This documentation file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can easily modify all experimental parameters, including the number of runs/episodes, SDE parameter distributions, agent hyperparameters, and evaluation settings, without altering the core Python code.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Alternative SDE Models:** Integrating more complex market models, such as those with stochastic volatility (e.g., Heston model) or jumps.
-   **Multi-Asset Formulations:** Extending the state and action spaces to handle a portfolio of multiple assets.
-   **Automated Hyperparameter Tuning:** Wrapping the pipeline with a hyperparameter optimization library (e.g., Optuna) to automatically find the best settings for the ALM-RL agent.
-   **Real-World Data Application:** Adapting the framework to use historical financial data by first estimating the SDE parameters from time series data.

## License

This project is licensed under the MIT License.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@inproceedings{huang2025continuous,
  author    = {Huang, Yilie},
  title     = {Continuous-Time Reinforcement Learning for Asset-Liability Management},
  booktitle = {Proceedings of the 6th ACM International Conference on AI in Finance},
  series    = {ICAIF '25},
  year      = {2025},
  publisher = {ACM},
  note      = {arXiv:2509.23280}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2025). A Professional-Grade Implementation of the "Continuous-Time RL for ALM" Framework.
GitHub repository: https://github.com/chirindaopensource/continuous_time_rl_for_alm
```

## Acknowledgments

-   Credit to **Yilie Huang** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, including **NumPy, Pandas, SciPy, PyTorch, Gymnasium, Matplotlib, and Jupyter**, whose work makes complex computational analysis accessible and robust.

--

*This README was generated based on the structure and content of `continuous_time_reinforcement_learning_asset_liability_management_draft.ipynb` and follows best practices for research software documentation.*
