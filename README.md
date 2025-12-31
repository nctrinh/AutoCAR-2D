# Autonomous Car 2D Simulation

A 2D autonomous vehicle simulation project that implements various planning, control, and reinforcement learning algorithms for autonomous navigation in a 2D environment.

## Features

- **Core Components**: Vehicle dynamics model and 2D map representation with obstacles
- **Planning Algorithms**: A* path planning and RRT (Rapidly-exploring Random Tree) for path generation
- **Control Systems**: PID controller and Pure Pursuit for vehicle control
- **Reinforcement Learning**: Integration with PPO and SAC algorithms using Stable Baselines3
- **Simulation**: Real-time 2D visualization using Pygame and Matplotlib
- **Configuration**: YAML-based configuration system for easy parameter tuning

## Installation

### Prerequisites
- Python 3.12+
- Poetry (for dependency management)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd autonomous_car_2d
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Alternatively, install dependencies manually:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Scripts

All executable scripts are located in the `scripts/` directory. Below are the available scripts and how to run them:

#### Training Reinforcement Learning Models
- **run_train_RL.py**: Train PPO or SAC models for autonomous driving
  ```bash
  python scripts/run_train_RL.py --config config/RL_config.yaml --algorithm PPO --timesteps 100000
  ```
  Options:
  - `--config`: Path to configuration file (default: config/RL_config.yaml)
  - `--algorithm`: RL algorithm to use (PPO or SAC)
  - `--timesteps`: Number of training timesteps

#### Running 2D Simulation
- **run_sim2d.py**: Run the 2D autonomous car simulation
  ```bash
  python scripts/run_sim2d.py --config config/default_config.yaml --max-steps 1000 --fps 30
  ```
  Options:
  - `--config`: Path to configuration file (default: config/default_config.yaml)
  - `--max-steps`: Maximum simulation steps
  - `--fps`: Target frames per second

- **run_sim2d.sh**: Shell script wrapper for running the simulation
  ```bash
  ./scripts/run_sim2d.sh
  ```

#### Evaluating Trained Models
- **run_evaluate_RL.py**: Evaluate trained RL models
  ```bash
  python scripts/run_evaluate_RL.py --model trained_models/ppo/best_model.zip --episodes 10
  ```
  Options:
  - `--model`: Path to trained model file
  - `--episodes`: Number of evaluation episodes

#### Testing Core Components
- **test_core_components.py**: Test vehicle and map core components
  ```bash
  python scripts/test_core_components.py
  ```

- **test_core_components.sh**: Shell script wrapper for testing core components
  ```bash
  ./scripts/test_core_components.sh
  ```

#### Testing Planning and Control
- **test_core_planning_control.py**: Test planning algorithms (A*, RRT) and control systems (PID, Pure Pursuit)
  ```bash
  python scripts/test_core_planning_control.py
  ```

- **test_core_planning_control.sh**: Shell script wrapper for testing planning and control
  ```bash
  ./scripts/test_core_planning_control.sh
  ```

## Project Structure

```
autonomous_car_2d/
├── config/                 # Configuration files
│   ├── default_config.yaml
│   └── RL_config.yaml
├── docs/                   # Documentation
├── logs/                   # Training logs and TensorBoard events
├── maps/                   # Sample map files
├── notebooks/              # Jupyter notebooks for analysis
├── scripts/                # Executable scripts
├── src/                    # Source code
│   ├── core/               # Core components (vehicle, map)
│   ├── control/            # Control algorithms (PID, Pure Pursuit)
│   ├── learning/           # RL environment
│   ├── planning/           # Path planning algorithms (A*, RRT)
│   ├── simulation/         # Simulation and rendering
│   └── utils/              # Utilities (config loader)
├── tests/                  # Unit tests
└── trained_models/         # Saved trained models
    ├── ppo/
    └── sac/
```

## Dependencies

- numpy: Numerical computations
- scipy: Scientific computing
- matplotlib: Plotting and visualization
- pygame: 2D game development library for simulation
- opencv-python: Computer vision tasks
- pyyaml: YAML configuration parsing
- gymnasium: Reinforcement learning environments
- stable-baselines3: RL algorithms (PPO, SAC)
- tqdm: Progress bars
- rich: Rich text and beautiful formatting
- tensorboard: Experiment tracking and visualization

## Configuration

The project uses YAML configuration files located in the `config/` directory:
- `default_config.yaml`: Default simulation parameters
- `RL_config.yaml`: Reinforcement learning training parameters

Modify these files to adjust simulation settings, vehicle parameters, map configurations, and training hyperparameters.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request