# NEATRun: AI Evolution Simulator 🏃‍♂️🤖

NEATRun is an AI-driven side-scrolling game where a neural network learns to navigate obstacles using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm. The AI evolves over generations to improve its performance in avoiding obstacles and achieving higher scores.

## Features

- **Neural Network-Based Learning**: Uses the NEAT algorithm to evolve AI decision-making based on obstacle distance and type.
- **Obstacle Avoidance**: The AI learns to evade both ground and elevated obstacles by predicting movements through neural network outputs.
- **Real-Time Training**: AI performance is evaluated live, with improvements seen across multiple generations.
- **Dynamic Obstacle Spawning**: Randomized obstacle appearance to add difficulty and ensure varied gameplay.
- **Scoring System**: Tracks the generation count, current score, and high score, helping analyze AI performance over time.
- **Visual Feedback**: The game provides a real-time graphical representation of AI players, obstacles, and score metrics.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/neat-run-simulator.git
    ```
2. Navigate to the project folder:
    ```bash
    cd neat-run-simulator
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the simulation:
    ```bash
    python neat_run_sim.py
    ```
2. The AI will control the player and attempt to avoid obstacles while continuously learning.
3. The game will display key metrics such as:
    - **Generation Number**
    - **Current Score**
    - **High Score**
4. The AI evolves automatically, refining its performance over multiple generations.

## Configuration

- The NEAT algorithm is configured using the `config-feedforward` file, defining parameters like population size, mutation rates, and neural network structure.
- Modify this file to adjust training settings for enhanced AI performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the [NEAT-Python](https://neat-python.readthedocs.io/en/latest/) library for enabling evolutionary AI learning.
- Inspired by classic endless runner games but enhanced with AI-driven gameplay mechanics.
