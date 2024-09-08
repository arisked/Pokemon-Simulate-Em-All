# Pokémon: Simulate 'Em All
This project extends the [Pokémon Battle Engine](https://github.com/arisked/Pokemon-The-Engine.git) with reinforcement learning capabilities, allowing AI agents to learn and improve their battle strategies over time, to search for the best efficient moves a pokémon can use.

# Features

All features from the original Pokémon Battle Engine:

- Turn-based battle system
- Pokémon with stats, levels, and types
- Move system with various effects
- Status conditions (e.g., sleep, paralysis, burn)
- Stat stage changes
- Type effectiveness calculations
- Critical hit system
- Random elements for realistic battle outcomes


New Reinforcement Learning features:

- Deep Q-Network (DQN) agent for Pokémon battling
- Experience replay for efficient learning
- Customizable neural network architecture
- Temperature-based exploration strategy
- Model saving and loading capabilities

# Main Components
## From the original Battle Engine:

- pokemon_models.py: Core classes for Pokémon and Moves
- pokemon_loader.py: Functions to load Pokémon and move data
- battle_engine.py: Battle logic and mechanics

## New RL components:

- rl_models.py: Implements the DQN agent, neural network, and replay buffer
- main.py: Orchestrates the training and evaluation of the RL agent

# Usage
To train an RL agent:

1. Ensure you have the required dependencies installed (pandas, openpyxl, PyTorch, etc.).
2. Prepare your Pokémon data in an Excel file (pokemon.xlsx).
3. Edit the main.py file to specify the location of your pokemon.xlsx file:
```
def main():
    pokemons = load_pokemon_list(---pokemon.xlsx location---)
...
```
4. Run main.py to start the training process:
```
python main.py
```
# Configuration
You can adjust various hyperparameters in the main.py and rl_models.py files:

- Number of training episodes
- Neural network architecture (hidden layer sizes)
- Learning rate
- Discount factor
- Exploration temperature and decay rate
- Replay buffer size
- Batch size for training
- Reward policy

# Future Improvements

- Implement more advanced RL algorithms (e.g., PPO, A3C)
- Add support for multi-agent learning


# License and Copyright
The software is licensed under the MIT license. See the LICENSE file for full copyright and license text. The short version is that you can do what you like with the code, as long as you say where you got it.

This repository includes data extracted from the Pokémon series of video games. All of it is the intellectual property of Nintendo, Creatures, inc., and GAME FREAK, inc. and is protected by various copyrights and trademarks. The author believes that the use of this intellectual property for a fan reference is covered by fair use — the use is inherently educational, and the software would be severely impaired without the copyrighted material.

That said, any use of this library and its included data is at your own legal risk.
