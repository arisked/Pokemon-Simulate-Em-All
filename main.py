# main.py

import copy
import random
import torch
from typing import List, Dict, Any

from rl_models import DQNAgent
from pokemon_models import Pokemon
from battle_engine import execute_turn
from pokemon_loader import load_pokemon_list

def run_battle(agent: DQNAgent, pokemon1_real: Pokemon, pokemon2_real: Pokemon, training: bool = True) -> bool:
    """
    Simulates a battle between two Pokémon using a Deep Q-Network (DQN) agent.

    Args:
        agent (DQNAgent): The agent that decides the actions for pokemon1.
        pokemon1_real (Pokemon): The first Pokémon (controlled by the agent).
        pokemon2_real (Pokemon): The second Pokémon (opponent, controlled randomly).
        training (bool): Whether the agent is in training mode (default is True).

    Returns:
        bool: True if pokemon1 wins the battle, False otherwise.
    """
    # Deep copy the original Pokémon to avoid modifying the input objects
    pokemon1 = copy.deepcopy(pokemon1_real)
    pokemon2 = copy.deepcopy(pokemon2_real)

    # Initialize stats for both Pokémon
    pokemon1.init_stat()
    pokemon2.init_stat()
    
    # Get the initial state from the agent
    state = agent.get_state(pokemon1, pokemon2)
    turn_count = 0
    reward = 0.0

    # Battle loop: continues until one Pokémon faints or the turn count reaches 100
    while pokemon1.battle_stats['hp'] > 0 and pokemon2.battle_stats['hp'] > 0 and turn_count < 100:
        reward -= 0.5  # Penalize the agent for long battles
        action = agent.get_action(state, pokemon1.moves)  # Agent selects an action
        
        # Set selected moves for both Pokémon
        pokemon1.selected_move = pokemon1.moves[action]
        pokemon2.selected_move = random.choice(pokemon2.moves)
        
        # Execute turn and get turn logs
        log, turn_count, turn_info1, turn_info2 = execute_turn(pokemon1, pokemon2, turn_count)

        # Update turn information for reward calculation
        turn_info1['max_hp'] = pokemon1.max_stats['hp']
        turn_info1['attacker_type'] = pokemon1.type
        turn_info2['max_hp'] = pokemon2.max_stats['hp']
        turn_info2['attacker_type'] = pokemon2.type

        # Get the next state
        next_state = agent.get_state(pokemon1, pokemon2)

        # Training mode: update reward and experience replay memory
        if training:
            reward += calculate_reward(turn_info1)
            done = pokemon1.battle_stats['hp'] <= 0 or pokemon2.battle_stats['hp'] <= 0
            agent.memory.add(state, action, 0.0, next_state, done)

        state = next_state
    
    # Win/Loss reward adjustment
    if pokemon1.battle_stats['hp'] > 0:
        reward += 10  # Reward for winning
    else:
        reward -= 10  # Penalty for losing

    # Finalize experience for the battle
    agent.memory.end_battle(reward)

    return pokemon1.battle_stats['hp'] > 0


def calculate_reward(turn_info: Dict[str, Any]) -> float:
    """
    Calculates the reward for the agent based on the battle turn information.

    Args:
        turn_info (Dict[str, Any]): Dictionary containing turn-related information.

    Returns:
        float: Calculated reward for the agent.
    """
    reward = 0
    initial_state = turn_info['initial_state']
    final_state = turn_info['final_state']
    initial_state_enemy = turn_info['initial_state_enemy']
    final_state_enemy = turn_info['final_state_enemy']
    move = turn_info['move_used']

    # Calculate reward based on damage dealt
    damage_dealt = initial_state_enemy['hp'] - final_state_enemy['hp']
    reward += damage_dealt * 0.01

    # Adjust reward based on type effectiveness
    multiplier_calculation = lambda x: (2 * (x - 1)) * (x >= 0 and x < 1) + (0.5 * x - 0.5) * (x >= 1 and x <= 2) + (0.125 * (x - 2) + 0.5) * (x > 2 and x <= 6)
    reward += multiplier_calculation(turn_info['type_effectiveness'])

    # Bonus reward if the move's type matches the attacker's type
    if move.type in turn_info['attacker_type']:
        reward += 1

    return reward


def train_agent(agent: DQNAgent, pokemon1: Pokemon, pokemon_list: List[Pokemon], num_episodes: int, save_interval: int):
    """
    Trains a DQN agent by simulating multiple battles.

    Args:
        agent (DQNAgent): The DQN agent to be trained.
        pokemon1 (Pokemon): The Pokémon controlled by the agent.
        pokemon_list (List[Pokemon]): A list of all available Pokémon.
        num_episodes (int): The number of episodes to train the agent.
        save_interval (int): The interval for saving the model checkpoints.
    """
    for episode in range(num_episodes):
        # Sample a subset of Pokémon for each training episode
        for pokemon2 in random.sample(pokemon_list, min(len(pokemon_list), 10)):  
            if pokemon1 != pokemon2:
                run_battle(agent, pokemon1, pokemon2, training=True)
            
            # Train the agent with the current batch
            agent.train(agent.batch_size)

        # Perform soft updates periodically
        if episode % 100 == 0:
            agent.update_temperature()

        # Save model checkpoints at specified intervals
        if episode % save_interval == 0:
            agent.save_model(---path to save---)
            torch.save(agent.model.state_dict(), f"gen 1/rl_gen1/model/model_{pokemon1.name}_checkpoint_{episode}.pth")

        # Logging
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed. Exploration rate: {agent.temperature:.4f}")

        # Update exploration temperature
        if episode % 2 == 0:
            agent.update_temperature()


if __name__ == "__main__":
    # Load the Pokémon list from the specified file path
    pokemon_list = load_pokemon_list(---path to pokemon.xlsx---)
    
    if len(pokemon_list) < 2:
        print("Not enough Pokémon loaded. Please check your data file.")
    else:
        state_size = 70  # Set the size of the state vector 
        num_episodes = 5000  # Define the number of episodes for training
        save_interval = 1000  # Interval for saving model checkpoints
        hidden_size = 64  # Number of neurons in the neural network

        # Train the agent for each Pokémon
        for pokemon1 in pokemon_list:
            action_size = len(pokemon1.moves)  # Define the action space size
            print(action_size)
            agent = DQNAgent(state_size, action_size, hidden_size)  # Initialize the DQN agent

            print(f"Training agent with Pokémon: {pokemon1.name}\n")

            # Train the agent
            train_agent(agent, pokemon1, pokemon_list, num_episodes, save_interval)

            print(f"Training and evaluation completed for Pokémon: {pokemon1.name}")

        print("Training and evaluation completed for all Pokémon")
