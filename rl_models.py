# rl_models.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List

from pokemon_models import Pokemon, Move

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a namedtuple for transitions in the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model.
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        """
        Initialize the DQN model.
        
        Args:
            state_size (int): The dimension of the state space.
            action_size (int): The dimension of the action space.
            hidden_size (int): The number of neurons in the hidden layers.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.to(device)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor.
        
        Returns:
            torch.Tensor: Output Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Replay buffer to store and sample experience tuples.
    """
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)
        self.episode_transitions = []

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode has ended.
        """
        transition = Transition(state, action, reward, next_state, done)
        self.episode_transitions.append(transition)

    def end_battle(self, win_loss_reward: float):
        """
        End an episode and store all transitions in the replay buffer.
        
        Args:
            win_loss_reward (float): Reward for winning or losing the battle.
        """
        num_transitions = len(self.episode_transitions)
        if num_transitions > 0:
            reward_per_transition = win_loss_reward

            # Convert episode_transitions to numpy arrays
            states = np.array([t.state for t in self.episode_transitions])
            actions = np.array([t.action for t in self.episode_transitions])
            rewards = np.array([t.reward for t in self.episode_transitions])
            next_states = np.array([t.next_state for t in self.episode_transitions])
            dones = np.array([t.done for t in self.episode_transitions])

            # Add reward_per_transition to all rewards
            rewards += reward_per_transition

            # Create new Transition objects and extend the buffer
            battle_transitions = [Transition(*t) for t in zip(states, actions, rewards, next_states, dones)]
            self.buffer.extend(battle_transitions)

        # Clear episode transitions
        self.episode_transitions = []

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample.
        
        Returns:
            List[Transition]: A list of sampled transitions.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Learning Agent.
    """
    def __init__(self, state_size: int, action_size: int, hidden_size: int, learning_rate: float = 0.001, 
                 discount_factor: float = 0.99, temperature: float = 1.0,
                 temperature_decay: float = 0.999, temperature_min: float = 0.01):
        """
        Initialize the DQN agent.
        
        Args:
            state_size (int): The dimension of the state space.
            action_size (int): The dimension of the action space.
            hidden_size (int): The number of neurons in the hidden layers.
            learning_rate (float): Learning rate for the optimizer.
            discount_factor (float): Discount factor for future rewards.
            temperature (float): Temperature for Boltzmann exploration.
            temperature_decay (float): Decay rate for temperature.
            temperature_min (float): Minimum temperature value.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min

        self.memory = ReplayBuffer(10000)
        self.batch_size = 1024

        self.model = DQN(state_size, action_size, hidden_size)
        self.target_model = DQN(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.update_target_model()

    def update_target_model(self):
        """
        Update the target model with the current model's weights.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, pokemon: Pokemon, opponent: Pokemon) -> np.ndarray:
        """
        Get the state representation for the given Pokémon and opponent.
        
        Args:
            pokemon (Pokemon): The player's Pokémon.
            opponent (Pokemon): The opponent's Pokémon.
        
        Returns:
            np.ndarray: The state representation.
        """
        def get_pokemon_state(p: Pokemon) -> np.ndarray:
            stats = np.array([
                p.battle_stats['hp'],
                p.battle_stats['atk'],
                p.battle_stats['def'],
                p.battle_stats['sp_atk'],
                p.battle_stats['sp_def'],
                p.battle_stats['spd']
            ], dtype=np.float32)

            stats = stats / 2880

            all_types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon', 'Dark', 'Steel', 'Fairy']
            types = np.array([1.0 if t in p.type else 0.0 for t in all_types], dtype=np.float32)
            
            all_statuses = ['paralyze', 'sleep', 'freeze', 'burn', 'poison', 'badly_poison', 'confuse', 'flinch', 'trap', 'recharge', 'seed']
            statuses = np.array([1.0 if status in p.statuses else 0.0 for status in all_statuses], dtype=np.float32)
            
            return np.concatenate([stats, types, statuses])

        pokemon1_state = get_pokemon_state(pokemon)
        pokemon2_state = get_pokemon_state(opponent)
        
        return np.concatenate([pokemon1_state, pokemon2_state])
   
    def get_action(self, state: np.ndarray, available_moves: List[Move]) -> int:
        """
        Get the action to take based on the current state.
        
        Args:
            state (np.ndarray): The current state.
            available_moves (List[Move]): List of available moves.
        
        Returns:
            int: The index of the selected action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.model(state_tensor).squeeze(0)
        probabilities = torch.softmax(q_values / self.temperature, dim=0)
        action = int(torch.multinomial(probabilities, 1).item())

        return action
    
    def train(self, batch_size: int):
        """
        Train the DQN agent using a batch of experiences from the replay buffer.
        
        Args:
            batch_size (int): Number of transitions to use for training.
        """
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_temperature(self):
        """
        Update the exploration temperature.
        """
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)

    def save_model(self, path: str):
        """
        Save the model parameters to a file.
        
        Args:
            path (str): Path to the file where the model will be saved.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """
        Load the model parameters from a file.
        
        Args:
            path (str): Path to the file from which the model will be loaded.
        """
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()
