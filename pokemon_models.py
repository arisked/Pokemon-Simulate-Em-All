# pokemon_models.py

import random
from typing import List, Dict, Optional, Union

class Move:
    def __init__(self, name: str = "", type: str = "", category: str = "", power: Optional[int] = None, 
                 accuracy: Optional[int] = None, pp: int = 0, effect: Optional[List[Dict[str, int | str | float]]] = None):
        self._name = name
        self._type = type
        self._category = category
        self._power = power
        self._accuracy = accuracy
        self._pp = pp
        self._effect = effect if effect is not None else []

    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value

    @property
    def type(self) -> str:
        return self._type
    
    @type.setter
    def type(self, value: str) -> None:
        if not value:
            raise ValueError("Type cannot be empty")
        self._type = value

    @property
    def category(self) -> str:
        return self._category
    
    @category.setter
    def category(self, value: str) -> None:
        if not value:
            raise ValueError("Category cannot be empty")
        self._category = value

    @property
    def power(self) -> Optional[int]:
        return self._power
    
    @power.setter
    def power(self, value: Optional[int]) -> None:
        if value is not None and value < 0:
            raise ValueError("Power cannot be negative")
        self._power = value

    @property
    def accuracy(self) -> Optional[int]:
        return self._accuracy
    
    @accuracy.setter
    def accuracy(self, value: Optional[int]) -> None:
        if value is not None and (value < 0 or value > 100):
            raise ValueError("Accuracy must be between 0 and 100")
        self._accuracy = value

    @property
    def pp(self) -> int:
        return self._pp
    
    @pp.setter
    def pp(self, value: int) -> None:
        if value < 0:
            raise ValueError("PP cannot be negative")
        self._pp = value

    @property
    def effect(self) -> List[Dict[str, int | str | float]]:
        return self._effect
    
    def has_effect(self, effect_name: str) -> bool:
        """
        Checks if the move has a specific effect.

        Args:
            effect_name (str): The name of the effect to check for.

        Returns:
            bool: True if the move has the specified effect, False otherwise.
        """
        return any(effect.get('effect') == effect_name for effect in self._effect)
   
    def find_related_value(self, key: str, value: str, target_key: str) -> Optional[Union[str, int, float]]:
        """
        Finds the value associated with `target_key` in the dictionary where `key` equals `value`.

        Args:
            key (str): The key to search for in each dictionary.
            value (str): The value that the `key` should match.
            target_key (str): The key whose value is to be returned.

        Returns:
            Optional[Union[str, int, float]]: The value associated with `target_key` if found, otherwise None.
        """
        for effect in self._effect:
            if effect.get(key) == value:
                return effect.get(target_key, None)
        return None

    def __str__(self) -> str:
        return (f"Move(name='{self._name}', type='{self._type}', category='{self._category}', "
                f"power={self._power}, accuracy={self._accuracy}, pp={self._pp}, effect={self._effect})")

class Pokemon:
    def __init__(self, name: str, types: List[str], hp: int, attack: int, defense: int,
                 special_attack: int, special_defense: int, speed: int, moves_list: List[str], level: int):
        # Basic Information
        self._name = name
        self._type = types
        self._level = level
        self._moves_list = moves_list

        # Moves
        self._moves: List[Move] = []
        self._selected_move: Optional[Move] = None
        self._last_move: Optional[Move] = None

        # Stats
        self._stat_stages: Dict[str, int] = {stat: 0 for stat in ['atk', 'def', 'sp_atk', 'sp_def', 'spd', 'eva', 'acc']}
        self._stat_multipliers: Dict[str, float] = {stat: 1 for stat in ['atk', 'def', 'sp_atk', 'sp_def', 'spd']} # this handle multiplier for status that gives multiplier effect on stat, but not necessarily affect stat stages
        self._base_stats: Dict[str, int] = {
            'hp': hp, 'atk': attack, 'def': defense,
            'sp_atk': special_attack, 'sp_def': special_defense, 'spd': speed
        }
        self._max_stats = {}
        self._battle_stats = {}

        # Battle-related
        self._statuses: Dict[str, int] = {}
        self._last_damage: int = 0
        self._can_move: bool = True

    # Basic Information
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value
    
    @property
    def level(self) -> int:
        return self._level
    
    @level.setter
    def level(self, value: int) -> None:
        if value <= 0:
            raise ValueError("Level must be positive")
        self._level = value
    
    @property
    def type(self) -> List[str]:
        return self._type
    
    @type.setter
    def type(self, value: List[str]) -> None:
        if not value:
            raise ValueError("Type list cannot be empty")
        self._type = value

    @property
    def moves_list(self) -> List[str]:
        return self._moves_list

    @moves_list.setter
    def moves_list(self, value: List[str]) -> None:
        if not value:
            raise ValueError("Moves list cannot be empty")
        self._moves_list = value

    # Moves
    @property
    def moves(self) -> List[Move]:
        return self._moves
    
    @moves.setter
    def moves(self, value: List[Move]) -> None:
        if not all(isinstance(move, Move) for move in value):
            raise ValueError("Invalid move in moves list")
        self._moves = value

    @property
    def selected_move(self) -> Optional[Move]:
        return self._selected_move
    
    @selected_move.setter
    def selected_move(self, value: Optional[Move]) -> None:
        if not isinstance(value, Optional[Move]):
            raise ValueError("Invalid move")
        self._selected_move = value

    @property
    def last_move(self) -> Optional[Move]:
        return self._last_move
    
    @last_move.setter
    def last_move(self, value: Optional[Move]) -> None:
        if not isinstance(value, Optional[Move]):
            raise ValueError("Invalid move")
        self._last_move = value

    # Stats
    @property
    def base_stats(self) -> Dict[str, int]:
        return self._base_stats

    @property
    def max_stats(self) -> Dict[str, int]:
        return self._max_stats

    @property
    def battle_stats(self) -> Dict[str, int]:
        return self._battle_stats
    
    @battle_stats.setter
    def battle_stats(self, value: Dict[str, int]) -> None:
        required_keys = {'hp', 'atk', 'def', 'sp_atk', 'sp_def', 'spd'}
        if not all(key in value for key in required_keys):
            raise ValueError("Stats must include all required stats")
        self._battle_stats = value
        
    @property
    def stat_stages(self) -> Dict[str, int]:
        return self._stat_stages

    @stat_stages.setter
    def stat_stages(self, value: Dict[str, int]) -> None:
        self._stat_stages.update(value)
        required_keys = {'atk', 'def', 'sp_atk', 'sp_def', 'spd', 'eva', 'acc'}
        if not all(key in self._stat_stages for key in required_keys):
            raise ValueError("Stat stages must include all required stats")
        self.battle_stats = self._calculate_battle_stats()
    
    @property
    def stat_multipliers(self) -> Dict[str, float]:
        return self._stat_multipliers

    @stat_multipliers.setter
    def stat_multipliers(self, value: Dict[str, float]) -> None:
        self._stat_multipliers.update(value)
        required_keys = {'atk', 'def', 'sp_atk', 'sp_def', 'spd'}
        if not all(key in self._stat_multipliers for key in required_keys):
            raise ValueError("Stat stages must include all required stats")
        self.battle_stats = self._calculate_battle_stats()

    # Battle-related
    @property
    def statuses(self) -> Dict[str, int]:
        return self._statuses

    @statuses.setter
    def statuses(self, value: Dict[str, int]) -> None:
        if not isinstance(value, dict):
            raise ValueError("Statuses must be a dictionary")
        if not all(isinstance(k, str) and isinstance(v, int) for k, v in value.items()):
            raise ValueError("Status keys must be strings and values must be integers")
        self._statuses = value

    @property
    def last_damage(self) -> int:
        return self._last_damage

    @last_damage.setter
    def last_damage(self, value: int) -> None:
        if value < 0:
            raise ValueError("Damage cannot be negative")
        self._last_damage = value

    @property
    def can_move(self) -> bool:
        return self._can_move

    @can_move.setter
    def can_move(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("Can move must be a boolean value")
        self._can_move = value

    # Stat
    def init_stat(self):
        self._max_stats = self._calculate_stats()
        self._battle_stats = self._calculate_battle_stats(True)
        
    def update_stat_stage(self, stat: str, stage_change: int) -> None:
        """
        Updates the stat stage for a specific stat.

        Args:
            stat (str): The stat to update.
            stage_change (int): The change in stage value.

        Raises:
            ValueError: If the stat is invalid.
        """
        if stat not in self._stat_stages:
            raise ValueError(f"Invalid stat stage: {stat}")
        self._stat_stages[stat] += stage_change
        self._stat_stages[stat] = max(-6, min(self._stat_stages[stat], 6))
        self.battle_stats = self._calculate_battle_stats()

    def reset_stat_stages(self) -> None:
        """
        Resets all stat stages to 0 and recalculates battle stats.
        """
        for stat in self._stat_stages:
            self._stat_stages[stat] = 0
        self.battle_stats = self._calculate_battle_stats()
        
    def update_stat_multiplier(self, stat: str, factor: float) -> None:
        """
        Updates the multiplier for a specific stat.

        Args:
            stat (str): The stat to update.
            factor (float): The factor to multiply the stat by. Must be either 0.5 or 2.0.

        Raises:
            ValueError: If the stat is invalid or the factor is not 0.5 or 2.0.
        """
        if stat not in self._stat_multipliers:
            raise ValueError(f"Invalid stat: {stat}")
        if factor not in [0.5, 2.0]:
            raise ValueError("Factor must be either 0.5 or 2.0")
        self._stat_multipliers[stat] *= factor
        self.battle_stats = self._calculate_battle_stats()

    def reset_stat_multiplier(self, stat: str) -> None:
        """
        Resets the multiplier for a specific stat to 1.

        Args:
            stat (str): The stat to reset.

        Raises:
            ValueError: If the stat is invalid.
        """
        if stat not in self._stat_multipliers:
            raise ValueError(f"Invalid stat: {stat}")
        self._stat_multipliers[stat] = 1
        self.battle_stats = self._calculate_battle_stats()

    def _calculate_stats(self) -> Dict[str, int]:
        """
        Calculates the Pokémon's stats based on base stats, IVs, and EVs.

        Returns:
            Dict[str, int]: A dictionary containing the calculated stats.
        """
        evs = self._generate_random_evs()
        ivs = self._generate_random_ivs()
        stats: Dict[str, int] = {}
        for stat, base in self._base_stats.items():
            if stat == 'hp':
                stats[stat] = self._calculate_hp(base, ivs[stat], evs[stat])
            else:
                stats[stat] = self._calculate_other_stat(base, ivs[stat], evs[stat])
        return stats

    def _calculate_battle_stats(self, initialize: bool = False) -> Dict[str, int]:
        """
        Calculates the Pokémon's battle stats, taking into account stat stages and multipliers.

        Args:
            initialize (bool, optional): If True, initializes HP to max value. Defaults to False.

        Returns:
            Dict[str, int]: A dictionary containing the calculated battle stats.
        """
        stat_stage_multiplier: List[float] = [2/8, 2/7, 2/6, 2/5, 2/4, 2/3, 2/2, 3/2, 4/2, 5/2, 6/2, 7/2, 8/2]
        stats: Dict[str, int] = {}
        for stat, base in self.max_stats.items():
            if stat == 'hp':
                if initialize:
                    stats[stat] = base
                else:
                    stats[stat] = self.battle_stats[stat]
            else:
                stats[stat] = int(base * stat_stage_multiplier[self.stat_stages[stat] + 6] * self.stat_multipliers[stat])
        return stats

    def _calculate_hp(self, base: int, iv: int, ev: int) -> int:
        """
        Calculates the HP stat.

        Args:
            base (int): Base HP stat.
            iv (int): Individual Value for HP.
            ev (int): Effort Value for HP.

        Returns:
            int: The calculated HP stat.
        """
        return ((2 * base + iv + ev // 4) * self._level) // 100 + self._level + 10

    def _calculate_other_stat(self, base: int, iv: int, ev: int) -> int:
        """
        Calculates stats other than HP.

        Args:
            base (int): Base stat value.
            iv (int): Individual Value for the stat.
            ev (int): Effort Value for the stat.

        Returns:
            int: The calculated stat value.
        """
        return int((((2 * base + iv + ev // 4) * self._level) // 100 + 5) * 1.0)

    @staticmethod
    def _generate_random_evs() -> Dict[str, int]:
        """
        Generate the random maximum Effort Values (EVs) for all stats, adhering to modern Pokémon rules where the total EVs cannot exceed 510, with a maximum of 255 for each stat.

        Returns:
            Dict[str, int]: A dictionary containing randomly generated EVs for each stat.
        """
        evs: Dict[str, int] = {stat: 0 for stat in ['hp', 'atk', 'def', 'sp_atk', 'sp_def', 'spd']}
        ev_total: int = 510
        while ev_total > 0:
            stat: str = random.choice(list(evs.keys()))
            increment: int = min(random.randint(0, 255), ev_total)
            evs[stat] += increment
            ev_total -= increment
        return evs

    @staticmethod
    def _generate_random_ivs() -> Dict[str, int]:
        """
        Generates random Individual Values (IVs) for all stats.

        Returns:
            Dict[str, int]: A dictionary containing randomly generated IVs for each stat.
        """
        return {stat: random.randint(0, 31) for stat in ['hp', 'atk', 'def', 'sp_atk', 'sp_def', 'spd']}

    # Status
    def apply_status(self, status_type: str, duration: int) -> None:
        """
        Applies a status condition to the Pokémon.

        Args:
            status_type (str): The type of status condition to apply.
            duration (int): The duration of the status condition.

        Raises:
            ValueError: If the status_type is not a string or duration is not an integer.
        """
        if not isinstance(status_type, str) or not isinstance(duration, int):
            raise ValueError("Invalid status type or duration")
        self._statuses[status_type] = duration
   
    def has_status(self, status_type: str) -> bool:
        """
        Checks if the Pokémon currently has the given status type.

        Args:
            status_type (str): The status type to check for.

        Returns:
            bool: True if the Pokémon has the specified status, False otherwise.
        """
        return status_type in self._statuses
   
    def has_non_volatile_status(self) -> bool:
        """
        Checks if the Pokémon currently has a non-volatile status condition.
        Non-volatile status conditions include:
        - Paralyze
        - Sleep
        - Freeze
        - Badly poisoned
        - Burn
        - Poison

        Returns:
            bool: True if the Pokémon has any non-volatile status condition, False otherwise.
        """
        non_volatile_status = ['paralyze', 'sleep', 'freeze', 'badly_poison', 'burn', 'poison']
        return any(status in self.statuses for status in non_volatile_status)
   
    def get_status_duration(self, status_type: str) -> int:
        """
        Gets the remaining duration of a specific status condition.

        Args:
            status_type (str): The status type to check.

        Returns:
            int: The remaining duration of the status, or 0 if the status is not present.
        """
        return self._statuses.get(status_type, 0)
   
    def add_status_duration(self, status_type: str, addition_amount: int = 1) -> None:
        """
        Increases the duration of a specific status condition.

        Args:
            status_type (str): The status type to modify.
            addition_amount (int, optional): The amount to increase the duration by. Defaults to 1.
        """
        if status_type in self._statuses:
            self._statuses[status_type] += addition_amount

    def deduct_status_duration(self, status_type: str, deduction_amount: int = 1) -> None:
        """
        Decreases the duration of a specific status condition.

        Args:
            status_type (str): The status type to modify.
            deduction_amount (int, optional): The amount to decrease the duration by. Defaults to 1.
        """
        if status_type in self._statuses:
            self._statuses[status_type] -= deduction_amount

    def remove_expired_statuses(self) -> None:
        """
        Removes all status conditions with a duration of 0 or less.
        """
        self._statuses = {status: duration for status, duration in self._statuses.items() if duration > 0}

    def remove_status(self, status_type: str) -> None:
        """
        Removes a specific status condition from the Pokémon.

        Args:
            status_type (str): The status type to remove.
        """
        if status_type in self._statuses:
            del self._statuses[status_type]

    def __str__(self) -> str:
        return f"Pokemon(name='{self._name}', type={self._type}, level={self._level})"

