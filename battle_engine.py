# battle_engine.py

import random
from typing import Dict, List, Tuple, Any
from pokemon_models import Pokemon, Move
from pokemon_loader import load_pokemon_list

def execute_turn(pokemon1: Pokemon, pokemon2: Pokemon, turn_count: int) -> Tuple[str, int, Dict[str, Any], Dict[str, Any]]:
    turn_count += 1
    log = f"Turn {turn_count}:\n"
    
    # Checking if any of the move is None at this point
    if pokemon1.selected_move is None or pokemon2.selected_move is None:
        raise ValueError("Selected move cannot be None")
    
    # storing moves
    move1 = pokemon1.selected_move
    move2 = pokemon2.selected_move
    
    # Store initial states for RL
    initial_state1 = {
        'hp': pokemon1.battle_stats['hp'],
        'stat_stages': pokemon1.stat_stages.copy(),
        'statuses': set(pokemon1.statuses.keys())
    }
    initial_state2 = {
        'hp': pokemon2.battle_stats['hp'],
        'stat_stages': pokemon2.stat_stages.copy(),
        'statuses': set(pokemon2.statuses.keys())
    }

    # Storing priority if exist or set to 0
    priority1 = int(move1.find_related_value('effect', 'priority', 'amount') or 0)
    priority2 = int(move2.find_related_value('effect', 'priority', 'amount') or 0)

    # print(pokemon1.name, pokemon1.battle_stats, pokemon1.stat_stages, pokemon1.statuses)
    # print(pokemon2.name, pokemon2.battle_stats, pokemon2.stat_stages, pokemon2.statuses)

    if priority1 != priority2:
        # Check for move priority
        first, second = (pokemon1, pokemon2) if priority1 > priority2 else (pokemon2, pokemon1)
    elif pokemon1.battle_stats['spd'] != pokemon2.battle_stats['spd']:
        # If priorities are the same, fall back to speed
        first, second = (pokemon1, pokemon2) if pokemon1.battle_stats['spd'] >= pokemon2.battle_stats['spd'] else (pokemon2, pokemon1)
    else:
        # Tie breaker
        first, second = random.choice([(pokemon1, pokemon2), (pokemon2, pokemon1)])
    
    log += execute_move(first, second, True)
    # Execute second Pokémon's move if it still has HP left
    if first.battle_stats['hp'] > 0 and second.battle_stats['hp'] > 0:
        log += execute_move(second, first)

    # dealing with condition after all pokemon move
    log += apply_end_turn(pokemon1, pokemon2)
    log += apply_end_turn(pokemon2, pokemon1)

    # Collect final states and turn information for RL
    final_state1 = {
        'hp': pokemon1.battle_stats['hp'],
        'stat_stages': pokemon1.stat_stages.copy(),
        'statuses': set(pokemon1.statuses.keys())
    }
    final_state2 = {
        'hp': pokemon2.battle_stats['hp'],
        'stat_stages': pokemon2.stat_stages.copy(),
        'statuses': set(pokemon2.statuses.keys())
    }

    turn_info1 = {
        'initial_state': initial_state1,
        'final_state': final_state1,
        'initial_state_enemy': initial_state2,
        'final_state_enemy': final_state2,
        'move_used': pokemon1.selected_move,
        'type_effectiveness': calculate_type_effectiveness(pokemon1.selected_move.type, pokemon2.type)
    }
    turn_info2 = {
        'initial_state': initial_state2,
        'final_state': final_state2,
        'initial_state_enemy': initial_state1,
        'final_state_enemy': final_state1,
        'move_used': pokemon2.selected_move,
        'type_effectiveness': calculate_type_effectiveness(pokemon2.selected_move.type, pokemon1.type)
    }

    return log, turn_count, turn_info1, turn_info2

def execute_move(attacker: Pokemon, defender: Pokemon, is_first_move: bool = False) -> str:
    log = ""

    # Applying start of the turn effects
    log += apply_start_move(attacker, defender)
    
    move = attacker.selected_move if attacker.can_move else None

    # Save the last move used by the attacker (including None)
    attacker.last_move = move

    # Check if the attacker has a move selected
    if move is None:
        log += f"{attacker.name} has no move to use!\n"
        return log
    
    # Define effect handlers
    effect_handlers = {
        'damage': handle_damage,
        'heal': handle_heal,
        'recoil': handle_recoil,
        'flinch': handle_flinch,
        'hits': handle_multi_hit,
        'sleep': handle_sleep,
        'stage': handle_stage,
        'crit_ratio': handle_crit_ratio,
        'stage_reset': handle_stage_reset,
        'badly_poison': handle_badly_poison,
        'random_level_damage': handle_random_level_damage,
        'counter': handle_counter,
        'faint': handle_faint,
        'burn': handle_burn,
        'absorb': handle_absorb,
        'seed': handle_seed,
        'recharge': handle_recharge,
        'freeze': handle_freeze,
        'double_hit': handle_double_hit,
        'trap': handle_trap,
        'poison': handle_poison,
        'level_damage': handle_level_damage,
        'multi_hit': handle_multi_hit,
        'paralyze': handle_paralyze,
        'confuse': handle_confuse,
        'half_hp': handle_half_hp,
    }

    log += f"{attacker.name} uses {move.name}!\n"

    # Check if the move hit or not
    if move_hit(attacker, defender, move):
        # Process move effects
        for effect in move.effect:
            effect_type = str(effect.get('effect'))
            if effect_type in effect_handlers:
                log += effect_handlers[effect_type](attacker, defender, effect, move, is_first_move)
    else:
        log += "The move missed!\n"
        if move.has_effect('miss_recoil'):
            attacker.battle_stats['hp'] -= int(attacker.max_stats['hp'] * 0.5)
            log += "{attacker.name} keeps going and crashes!\n"
    return log

# Damage type handle
def handle_damage(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    damage, multiplier = calculate_damage(attacker, defender, move)
    attacker.last_damage = damage
    defender.battle_stats['hp'] -= damage
    return f"{move.name} deals {damage} HP!\n"

def handle_recoil(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    recoil_damage = int(attacker.last_damage * float(effect.get('percentage', 0)))
    attacker.battle_stats['hp'] -= recoil_damage
    return f"{attacker.name} took {recoil_damage} HP recoil damage!\n"

def handle_counter(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    immune: bool= calculate_type_effectiveness(move.type, defender.type) == 0
    if immune:
        return f"{defender.name} is immune!\n"
    if defender.last_move is not None and defender.last_move.category == 'Physical':
        counter_damage = defender.last_damage * 2
        attacker.last_damage = counter_damage
        defender.battle_stats['hp'] -= counter_damage
        return f"{move.name} deals {counter_damage} damage!\n"
    else:
        return f"{move.name} missed!\n"

def handle_multi_hit(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    hit_count = random.choices([2, 3, 4, 5], [3/8, 3/8, 1/8, 1/8])[0]
    total_damage: int = 0
    for _ in range(hit_count):
        damage, multiplier = calculate_damage(attacker, defender, move)
        total_damage += damage
    attacker.last_damage = total_damage
    defender.battle_stats['hp'] -= total_damage
    return f"{move.name} hits {hit_count} time{'s' if hit_count > 1 else ''}, deals {total_damage} HP!\n"

def handle_double_hit(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    total_damage: int = 0
    for _ in range(2):
        damage, multiplier = calculate_damage(attacker, defender, move)
        total_damage += damage
    attacker.last_damage = total_damage
    defender.battle_stats['hp'] -= total_damage
    return f"{move.name} hits 2 times, deals {total_damage} HP!\n"

def handle_crit_ratio(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    damage, multiplier = calculate_damage(attacker, defender, move, float(effect.get('ratio', 1/24)))
    attacker.last_damage = damage
    defender.battle_stats['hp'] -= damage
    return f"{move.name} deals {damage} HP!\n"

def handle_half_hp(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    damage = defender.battle_stats['hp'] // 2
    immune: bool= calculate_type_effectiveness(move.type, defender.type) == 0
    if immune:
        return f"{defender.name} is immune!\n"
    attacker.last_damage = damage
    defender.battle_stats['hp'] -= damage
    return f"{move.name} deals half {defender.name} HP damage!\n"

def handle_level_damage(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    damage = attacker.level
    immune: bool= calculate_type_effectiveness(move.type, defender.type) == 0
    if immune:
        return f"{defender.name} is immune!\n"
    attacker.last_damage = damage
    defender.battle_stats['hp'] -= damage
    return f"{move.name} deals {damage}!\n"
    

def handle_random_level_damage(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    min = float(effect.get('min', 0.0))
    max = float(effect.get('max', 0.0))
    damage = int(attacker.level * random.uniform(min, max))
    immune: bool= calculate_type_effectiveness(move.type, defender.type) == 0
    if immune:
        return f"{defender.name} is immune!\n"
    attacker.last_damage = damage
    defender.battle_stats['hp'] -= damage
    return f"{move.name} deals {damage} damage!\n"
    

def handle_faint(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    target = str(effect.get('target', ''))
    probability = float(effect.get('probability', 0))
    if random.random() <= probability:
        if target == 'user':
            attacker.battle_stats['hp'] = 0
            return f"{attacker.name} fainted!\n"
        elif target == 'opp':
            defender.battle_stats['hp'] = 0
            return f"{defender.name} fainted!\n"
    return ""

# Heal type handle
def handle_heal(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    heal_amount = int(float(effect.get('max_hp', 0.0)) * attacker.max_stats['hp'])
    attacker.battle_stats['hp'] = min(attacker.max_stats['hp'], attacker.battle_stats['hp'] + heal_amount)
    return f"{attacker.name} recovered {heal_amount} HP!\n"

def handle_absorb(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    absorb_amout = int(attacker.last_damage * float(effect.get('percentage', 0)))
    attacker.battle_stats['hp'] += absorb_amout
    return f"{attacker.name} absorb {absorb_amout} HP!\n"

# Status type handle
## Start of the turn type
def handle_paralyze(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    # Non volatile status
    immune: bool = 'Electric' in defender.type
    if immune:
        return f"{defender.name} is immune to paralyze!\n"
    if defender.has_non_volatile_status():
        return f"{defender.name} already has non volatile status!\n"
    if random.random() <= float(effect.get('probability', 0)) and not immune and not defender.has_non_volatile_status():
        defender.apply_status('paralyze', 100)
        defender.update_stat_multiplier('spd', 1/2)
        return f"{defender.name} is paralyzed!\n"
    return ""

def handle_sleep(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    # Non volatile status
    if random.random() <= float(effect.get('probability', 0)) and not defender.has_non_volatile_status():
        defender.apply_status('sleep', random.randint(1,3))
        return f"{defender.name} fell asleep!\n"
    return ""

def handle_freeze(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    # Non volatile status
    immune: bool = 'Ice' in defender.type
    if immune:
        return f"{defender.name} is immune to freezing!\n"
    if defender.has_non_volatile_status():
        return f"{defender.name} already has non volatile status!\n"
    if random.random() <= float(effect.get('probability', 0)) and not immune and not defender.has_non_volatile_status():
        defender.apply_status('freeze', 100)
        return f"{defender.name} is frozen solid!\n"
    # pokemon have the possibility of immediately thawing after frozen
    if random.random() <= 0.25:
        defender.remove_status('freeze')
    return ""

def handle_recharge(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    attacker.apply_status('recharge', 1)
    return f"{attacker.name} needs to recharge!\n"

def handle_flinch(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    if random.random() <= float(effect.get('probability', 0)) and is_first_move:
        defender.apply_status('flinch', 1)
        return f"{defender.name} flinched!\n"
    return ""

def handle_confuse(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    if random.random() <= float(effect.get('probability', 0)):
        defender.apply_status('confuse', random.randint(1,4))
        return f"{defender.name} is confused!\n"
    return ""

## End of the turn type
def handle_badly_poison(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    # Non volatile status
    if defender.has_non_volatile_status():
        return f"{defender.name} already has non volatile status!\n"
    if random.random() <= float(effect.get('probability', 0)):
        defender.apply_status('badly_poison', 1) # start at 1 to count how long has it been taking effect
        return f"{defender.name} is badly poisoned!\n"
    return ""

def handle_burn(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    # Non volatile status
    immune: bool = 'Fire' in defender.type
    if immune:
        return f"{defender.name} is immune to burn!\n"
    if defender.has_non_volatile_status():
        return f"{defender.name} already has non volatile status!\n"
    if random.random() <= float(effect.get('probability', 0)):
        defender.apply_status('burn', 100)
        return f"{defender.name} is burned!\n"

    return ""

def handle_poison(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    # Non volatile status
    immune: bool = 'Steel' in defender.type or 'Poison' in defender.type
    if immune:
        return f"{defender.name} is immune to poison!\n"
    if defender.has_non_volatile_status():
        return f"{defender.name} already has non volatile status!\n"
    if random.random() <= float(effect.get('probability', 0)):
        defender.apply_status('poison', 100)
        return f"{defender.name} is poisoned!\n"
    return ""

def handle_seed(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    defender.apply_status('seed', 100)
    return f"{defender.name} is seeded!\n"

def handle_trap(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    defender.apply_status('trap', random.randint(4,5))
    return f"{defender.name} is trapped!\n"

# Stage type handle
def handle_stage(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    target = str(effect.get('target', ''))
    stat = str(effect.get('stat', ''))
    amount = int(effect.get('amount', 0))
    probability = float(effect.get('probability', 0))
    if random.random() <= probability:
        if target == 'user':
            attacker.update_stat_stage(stat, amount)
            return f"{attacker.name}'s {stat} stage changed by {amount}!\n"
        elif target == 'opp':
            defender.update_stat_stage(stat, amount)
            return f"{defender.name}'s {stat} stage changed by {amount}!\n"
    return ""

def handle_stage_reset(attacker: Pokemon, defender: Pokemon, effect: Dict[str, str | int | float], move: Move, is_first_move: bool) -> str:
    attacker.reset_stat_stages()
    defender.reset_stat_stages()
    return f"{move.name} eliminates stats stage changes!\n"

# Applying effect for status that take effect on the start of move 
def apply_start_move(pokemon: Pokemon, enemy: Pokemon) -> str:
    # deduct every moving turn
    def flinch_action() -> Tuple[str, bool]:
        pokemon.deduct_status_duration("flinch")
        return f"{pokemon.name} flinched!\n", False # False when the pokemon cannot move
    
    def sleep_action() -> Tuple[str, bool]:
        pokemon.deduct_status_duration("sleep")
        return f"{pokemon.name} is asleep!\n", False
    
    def recharge_action() -> Tuple[str, bool]:
        pokemon.deduct_status_duration("recharge")
        return f"{pokemon.name} is recharging!\n", False
    
    def confuse_action() -> Tuple[str, bool]:
        pokemon.deduct_status_duration("confuse")
        if random.random() <= 0.33:
            pokemon.battle_stats['hp'] -= 40
            return f"{pokemon.name} is confused and hit itself in the process!\n", False
        return "", True
    
    # unremovable
    def paralyze_action() -> Tuple[str, bool]:
        if random.random() <= 0.25:
            return f"{pokemon.name} is paralyzed!\n", False
        return "", True
    
    # removable
    def freeze_action() -> Tuple[str, bool]:
        if random.random() <= 0.8: # 20% of thawing
            return f"{pokemon.name} is frozen solid!\n", False
        statuses_to_remove.append('freeze')
        return f"{pokemon.name} thawed!\n", True

    status_actions = {
        'flinch': flinch_action,
        'sleep': sleep_action,
        'recharge': recharge_action,
        'confuse': confuse_action,
        'paralyze': paralyze_action,
        'freeze': freeze_action,
    }

    log = ""
    pokemon.can_move = can_move = True  # Assume the Pokémon can move initially
    statuses_to_remove: List[str] = []  # List to collect statuses to remove

    for status, duration in pokemon.statuses.items():
        if status in status_actions and duration > 0:
            message, status_can_move = status_actions[status]()
            log += message
            can_move = can_move and status_can_move  # Use logical AND to update can_move

    pokemon.can_move = can_move  # Set the final can_move status
    
    # Remove statuses after iteration
    for status in statuses_to_remove:
        pokemon.remove_status(status)

    # Handle expired statuses
    pokemon.remove_expired_statuses()

    return log

# Applying effect for status that take effect on the end of turn
def apply_end_turn(pokemon: Pokemon, enemy: Pokemon) -> str:
    def badly_poison_action() -> str: # deals n/16 of max hp where n is how long the effect has been running
        n = pokemon.get_status_duration('badly_poison')
        damage = int(pokemon.max_stats['hp'] * 0.0625 * n) 
        pokemon.battle_stats['hp'] -= damage
        pokemon.add_status_duration('badly_poison')
        return f"{pokemon.name} is badly poisoned for {damage} HP!\n"
    
    def burn_action() -> str:
        damage = int(pokemon.max_stats['hp'] * 0.125) # apply damage for 1/8 of max hp
        pokemon.battle_stats['hp'] -= damage
        return f"{pokemon.name} is burned for {damage} HP!\n"
    
    def poison_action() -> str:
        damage = int(pokemon.max_stats['hp'] * 0.125) 
        pokemon.battle_stats['hp'] -= damage
        return f"{pokemon.name} is poisoned for {damage} HP!\n"
    
    def seed_action() -> str:
        damage = int(pokemon.max_stats['hp'] * 0.125) 
        pokemon.battle_stats['hp'] -= damage # apply damage for 1/8 of max hp
        enemy.battle_stats['hp'] += damage # and heal the the pokemon that applied seeda
        return f"{pokemon.name} is drained for {damage} HP!\n"

    def trap_action() -> str:
        damage = int(pokemon.max_stats['hp'] * 0.125)
        pokemon.battle_stats['hp'] -= damage
        pokemon.deduct_status_duration("trap")
        return f"{pokemon.name} is trapped and received {damage} damage!\n"

    status_actions = {
        'badly_poison': badly_poison_action,
        'burn': burn_action,
        'poison': poison_action,
        'seed': seed_action,
        'trap': trap_action,
    }

    log = ""

    for status, duration in pokemon.statuses.items():
        if status in status_actions and duration > 0:
            message = status_actions[status]()
            log += message

    # Handle expired statuses
    pokemon.remove_expired_statuses()

    return log
    
def move_hit(attacker: Pokemon, defender: Pokemon, move: Move) -> bool:
    stat_stage_multiplier: List[float] =  [3/9, 3/8, 3/7, 3/6, 3/5, 3/4, 3/3, 4/3, 5/3, 6/3, 7/3, 8/3, 9/3]
    if move.accuracy is None:
        return True
    
    # From gen III, evasion and accuracy are combined and capped from [-6,  6]
    combined_stage = max(-6, min(6, attacker.stat_stages['acc'] - defender.stat_stages['eva']))

    if random.randint(0,100) <= float(move.accuracy) * stat_stage_multiplier[combined_stage + 6]:
        return True
    
    return False

def calculate_damage(attacker: Pokemon, defender: Pokemon, move: Move, crit_ratio: float = 1/24) -> tuple[int, float]:
    level = attacker.level

    if move.category == "Physical":
        a = attacker.battle_stats['atk']
        d = defender.battle_stats['def']
    else:
        a = attacker.battle_stats['sp_atk']
        d = defender.battle_stats['sp_def']
    
    crit_multiplier = 1.5 if random.random() <= crit_ratio else 1.0
    random_factor = random.randint(85, 100) / 100

    stab = 1.5 if move.type in attacker._type else 1.0
    burn = 0.5 if attacker.has_status('burn') and move.category == "Physical" else 1.0
    type_effectiveness = calculate_type_effectiveness(move.type, defender.type)
    
    base_damage = int((((2 * level / 5 + 2) * (move.power if move.power is not None else 0) * (a / d)) / 50 + burn * 2))
    damage = int(base_damage * crit_multiplier * random_factor * stab * type_effectiveness)
    
    return damage, type_effectiveness

def calculate_type_effectiveness(move_type: str, defender_types: list[str]) -> float:
    type_chart = {
        "Normal": {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
        "Fire": {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
        "Water": {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
        "Electric": {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0, "Flying": 2, "Dragon": 0.5},
        "Grass": {"Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Dragon": 0.5, "Steel": 0.5},
        "Ice": {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 0.5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5},
        "Fighting": {"Normal": 2, "Ice": 2, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Rock": 2, "Ghost": 0, "Dark": 2, "Steel": 2, "Fairy": 0.5},
        "Poison": {"Grass": 2, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5, "Ghost": 0.5, "Steel": 0, "Fairy": 2},
        "Ground": {"Fire": 2, "Electric": 2, "Grass": 0.5, "Poison": 2, "Flying": 0, "Bug": 0.5, "Rock": 2, "Steel": 2},
        "Flying": {"Grass": 2, "Electric": 0.5, "Fighting": 2, "Bug": 2, "Rock": 0.5, "Steel": 0.5},
        "Psychic": {"Fighting": 2, "Poison": 2, "Psychic": 0.5, "Dark": 0, "Steel": 0.5},
        "Bug": {"Fire": 0.5, "Grass": 2, "Fighting": 0.5, "Poison": 0.5, "Flying": 0.5, "Ghost": 0.5, "Steel": 0.5, "Fairy": 0.5},
        "Rock": {"Fire": 2, "Ice": 2, "Fighting": 0.5, "Ground": 0.5, "Flying": 2, "Bug": 2, "Steel": 0.5},
        "Ghost": {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": 0.5},
        "Dragon": {"Dragon": 2, "Steel": 0.5, "Fairy": 0},
        "Dark": {"Fighting": 0.5, "Psychic": 2, "Ghost": 2, "Dark": 0.5, "Fairy": 0.5},
        "Steel": {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2, "Rock": 2, "Steel": 0.5, "Fairy": 2},
        "Fairy": {"Fire": 0.5, "Fighting": 2, "Poison": 0.5, "Dragon": 2, "Dark": 2, "Steel": 0.5}
    }

    type_effectiveness = 1.0
    for def_type in defender_types:
        type_effectiveness *= type_chart.get(move_type, {}).get(def_type, 1.0)
    return type_effectiveness
