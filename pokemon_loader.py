# pokemon_loader.py

from pokemon_models import Pokemon, Move
from typing import List, Dict
import pandas as pd
import json

def load_pokemon_data(file_path: str) -> List[Pokemon]:
    df = pd.read_excel(file_path, sheet_name='Pokemon')
    pokemon_list = []
    for _, row in df.iterrows():
        moves_list = row['Moves'].split(', ')
        types = [t.strip() for t in row['Type'].split(',')] # Properly split the type for dual type Pokemon
        pokemon = Pokemon(
            name=row['Name'],
            types=types,
            hp=row['HP'],
            attack=row['Attack'],
            defense=row['Defense'],
            special_attack=row['Sp. Atk'],
            special_defense=row['Sp. Def'],
            speed=row['Speed'],
            moves_list=moves_list,
            level=90
        )
        pokemon_list.append(pokemon)
    return pokemon_list

def load_move_data(file_path: str) -> Dict[str, Move]:
    df = pd.read_excel(file_path, sheet_name='Move')
    move_dict: Dict[str, Move] = {}
    
    for _, row in df.iterrows():
        effect_string = row['Effect']
        try:
            effect = json.loads(effect_string)
        except json.JSONDecodeError:
            effect = effect_string  # If it fails to parse, keep the original string
            print('JSONDecodeError: failed to parse effect string for move:', row['Name'])
        
        move = Move(
            name=row['Name'],
            type=row['Type'],
            category=row['Category'],
            power=row['Power'] if pd.notna(row['Power']) and row['Power'] != '—' else None,
            accuracy=row['Accuracy'] if pd.notna(row['Accuracy']) and row['Accuracy'] != '—' else None,
            pp=row['PP'],
            effect=effect
        )
        move_dict[move.name] = move
    
    return move_dict

def link_pokemon_moves(pokemon_list: List[Pokemon], move_dict: Dict[str, Move]):
    for pokemon in pokemon_list:
        moves = [move_dict.get(move_name, None) for move_name in pokemon.moves_list]
        pokemon.moves = [move for move in moves if move is not None]

def load_pokemon_list(file_path: str) -> List[Pokemon]:
    pokemon_list = load_pokemon_data(file_path)
    move_dict = load_move_data(file_path)
    link_pokemon_moves(pokemon_list, move_dict)
    return pokemon_list