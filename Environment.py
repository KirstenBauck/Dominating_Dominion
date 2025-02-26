import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
import random

class CardType(Enum):
    TREASURE = 1
    VICTORY = 2
    ACTION = 3
    CURSE = 4

class Card:
    def __init__(self, name:str, cost:int, card_type:CardType, value:int=0, points:int=0):
        self.name = name
        self.cost = cost
        self.card_type = card_type
        self.value = value # For treasure cards
        self.points = points # For victory cards

class DominionEnv(gym.Env):
    def __init__(self, render_mode: Optional[str]=None):
        self.render_mode = render_mode

        # Define the basic cards
        self.all_cards = {
            # Treasure Cards
            "Copper": Card("Copper", 0, CardType.TREASURE, value=1),
            "Silver": Card("Silver", 3, CardType.TREASURE, value=2),
            "Gold": Card("Gold", 6, CardType.TREASURE, value=3),

            # Victory Cards
            "Estate": Card("Estate", 2, CardType.VICTORY, points=1),
            "Duchy": Card("Duchy", 5, CardType.VICTORY, points=3),
            "Province": Card("Province", 8, CardType.VICTORY, points=6)
        }

        # Game constants
        self.num_players = 2 # Start with 2?
        self.max_turns = 100 #just to prevent infinite as we debug

        # Define observation and action space
        self._setup_spaces()

        # Game State
        self.current_player = 0
        self.turn_number = 0
        self.game_over = False

        # Reset the environment
        self.reset()

    def _setup_spaces(self):

        # TODO: Need to think more about this observation space - what is this really?
        self.observation_space = spaces.Dict({
            "hand": spaces.Discrete(), # What is the max cards in hand?
            "deck_size": spaces.Disccrete(), # Max size of the deck
            "discard_size": spaces.Discrete(), # Max size of discard pile
            "player_turn": spaces.Discrete(self.num_players),
            "actions_remaining": spaces.Discrete(), # Max actions
            "buys_remaining": spaces.Discrete(), # Max buys
            "coins": spaces.Discrete(), # Max coins
            "supply_piles": spaces.Dict({
                card_name: spaces.Discrete() for card_name in self.all_cards
            })
        })

        # TODO: Set up an action space
        self.action_space = spaces.Discrete(len(self.all_cards) + 2)

    def reset(self, seed: Optional[int] =  None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize supply piles
        self.supply = {
            "Copper": 60 - (self.num_players * 7),
            "Silver": 40,
            "Gold": 30,
            "Estate": 8 if self.num_players==2 else 12,
            "Duchy": 8 if self.num_players==2 else 12,
            "Province": 8 if self.num_players==2 else 12
        }

        # Initialize player states
        self.players = []
        for i in range(self.num_players):
            # Set up the hand
            inital_deck = ["Copper"] * 7 + ["Estate"] * 3
            random.shuffle(inital_deck)

            # Set up the player
            player = {
                "deck": inital_deck,
                "hand": [],
                "discard": [],
                "play_area": [],
                "actions": 0,
                "buys": 0,
                "coins": 0
            }
            self.players.append(player)
        
        # Set up the first turn
        self.current_player = 0
        self.turn_number = 0
        self.game_over = False
        self.current_phase = "action" # action, buy, cleanup phases

        # Draw the inital hand for the first player
        self._draw_hand(self.current_player)
        self._start_turn()

        # Get the inital observation and info
        observation = self._get_obs()
        info = self._get_info

        return observation, info
    
    def _start_turn(self):
        """Initialize a players turn with actions, buys, etc."""
        current_player = self.players[self.current_player]
        current_player["actions"] = 1
        current_player["buys"] = 1
        current_player["coins"] = 0
        self.current_phase = "action"
    
    def _draw_hand(self, player_idx: int, num_cards: int=5):
        """Draw a hand of cards for the specified player"""
        player = self.players[player_idx]

        for _ in range(num_cards):
            # If the deck is empty, shuffle discard pile into deck
            if not player["deck"]:
                if not player["discard"]:
                    print("Uh oh! No cards in deck or discard!")
                    break 
                player["deck"] = player["discard"]
                player["discard"] = []
                random.shuffle(player["deck"])
        
            # Draw the card
            player["hand"].append(player["deck"].pop(0))


    def _get_obs(self):
        """Convert the game state into the observation format"""
        current_player = self.players[self.current_player]

        # Convert hand to a sparse representation
        hand_array = np.zeros(10, dtype=np.int32) #TODO: Figure out max hand size
        for i, card_name in enumerate(current_player["hand"]):
            if i < 10: #TODO: only consider the max hand size
                hand_array[i] = list(self.all_cards.keys()).index(card_name)
    
        # Create observation dictionary
        observation = {
            "hand": hand_array,
            "deck_size": len(current_player["deck"]),
            "discard_size": len(current_player["discard"]),
            "player_turn": self.current_player,
            "actions_remaining": current_player["actions"],
            "buys_remaining": current_player["buys"],
            "coins": current_player["coins"],
            "supply_piles": {card_name: count for card_name, count in self.supply.items()}
        }
        return observation

    def _get_info(self):
        """Return additional information that isn't a part of observation"""
        return {
            "turn_number": self.turn_number,
            "game_over": self.game_over,
            "current_phase": self.current_phase
        }

    def step(self, action: int):
        """Take an action in the environment"""
        #TODO: Make this funcition
        # Check if game over

        # Setup

        # Check if in action phase
            # Check if in special "end action phase"
                # Move to buy phase and count treasure
            # Otherwise
                # Play action card
                # If not action cards left, move to buy phase and count treasure

        # Check if in buy phase
            # Check if in special "end buy phase"
                # End turn
            # otherwise
                # buy a card
                # if no buys left, end turn

        # Get updated observation and info

        # return observation, reward, if game ends, and info
    
    def _cleanup_phase(self):
        """Handle the cleanup phase - discard hand and play area, draw new hand"""
        current_player = self.players(self.current_player)

        # Discard hand and play area
        current_player["discard"].extend(current_player["hand"])
        current_player["discard"].extend(current_player["play_area"])
        current_player["hand"] = []
        current_player["play_area"] = []

        # Draw new hand
        self._draw_hand(self.current_player)

    def _next_player(self):
        """Switch to the next player"""
        self.current_player = (self.current_player + 1) % self.num_players
        self.turn_number + 1
        self._start_turn()
    
    def _check_game_end(self):
        """Check if the game has ended"""
        #TODO: Do we want to change this game end at all?
        # Game end if Province supply empty
        if self.supply.get("Province", 0) == 0:
            self.game_over = True
            return True
        
        # Also and games if we hit the maximum turn limit
        if self.turn_number >= self.max_turns:
            self.game_over = True
            return True
        
        return False
    
    def render(self):
        """Render the game state"""
        #TODO: decide what would be helpful to have? Some ideas...
        print(f"Turn {self.turn_number}, Player {self.current_player}")
        print(f"Phase: {self.current_phase}")
        current_player = self.players[self.current_player]
        print(f"Hand: {current_player["hand"]}")
        print(f"Actions: {current_player["actions"]}, Buys: {current_player["buys"]}, Coins: {current_player["coins"]}")
        print(f"Deck size: {len(current_player["deck"])}, Discard size: {len(current_player["discard"])}")
        print(f"Supply: {self.supply}")
    
    def calculate_scores(self):
        """Calculate teh final score for all players"""
        scores = []
        for player in self.players:
            # Combine all the cards
            all_cards = player["hand"] + player["deck"] + player["discard"] + player["play_area"]
            # Calculate score
            score = sum(self.all_cards[card].points for card in all_cards if self.all_cards[card].card_type == CardType.VICTORY)
            scores.append(score)
        return scores