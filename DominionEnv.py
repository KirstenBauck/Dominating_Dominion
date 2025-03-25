import gymnasium as gym
import numpy as np
from gymnasium import spaces
from external.pydominion.dominion import Piles
from external.pydominion.dominion.Game import Game
from typing import Optional

class DominionEnv(gym.Env):
    "Wraps dwagon/pydominion to create a reinforcement learning environment"
    
    # Game phases
    ACTION_PHASE = 0
    BUY_PHASE = 1
    CLEANUP_PHASE = 2

    def __init__(self, num_players=2, card_set=None):
        super(DominionEnv, self).__init__()
        self.num_players = num_players
        # Properly initialize the cards in play (Base + Kingdom)
        self.card_set = card_set if card_set else [
            "Cellar", "Market", "Militia", "Mine", "Moat", "Remodel", "Smithy", "Village", "Throne Room", "Workshop"
        ]
        self.base_cards = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province"]
        self.cards = self.base_cards + self.card_set

        # Initialize Dominion game
        self.game = Game(
            numplayers=num_players,
            initcards=card_set,
            validate_only=False,
            prosperity=False,
            potions = False,
            shelters = False,
            card_path="external/pydominion/dominion/cards"
        )
        
        # Using a box because PPO doesn't support a dict of dicts
        self.observation_space = spaces.Box(
            low=0,
            high=60,  # Maximum possible number of cards <-- Copper = 60
            shape=(len(self.cards) * 8,),  # 8 zones: hand, duration, defer, deck, played, discard, supply, trash
            dtype=np.int32
        )
                
        # Define action space
        #TODO: test run to see if these is a good number for the action space?
        self.action_space = spaces.Discrete(len(self.cards))
        

    def reset(self, seed=None, options: Optional[dict] = None):
        """Reset the game"""
        super().reset(seed=seed)
        self.game.start_game()
        return self._get_observation(), self._get_info()
    
    def _get_observation(self):
        """Get the current state of the game."""
        player = next(iter(self.game.players.values()))

        # Initialize card counts for each player pile
        #   For now, not including EXILE and RESERVE
        piles = {
            "hand": player.piles[Piles.HAND]._cards,
            "duration": player.piles[Piles.DURATION]._cards,
            "defer": player.piles[Piles.DEFER]._cards,
            "deck": player.piles[Piles.DECK]._cards,
            "played": player.piles[Piles.PLAYED]._cards,
            "discard": player.piles[Piles.DISCARD]._cards
        }

        # Empty array for observation - player
        observation = np.zeros(len(self.cards) * 8, dtype=np.int32)
        def get_card_counts(pile):
            return [sum(1 for c in pile if c.name == card) for card in self.cards]

        # Fill the observation array - player
        index = 0
        for pile_name in ["hand", "duration", "defer", "deck", "played", "discard"]:
            observation[index:index + len(self.cards)] = get_card_counts(piles[pile_name])
            index += len(self.cards)

        # Get supply and trash pile counts
        observation[index:index + len(self.cards)] = [len(self.game.card_piles.get(card, [])) for card in self.cards]
        index += len(self.cards)
        observation[index:index + len(self.cards)] = [sum(1 for c in self.game.trash_pile._cards if c.name == card) for card in self.cards]

        return observation
    
    def _get_info(self):
        return {}

    # TODO: work on this step so an agent can play
    def step(self, action):
        """Take an action in the game (buy a card, play an action, etc.)."""
        if self.game.game_over:
            return self._get_observation(), 0, True, False, {}
        
        # Perform a turn
        self.game.turn()
        
        # Check if the game is over
        terminated = self.game.game_over

        # A simple reward concerning if won the game
        # TODO: Expand upon this reward function (not a huge problem)
        reward = 1 if terminated and self._who_won() else 0
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _who_won(self):
        """Determine if Player 0 won"""\
        # Get player 0 score, default to 0 if not found
        scores = self.game.whoWon()
        player_0_score = scores.get(self.game.players[0].name, 0)

        # If player 0 has maximum score, they won!
        max_score = max(scores.values())
        return player_0_score == max_score

    def render(self, mode="human"):
        """Print game state for debugging"""
        self.game.print_state(card_dump=True)
