import gymnasium as gym
import numpy as np
from gymnasium import spaces
from external.pydominion.dominion import Piles, Phase
from external.pydominion.dominion.Game import Game
from typing import Optional

class DominionEnv(gym.Env):
    "Wraps dwagon/pydominion to create a reinforcement learning environment"
    
    # Game phases
    ACTION_PHASE = 0
    BUY_PHASE = 1
    CLEANUP_PHASE = 2

    def __init__(self, num_players=2, card_set=None, quiet_flag=True):
        super(DominionEnv, self).__init__()
        self.num_players = num_players
        # Properly initialize the cards in play (Base + Kingdom)
        #TODO: ALphatize
        self.card_set = card_set if card_set else [
            "Cellar", "Market", "Militia", "Mine", "Moat", "Remodel", "Smithy", "Village", "Throne Room", "Workshop"
        ]
        self.base_cards = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province"]
        self.cards = self.base_cards + self.card_set
        self.quiet = quiet_flag

        # Initialize Dominion game
        self.game = Game(
            numplayers=num_players,
            initcards=card_set,
            validate_only=False,
            prosperity=False,
            potions = False,
            shelters = False,
            card_path="external/pydominion/dominion/cards",
            quiet = quiet_flag
        )
        
        # Using a box because PPO doesn't support a dict of dicts
        self.observation_space = spaces.Box(
            low=0,
            high=60,  # Maximum possible number of cards <-- Copper = 60
            shape=(len(self.cards) * 8,),  # 8 zones: hand, duration, defer, deck, played, discard, supply, trash
            dtype=np.int32
        )
                
        # Define action space
            # ACTION phase: 10 action cards + end phase = 11
            # BUY phase: 16 buyable cards + end phase = 17
        self.action_space = spaces.Discrete(28)
        

    def reset(self, seed=None, options: Optional[dict] = None):
        """Reset the game"""
        super().reset(seed=seed)
        self.game.game_over = False # Ensure game state is reset
        self.game.players.clear()
        self.game.start_game()
        self.game.current_player.phase = Phase.NONE

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

    def step(self, action):
        """Take an action in the game (buy a card, play an action, etc.)."""

        # Check if game is over
        if self.game.game_over:
            return self._get_observation(), 0, True, False, {}

        # Check if this is a new turn and start it
        if self.game.current_player.phase == Phase.NONE:
            self.game.current_player.output(f"NEW TURN")
            self.game._validate_cards()
            self.game.current_player = self.game.player_to_left(self.game.current_player)
            self.game.current_player.output(f"Next Player: {self.game.current_player.name}")
            self.game.current_player.start_turn()
            self.game.current_player.turn_number += 1
            if self.game.current_player.skip_turn:
                self.game.current_player.skip_turn = False
                return self._get_observation(), 0, False, False, {}
            self.game.current_player.output(f"{'#' * 30} Turn {self.game.current_player.turn_number} {'#' * 30}")
            stats = f"({self.game.current_player.get_score()} points, {self.game.current_player.count_cards()} cards)"
            self.game.current_player.output(f"{self.game.current_player.name}'s Turn {stats}")
            self.game.current_player.phase = Phase.ACTION
            self.game.current_player.output(f"*****ACTION PHASE****")

        # Get available options
        options = self.game.current_player._choice_selection()

        # Ensure an action is valid
        if action >= len(options):
            return self._get_observation(), -1, False, False, {}  # Penalize invalid action by -1
        
        # Choose the options
        opt = options[action]
        self.game.current_player.output(f"Chosen Option: {opt['verb']}")
        self.game.current_player._perform_action(opt)  # Perform action

        # Check if phase should transition
        if opt["action"] in ["quit", None]:  
            # Transition from ACTION to BUY
            if self.game.current_player.phase == Phase.ACTION:
                self.game.current_player.output(f"*****BUY PHASE****")
                self.game.current_player.phase = Phase.BUY
            # Transition from BUY to CLEANUP
            elif self.game.current_player.phase == Phase.BUY:
                # Ensure any cards that have effects are triggered
                self.game.current_player.hook_end_buy_phase()
                self.game.current_player.output(f"*****CLEANUP PHASE****")
                self.game.current_player.phase = Phase.CLEANUP

        # If in cleanup phase, end turn and start next player's turn
        if self.game.current_player.phase == Phase.CLEANUP:
            self.game.current_player.output(f"Inside cleanup phase")
            self.game.current_player.cleanup_phase()
            self.game.current_player.output(f"Ending Turn {self.game.current_player.name}")
            self.game.current_player._card_check()
            self.game.current_player.end_turn()
            self.game._validate_cards()
            self.game._turns.append(self.game.current_player.uuid)
            self.game.current_player.phase = Phase.NONE  # Signal next player/new turn

        # Check if game is over
        terminated = self.game.isGameOver()
        if terminated:
            self.game.game_over = True
            for plr in self.game.player_list():
                plr.game.game_over()

        # TODO: Expand upon this reward function
        # Think about rewards for provinces, dutchy, trashing cards, etc.
        reward = 10 if terminated and self._who_won() else -1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

######################## Helper Functions ###################################################

    def _who_won(self):
        """Determine if Player 0 won"""\
        # Get the scores
        scores = self.game.whoWon()
        first_player = list(self.game.players.values())[0]
        player_0_score = scores.get(first_player.name, 0)
        # Check if first player score is the maximum score
        return player_0_score == max(scores.values())
