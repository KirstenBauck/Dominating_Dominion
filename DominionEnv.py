import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dominion import Piles, Phase
from dominion.Game import Game
from typing import Optional
import logging

# Set up a logging file
logging.basicConfig(level=logging.DEBUG, filename="logs/debug_masked.log", filemode="w", format="%(message)s")

class DominionEnv(gym.Env):
    "Wraps dwagon/pydominion to create a reinforcement learning environment"

    def __init__(self, num_players=2, card_set=None, quiet_flag=True, debug_flag=False):
        super(DominionEnv, self).__init__()
        # Have a default card set
        self.card_set = card_set if card_set else [
            "Cellar", "Market", "Militia", "Mine", "Moat", "Remodel", "Smithy", "Village", "Throne Room", "Workshop"
        ]
        # Alphabetize so that if given same cards, will be represented the same way
        self.card_set = sorted(self.card_set)
        self.base_cards = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]
        self.cards = self.base_cards + self.card_set

        # Initialize other arguments
        self.quiet = quiet_flag
        self.debug = debug_flag
        self.num_players = num_players

        # Initialize Dominion game
        self.game = Game(
            numplayers=self.num_players,
            initcards=self.card_set,
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
            # TODO: After looking over logs, the maximum valid moves always seems to be 23...
        self.action_space = spaces.Discrete(28)
        

    def reset(self, seed=None, options: Optional[dict] = None):
        """Reset the game"""
        super().reset(seed=seed)
        self.debug_output(f"\nResetting the game")
        self.game.game_over = False
        self.game.players.clear()
        self.game.start_game()
        self.debug_output(f"Started the game")
        self.game.current_player.phase = Phase.NONE
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get the current state of the game."""
        # Get the next player
        player = next(iter(self.game.players.values()))

        # Initialize card counts for each player pile
        #   For now, not including EXILE and RESERVE (these are in expansions)
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

    def step(self, action):
        """Take an action in the game (buy a card, play an action, etc.)."""

        # Check if game is over
        if self.game.game_over:
            self.debug_output(f"Game is over")
            return self._get_observation(), 0, True, False, {}

        # Check if this is a new turn and start it if it is
        #  Note: I have set Phase.NONE to represent when a player ends their turn
        if self.game.current_player.phase == Phase.NONE:
            self.game._validate_cards()
            self.debug_output(f"Before switching player, Current: {self.game.current_player.name}")
            self.game.current_player = self.game.player_to_left(self.game.current_player)
            self.debug_output(f"After switching player, Current: {self.game.current_player.name}")
            self.game.current_player.start_turn()
            self.game.current_player.turn_number += 1
            self.debug_output(f"This is a new turn for player: {self.game.current_player.name}")
            # Hard stop if too many turns
            self.debug_output(f"TURN {self.game.current_player.turn_number}: Starting new turn")
            ##### CHANGE THIS IF NEEDED #####
            if self.game.current_player.turn_number >= 75:
                self.debug_output(f"Reached a hard stop of 75 turns")
                self.game.current_player.output(f"Reached turn 75")
                return self._get_observation, -100, True, False, {}
            #################################
            # Check if player turn was skipped
            if self.game.current_player.skip_turn:
                self.game.current_player.skip_turn = False
                self.debug_output(f"Skipped players turn")
                return self._get_observation(), 0, False, False, {}
            # Change to action phase to start turn
            self.game.current_player.phase = Phase.ACTION
            self.debug_output(f"********* ACTION PHASE *********")
            self.game.current_player.output(f"{'#' * 30} Turn {self.game.current_player.turn_number} {'#' * 30}")
            #stats = f"({self.game.current_player.get_score()} points, {self.game.current_player.count_cards()} cards)"
            #self.game.current_player.output(f"{self.game.current_player.name}'s Turn {stats}")

        # Get available options for ACTION and BUY phase
        options = self.game.current_player._choice_selection()
        self.debug_output(f"The number of valid options are: {len(options)}")
        #self.game.current_player.output(f"There are {len(options)} valid options: {options}")

        # Ensure an action is valid
        #  Note: This is not needed for PPO masking
        ##### CHANGE THIS IF NEEDED #####
        if action >= len(options):
            return self._get_observation(), -50, False, False, {}
        #################################
        
        # Choose the option
        opt = options[action]
        self.debug_output(f"Chosen Option: {opt['verb']} with index: {action}")
        #self.game.current_player.output(f"Chosen Option: {opt['verb']}")

        # Perform the action
        self.game.current_player._perform_action(opt)
        self.debug_output(f"Performed action {opt['action']}")

        # Check if they buy a card that triggers the end of game
        terminated = self.game.isGameOver()
        if terminated:
            # They still need to enter cleanup phase for full completion
            self.debug_output(f"Buying a card triggered end of game")
            self.game.current_player.phase = Phase.CLEANUP

        # Check if phase should transition
        if opt["action"] in ["quit", None]:  
            # Transition from ACTION to BUY
            if self.game.current_player.phase == Phase.ACTION:
                self.debug_output(f"********* BUY PHASE *********")
                self.game.current_player.phase = Phase.BUY
            # Transition from BUY to CLEANUP
            elif self.game.current_player.phase == Phase.BUY:
                # Ensure any cards that have effects are triggered
                self.game.current_player.hook_end_buy_phase()
                self.game.current_player.phase = Phase.CLEANUP

        # If in cleanup phase, end turn and start next player's turn
        if self.game.current_player.phase == Phase.CLEANUP:
            self.debug_output(f"********* CLEANUP PHASE *********")
            self.debug_output(f"Ending Turn {self.game.current_player.name}")
            # Cleanup phase, end player turn, and validate environment
            self.game.current_player.cleanup_phase()
            self.game.current_player._card_check()
            self.game.current_player.end_turn()
            self.game._validate_cards()
            self.game._turns.append(self.game.current_player.uuid)
            # Signal that it is the next players turn by setting current player phase to NONE
            self.game.current_player.phase = Phase.NONE

        # Only print out terminated flag at the end of turn
        if self.debug and self.game.current_player.phase == Phase.NONE:
            logging.debug(f"Terminated is: {terminated}\n")
        
        # Do any special actions that occur at the end of the game
        if terminated:
            self.game.game_over = True
            for plr in self.game.player_list():
                plr.game_over()
            self.game.current_player.output(f"End of game cards are: {self.game.current_player.end_of_game_cards}")

        # Calculate the reward for training
        reward = self._calculate_reward(terminated)
        
        return self._get_observation(), reward, terminated, False, {}


################################## Helper Functions ###################################################

    def get_action_mask(self):
        """Make a binary mask of the valid actions (1 = valid, 0=invalid)"""
        options = self.game.current_player._choice_selection()
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        # Get only the valid moves
        for i, option in enumerate(options):
            if option['selector'] != "-":
                mask[i] = 1
        return mask

    def count_card_type(self, card_name):
        """Counts the total number of a specific card across all piles."""
        return sum(deck.count(card_name) for deck in self.game.current_player.piles.values())

    ##### CHANGE THIS IF NEEDED #####
    def _calculate_reward(self, terminated):
        """Calculate the reward function for agent"""
        reward = 0
        # Big reward for winning, big penalty for losing
        if terminated:
            reward += 100 if self._who_won() else -100

            #Bonus for final score
            final_score = self.game.current_player.get_score()
            reward += final_score * 10
        
        # Victory point rewards
        reward += (self.count_card_type("Estate") * 1) * 3
        reward += (self.count_card_type("Duchy") * 3) * 3
        reward += (self.count_card_type("Province") * 6) * 3

        # Curse Penalty
        reward -= self.count_card_type("Curse") * -1

        # Small penalty per turn
        #reward -= self.game.current_player.turn_number

        # TODO: Do we want to reward extra buys used??

        # TODO: Do we want to reward extra actions used??

        ## Ending a turn with zero actions?
        if self.game.current_player.phase == Phase.NONE and self.game.current_player.actions == 0:
            reward -= 2

        # TODO: Think about number of cards in hand?

        return reward
    #################################

    def _who_won(self):
        """Determine if Player 0 won"""
        # Get the scores
        scores = self.game.whoWon()
        first_player = list(self.game.players.values())[0]
        player_0_score = scores.get(first_player.name, 0)
        # Check if first player score is the maximum score
        return player_0_score == max(scores.values())
    
    def debug_output(self, msg):
        """Print debug messages"""
        if self.debug:
            logging.debug(msg)
