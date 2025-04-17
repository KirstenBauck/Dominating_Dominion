import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dominion import Piles, Phase
from dominion.Game import Game
from typing import Optional
import logging

# Set up a logging file
logging.basicConfig(level=logging.DEBUG, filename="logs/Rylan/debug_masked.log", filemode="w", format="%(message)s")

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
        #TODO: Change this for better representation if possible, so big :(
        self.observation_space = spaces.Box(
            low=0,
            high=60,  # Maximum possible value
            shape=(75,),  # Match this with your new observation vector length
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
        """Get a more compact and informative state representation"""
        player = next(iter(self.game.players.values()))

        # Get counts for each card type across different piles
        hand_counts = self._get_pile_counts(player.piles[Piles.HAND]._cards)
        played_counts = self._get_pile_counts(player.piles[Piles.PLAYED]._cards)
        discard_counts = self._get_pile_counts(player.piles[Piles.DISCARD]._cards)
        deck_size = len(player.piles[Piles.DECK]._cards)  # Just track size for deck

        # Supply pile counts
        supply_counts = {}
        for card in self.cards:
            supply_counts[card] = len(self.game.card_piles.get(card, []))

        # Get resources and state
        actions = player.actions
        buys = player.buys
        coins = player.coins
        phase_id = {"NONE": 0, "ACTION": 1, "BUY": 2, "CLEANUP": 3}.get(player.phase.name, 0)

        # Create a more compact observation vector
        obs = []

        # Resources (3 values)
        obs.extend([actions, buys, coins])

        # Phase (1 value)
        obs.append(phase_id)

        # Hand counts for each card (17 values)
        for card in self.cards:
            obs.append(hand_counts.get(card, 0))

        # Supply counts for key cards (17 values)
        for card in self.cards:
            obs.append(supply_counts.get(card, 0))

        # Played area counts (17 values)
        for card in self.cards:
            obs.append(played_counts.get(card, 0))

        # Discard counts (17 values)
        for card in self.cards:
            obs.append(discard_counts.get(card, 0))

        # Deck size (1 value)
        obs.append(deck_size)

        # Player metrics (2 values)
        obs.append(player.get_score())  # Current score
        obs.append(player.turn_number)  # Turn number

        obs = np.array(obs, dtype=np.int32)
        return obs
    

    def _get_pile_counts(self, pile):
        """Helper to count cards in a pile"""
        counts = {}
        for card in pile:
            counts[card.name] = counts.get(card.name, 0) + 1
        return counts

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
            # Count the number of used buys and used actions
            self.game.current_player._used_buys = 0
            self.game.current_player._used_actions = 0
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

        # Update used buys and actions
        # TODO: fix this!
        prev_num_actions = int(self.game.current_player.actions)
        prev_num_buys = int(self.game.current_player.buys)

        # Perform the action
        self.game.current_player._perform_action(opt)
        self.debug_output(f"Performed action {opt['action']}")

        # Update used buys and actions
        # TODO: fix this!
        if self.game.current_player.actions < prev_num_actions:
            self.game.current_player.used_actions += 1
        if self.game.current_player.actions < prev_num_buys:
            self.game.current_player.used_buys += 1

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

        # Memory Optimization
        if self.game.current_player.turn_number % 10 == 0:
            self.optimize_memory_usage()
            
        # Early stopping condition for stalled games
        if self.game.current_player.turn_number > 30:
            # Check if game is stalled by looking at province pile
            province_pile = len(self.game.card_piles.get("Province", []))
            starting_provinces = 8  # Standard number in base game

            # If no provinces were bought after 30 turns, end the game
            if province_pile == starting_provinces:
                self.debug_output("Game appears stalled - no provinces bought after 30 turns")

                # Determine winners based on current scores
                scores = {player.name: player.get_score() for player in self.game.player_list()}
                max_score = max(scores.values())

                # Penalize stalled games
                reward = -50 + (self.game.current_player.get_score() * 0.5)

                # Check if current player has max score (tie goes to current player)
                if scores[self.game.current_player.name] >= max_score:
                    reward += 25  # Less than winning normally but still positive

                self.game.game_over = True
                return self._get_observation(), reward, True, False, {"stalled_game": True}
        
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
        reward = 0

        # Game end reward
        if terminated:
            # Win/loss with significant weight
            reward += 50 if self._who_won() else -50

            # Final score relative to opponents
            player_score = self.game.current_player.get_score()
            opponent_scores = [p.get_score() for p in self.game.player_list() if p != self.game.current_player]
            avg_opponent_score = sum(opponent_scores) / len(opponent_scores) if opponent_scores else 0
            score_difference = player_score - avg_opponent_score
            reward += score_difference * 2  # Reward score advantage more heavily

            return reward

        # In-game rewards
        # current_player = self.game.current_player

        # Reward engine building more explicitly
        # if hasattr(current_player, 'last_bought'):
        #     last_card = current_player.last_bought

        #     # Strategic action cards (incentivize engine building)
        #     if last_card in ["Village", "Market", "Smithy"]:
        #         reward += 0.4  # Higher reward for key engine components
        #     elif last_card == "Throne Room":
        #         reward += 0.5  # High value for combo enablers

        #     # VP cards with diminishing returns based on timing
        #     if last_card == "Province":
        #         # Provinces more valuable later in game
        #         turn_factor = min(1.0, current_player.turn_number / 15)
        #         reward += 0.6 + (turn_factor * 0.4)
        #     elif last_card == "Duchy":
        #         # Duchies valuable in mid-to-late game
        #         mid_game_factor = min(1.0, current_player.turn_number / 12)
        #         reward += 0.2 + (mid_game_factor * 0.3)

        # # Reward for hand quality/potential
        # if current_player.phase == Phase.ACTION:
        #     # Count action cards in hand
        #     action_cards = sum(1 for card in current_player.piles[Piles.HAND]._cards if card.isAction())
        #     if action_cards >= 2:
        #         reward += 0.05 * action_cards  # Reward having multiple actions

        #     # Reward coin potential
        #     coin_potential = sum(card.coin for card in current_player.piles[Piles.HAND]._cards if hasattr(card, 'coin'))
        #     reward += 0.02 * coin_potential

        # Small time penalty
        # reward -= 0.001

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


    # Memory Optimization functions
    def __del__(self):
        """Clean up resources when the environment is deleted"""
        if hasattr(self, 'game'):
            del self.game

    def optimize_memory_usage(self):
        """Optimize memory usage during training"""
        # Reset logging buffers if they're getting too large
        if hasattr(self.game, 'log_buffer') and len(self.game.log_buffer) > 1000:
            self.game.log_buffer = []

        # Clean up any large temporary data structures
        if hasattr(self, '_temp_data'):
            del self._temp_data

        # Run Python's garbage collector explicitly
        import gc
        gc.collect()