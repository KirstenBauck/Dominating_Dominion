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
        self.learning_player_index = 0 # Set agent to be player 1
        self.current_player_index = 0
        self.spendall = 0 # For the big money bot

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
            high=75,  # Maximum number of turns (which would be the max)
            shape=((len(self.cards) * 8) + 9,),  # 8 zones: hand, duration, defer, deck, played, discard, supply, trash
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

        # Basic resource values
        actions = int(player.actions)
        buys = int(player.buys)
        coins = int(player.actions)
        phase_id = player.phase.value
        deck_size = len(player.piles[Piles.DECK]._cards)
        score = player.get_score()
        turn_number = player.turn_number
        current_player_index = self.current_player_index
        learning_player_index = self.learning_player_index

        # Intilize observation
        observation = [actions, buys, coins, deck_size, score, turn_number, phase_id, current_player_index, learning_player_index]

        # Card counts per pile (hand, duration, defer, deck, played, discard)
        for pile_name in [Piles.HAND, Piles.DURATION, Piles.DEFER, Piles.DECK, Piles.PLAYED, Piles.DISCARD]:
            pile = player.piles[pile_name]._cards
            for card_name in self.cards:
                count = sum(1 for c in pile if c.name == card_name)
                observation.append(count)

        # Supply pile (available cards in the game supply)
        for card_name in self.cards:
            count = len(self.game.card_piles.get(card_name, []))
            observation.append(count)

        # Trash pile (how many of each card are in the trash)
        for card_name in self.cards:
            count = sum(1 for c in self.game.trash_pile._cards if c.name == card_name)
            observation.append(count)

        return np.array(observation, dtype=np.int32)

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
            # Check to see if its the agent who should be playing
            self.current_player_index = self.game.player_list().index(self.game.current_player)
            self.debug_output(f"After switching player, Current: {self.game.current_player.name} with index: {self.current_player_index}")
            self.game.current_player.start_turn()
            self.game.current_player.turn_number += 1
            self.debug_output(f"This is a new turn for player: {self.game.current_player.name}")
            # Hard stop if too many turns
            self.debug_output(f"TURN {self.game.current_player.turn_number}: Starting new turn")
            ##### CHANGE THIS IF NEEDED #####
            if self.game.current_player.turn_number >= 75:
                self.debug_output(f"Reached a hard stop of 75 turns")
                self.game.current_player.output(f"Reached turn 75")
                return self._get_observation(), -60, True, False, {}
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
            hand = self.game.current_player.piles[Piles.HAND]._cards
            self.debug_output(f"Hand is: {hand}")
            #stats = f"({self.game.current_player.get_score()} points, {self.game.current_player.count_cards()} cards)"
            #self.game.current_player.output(f"{self.game.current_player.name}'s Turn {stats}")

        # Get available options for ACTION and BUY phase
        options = self.game.current_player._choice_selection()
        self.debug_output(f"The number of valid options are: {len(options)}")
        self.debug_output(f"Options are {options}")
        #self.game.current_player.output(f"There are {len(options)} valid options: {options}")

        # Ensure an action is valid
        #  Note: This is not needed for PPO masking
        ##### CHANGE THIS IF NEEDED #####
        #if action >= len(options):
        #    return self._get_observation(), -50, False, False, {}
        #################################
        
        # If it's the Big Money bot's turn, select action automatically
        if self.current_player_index != self.learning_player_index:
            opt = self.big_money_strategy(options)
        else:
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
            self.game.current_player._used_actions += 1
        if self.game.current_player.buys < prev_num_buys:
            self.game.current_player._used_buys += 1

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
        self.debug_output(f'reward is: {reward}')
        
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

        # Calculate victory points gained from buying card
        bought = self.game.player_list()[self.learning_player_index].stats['bought']
        gained = self.game.player_list()[self.learning_player_index].stats['gained']
        victory_points_gained = (
            bought.count('Province') * 6 +
            bought.count('Duchy') * 3 +
            bought.count('Estate') * 1 -
            gained.count('Curse') * 1
        )

        # Rewards for termination
        if terminated:
            # Rewards for if agent won
            final_score_agent = self.game.player_list()[self.learning_player_index].get_score()
            final_score_bot = self.game.player_list()[1].get_score()
            n_turns = self.game.player_list()[self.learning_player_index].turn_number
            win_reward = 25

            if self._who_won():
                reward += ((victory_points_gained + win_reward) + (final_score_agent) - (n_turns*4) + ((final_score_agent - final_score_bot)*3))
                self.debug_output("Agent won!")
            else:
                reward += ((victory_points_gained - win_reward) + (final_score_agent) - (n_turns*4) + ((final_score_agent - final_score_bot)*3))
                self.debug_output("Agent lost :(")
            
            self.debug_output(f"Final scores are, AGENT({final_score_agent}), BOT({final_score_bot}), and reward is: {reward}")

        # No rewards when not agents turn
        elif self.current_player_index != self.learning_player_index:
            return 0

        # Otherwise incrimental rewards
        else:
            # Reward victory points bought during turn
            reward += victory_points_gained

            # Encourage increasing buying power
            total_coppers = self.count_card_type("Copper")
            total_silvers = self.count_card_type("Silver")
            total_golds = self.count_card_type("Gold")
            reward += total_silvers * 0.3
            reward += total_golds * 0.5

            # A slight penalty for too many coppers (deck clog)
            reward -= max(0, total_coppers - 7) * 0.3

            # Reward using up actions and buys by end of turn
            if self.game.current_player.phase == Phase.NONE:
                self.debug_output("Rewarding player at end of turn for used buys/actions")
                # Reward using buys
                reward += self.game.current_player._used_buys * 1
                # Reward using actions
                reward += self.game.current_player._used_actions * 1

        return reward
    #################################

    def big_money_strategy(self, options):
        """Play the big money strategy"""

        # Should never have any action cards, always choose to quit
        if self.game.current_player.phase == Phase.ACTION:
            opt = options[0]

        # Otherwise, in Buy Phase
        else:
            # Calculate the money
            money = self.count_money_in_hand()
            # How Pydominion is setup is strange, so need to be able to keep track of "spend" vs "buy"
            if money == 0 and self.spendall != 0:
                money = self.spendall
            self.debug_output(f"Player has {money} money")

            # Buy Province if it can
            if money >= 8:
                # Check to see if can just buy province from the list of options
                province_index = next(
                    (i for i, option in enumerate(options)
                     if option['selector'] != "-" and option['action'] == "buy" 
                     and option['name'] == "Province"),
                     0  # default if not found
                     )
                opt = options[province_index]
                self.spendall = 0
                self.debug_output(f"Choose to buy a province: {opt}")
                # If you can't, then need to first choose to spend your money
                if province_index == 0:
                    spend_index = next(
                    (i for i, option in enumerate(options)
                     if option['selector'] != "-" and option['action'] == "spendall"), 0  # default if not found
                     )
                    opt = options[spend_index]
                    self.spendall = money
                    self.debug_output(f"Didn't find a province, choose to spend money first: {opt}")
            # Buy a Gold if it can
            elif money >= 6:
                # Check to see if can just buy gold from the list of options
                gold_index = next(
                    (i for i, option in enumerate(options)
                     if option['selector'] != "-" and option['action'] == "buy" 
                     and option['name'] == "Gold"),
                     0  # default if not found
                     )
                opt = options[gold_index]
                self.spendall = 0
                self.debug_output(f"Choose to buy a Gold: {opt}")
                # If you can't, then need to first choose to spend your money
                if gold_index == 0:
                    spend_index = next(
                    (i for i, option in enumerate(options)
                     if option['selector'] != "-" and option['action'] == "spendall"), 0  # default if not found
                     )
                    opt = options[spend_index]
                    self.spendall = money
                    self.debug_output(f"Didn't find a gold, choose to spend money first: {opt}")
            # Buy a Silver if it can
            elif money >= 3:
                # Check to see if can just buy silver from the list of options
                silver_index = next(
                    (i for i, option in enumerate(options)
                     if option['selector'] != "-" and option['action'] == "buy" 
                     and option['name'] == "Silver"),
                     0  # default if not found
                     )
                opt = options[silver_index]
                self.spendall = 0
                self.debug_output(f"Choose to buy a Silver: {opt}")
                # If you can't, then need to first choose to spend your money
                if silver_index == 0:
                    spend_index = next(
                    (i for i, option in enumerate(options)
                     if option['selector'] != "-" and option['action'] == "spendall"), 0  # default if not found
                     )
                    opt = options[spend_index]
                    self.spendall = money
                    self.debug_output(f"Didn't find a silver, choose to spend money first: {opt}")
            # Dont buy anything
            else:
                opt = options[0]
                self.debug_output(f"Didnt buy anything, dont have the money")
        return opt

    def _who_won(self):
        """Determine if Player 0 won"""
        # Get the scores
        scores = self.game.whoWon()
        first_player = list(self.game.players.values())[self.learning_player_index]
        player_0_score = scores.get(first_player.name, self.learning_player_index)
        # Check if first player score is the maximum score
        # Note: Consider a tie a win...
        return player_0_score == max(scores.values())
    
    def debug_output(self, msg):
        """Print debug messages"""
        if self.debug:
            logging.debug(msg)

    def count_money_in_hand(self):
        """Get the amount of money in player hand"""
        money_values = {
            "Copper": 1,
            "Silver": 2,
            "Gold": 3,
        }

        total = 0
        hand = self.game.current_player.piles[Piles.HAND]._cards

        for card in hand:
            if card.name in money_values:
                total += money_values[card.name]
        
        return total
