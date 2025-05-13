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

    def __init__(self, num_players=2, card_set=None, quiet_flag=True, debug_flag=False, opponent='bot'):
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
        self.terminated = False # Is game over?
        self.opponent = opponent

        # Initialize Dominion Base game
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
        
        #TODO: Change this for better representation if possible, so big :(
        self.observation_space = spaces.Box(
            low=0,
            high=75,  # Maximum number of turns (which would be the max)
            shape=((len(self.cards) * 7) + 7,),  # 7 zones: hand, duration, deck, played, discard, supply, trash
            dtype=np.int32
        )
                
        # Define action space
            # ACTION phase: 10 action cards + end phase = 11
            # BUY phase: 16 buyable cards + end phase = 17
            # TODO: After looking over logs, the maximum valid moves always seems to be 23...
        self.action_space = spaces.Discrete(28)
        
    def reset(self, seed=None, options: Optional[dict] = None):
        """Reset the dominion game"""
        super().reset(seed=seed)
        self.debug_output(f"\nResetting the game")
        self.game.game_over = False
        self.game.players.clear()
        self.game.start_game()
        self.game.current_player.phase = Phase.NONE
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get the current state of the game as a 1D numpy array"""
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

        # Initialize observation with basic resource values
        observation = [actions, buys, coins, deck_size, score, turn_number, phase_id]

        # Card counts per pile (hand, duration, deck, played, discard)
        for pile_name in [Piles.HAND, Piles.DURATION, Piles.DECK, Piles.PLAYED, Piles.DISCARD]:
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

        # Start new turn and check that agent hasn't taken too many turns
        self._start_new_turn_if_needed()
        if self.game.current_player.turn_number >= 75:
            self.debug_output(f"Reached a hard stop of 75 turns")
            return self._get_observation(), -60, True, False, {}

        # Make agent choose it's action and check if the action ended the game
        self._take_action_check_termination(action)

        # Play the bots turn immediately after cleanup phase of agent
        if self.game.current_player.phase == Phase.CLEANUP:
            self._handle_cleanup_and_bot_turn()
        
        # If needed, do any special actions that occur at the end of the game
        if self.terminated:
            self.game.game_over = True
            for plr in self.game.player_list():
                plr.game_over()
            self.game.current_player.output(f"End of game cards are: {self.game.current_player.end_of_game_cards}")

        # Calculate the reward for training
        reward = self.calculate_reward()
        self.debug_output(f'reward is: {reward}')
        
        return self._get_observation(), reward, self.terminated, False, {}


################################## Helper Functions ###################################################

    def debug_output(self, msg):
        """Print debug messages"""
        if self.debug:
            logging.debug(msg)

    ########### PPO Action Masking ###############
    def get_action_mask(self):
        """Make a binary mask of the valid actions (1 = valid, 0=invalid)"""
        options = self.game.current_player._choice_selection()
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        # Get only the valid moves from the list of options
        for i, option in enumerate(options):
            if option['selector'] != "-":
                mask[i] = 1
        return mask
    ##############################################

    ########## Calculating Reward ################
    def calculate_reward(self):
        """Calculate the reward function for agent"""        
        reward = 0

        # Calculate victory points gained from buying card
        bought = self.game.player_list()[self.learning_player_index].stats['bought']
        self.debug_output(f"Player bought: {bought}")
        gained = self.game.player_list()[self.learning_player_index].stats['gained']
        victory_points_gained = (
            bought.count('Province') * 6 +
            bought.count('Duchy') * 3 +
            bought.count('Estate') * 1 -
            gained.count('Curse') * 1
        )

        # Rewards for termination
        if self.terminated:
            # Rewards for if agent won
            final_score_agent = self.game.player_list()[self.learning_player_index].get_score()
            final_score_bot = self.game.player_list()[1].get_score()
            n_turns = self.game.player_list()[self.learning_player_index].turn_number
            win_reward = 25

            if self._who_won():
                reward += ((victory_points_gained + win_reward) + (final_score_agent) - (n_turns*4) + ((final_score_agent - final_score_bot)*2))
                self.debug_output("Agent won!")
            else:
                reward += ((victory_points_gained - win_reward) + (final_score_agent) - (n_turns*4) + ((final_score_agent - final_score_bot)*2))
                self.debug_output("Agent lost :(")
            
            self.debug_output(f"Final scores are, AGENT({final_score_agent}), BOT({final_score_bot}), and reward is: {reward}")

        # Otherwise incremental rewards
        else:
            # Reward victory points bought during turn
            reward += victory_points_gained

            # Encourage increasing buying power
            total_coppers = self._count_card_type("Copper")
            total_silvers = self._count_card_type("Silver")
            total_golds = self._count_card_type("Gold")
            reward += total_silvers * 0.5
            reward += total_golds * 1
            # A slight penalty for too many coppers (deck clog)
            reward -= max(0, total_coppers - 7) * 0.3

            # Reward using more actions and buys by end of turn
            if self.game.current_player.phase == Phase.NONE:
                reward += self.game.current_player._used_buys * 1
                reward += self.game.current_player._used_actions * 0.4
            
            # Reward Engine
            last_card = bought[-1]
            self.debug_output(f"Last card bought is: {last_card}")
            if last_card in ["Village", "Market", "Smithy"]:
                reward += 0.4  # Higher reward for key engine components
            elif last_card == "Throne Room":
                reward += 0.5  # High value for combo enablers

            # VP cards with diminishing returns based on timing
            if last_card == "Province":
                # Provinces more valuable later in game
                turn_factor = min(1.0, self.game.current_player.turn_number / 15)
                reward += 0.6 + (turn_factor * 0.4)
            elif last_card == "Duchy":
                # Duchies valuable in mid-to-late game
                mid_game_factor = min(1.0, self.game.current_player.turn_number / 12)
                reward += 0.2 + (mid_game_factor * 0.3)
            
            # Small time penalty
            reward -= 0.1

        return reward

    def _count_card_type(self, card_name):
        """Counts the total number of a specific card across all piles."""
        return sum(deck.count(card_name) for deck in self.game.current_player.piles.values())

    def _who_won(self):
        """Determine if Player 0 won (agent), a tie is considered a win"""
        scores = self.game.whoWon()
        first_player = list(self.game.players.values())[self.learning_player_index]
        player_0_score = scores.get(first_player.name, self.learning_player_index)
        return player_0_score == max(scores.values())
    ######################################################
    
    ######## Opponent Option Choosing Strategy ###########

    def big_money_strategy(self, options):
        """
        A bot that play's Dominion's big money strategy. Buy Province with 8 money.
        Buy Gold with 6-7 money, buy Silver with 3-5 money and don't buy anything else.

        Args:
            options (list): List of possible actions for the player
        Returns:
            Option: The option to take based on "Big Money" Strategy rules
        """

        # Should never have any action cards with this strategy, always choose to quit
        if self.game.current_player.phase == Phase.ACTION:
            return options[0]

        # In BUY Phase
        money = self._count_money_in_hand()
        # How Pydominion is setup is strange, so need to be able to keep track of "spend" vs "buy"
        if money == 0 and self.spendall != 0:
            money = self.spendall
        self.debug_output(f"Player has {money} money")

        card_priority = [("Province", 8), ("Gold", 6), ("Silver", 3)]
        for card_name, cost in card_priority:
            # Check to see if bit can just buy the card from list of options
            if money >= cost:
                opt = self._select_buy_option(options, card_name)
                if opt:
                    self.spendall = 0
                    self.debug_output(f"Chose to buy {card_name}: {opt}")
                    return opt
                # If bot can't, than need to first choose to spend the money
                else:
                    spend_opt = self._select_spendall_option(options)
                    if spend_opt:
                        self.spendall = money
                        self.debug_output(f"Need to choose to spend first to gain {card_name} with {spend_opt}")
                        return spend_opt
        
        # Didn't have enough money, default to do nothing
        self.debug_output(f"Didnt buy anything, dont have the money")
        return options[0]

    def _count_money_in_hand(self):
        """Get the amount of money in a player's hand"""
        money_values = {
            "Copper": 1,
            "Silver": 2,
            "Gold": 3,
        }
        total = 0
        # Get all the cards in a players hand
        hand = self.game.current_player.piles[Piles.HAND]._cards
        # If that card has money value, add to total
        for card in hand:
            if card.name in money_values:
                total += money_values[card.name]
        return total

    def _select_buy_option(self, options, card_name):
        """
        Select a buy option for a specific card if available

        Args:
            options (list): List of possible actions for the player
            card_name (str): Name of the card to buy
        Returns:
            Option: The Option associated with buying specified card in PyDominion
        """
        for opt in options:
            if (
                opt['selector'] != "-"
                and opt['action'] == "buy"
                and opt['name'] == card_name
            ):
                return opt
        return None

    def _select_spendall_option(self, options):
        """
        Select the spendall option if available

        Args:
            options (list): List of possible actions for the player
        Returns:
            Option: The "spendall" money option in PyDominion, if not available, defaults to None
        """
        for opt in options:
            if opt['selector'] != "-" and opt['action'] == "spendall":
                return opt
        return None
    
    def human_player(self, options):
        """
        Play the game against a human player

        Args:
            options (list): List of possible actions for the player
        Returns:
            Option: The option to take based on player input
        """

        prompt = self.game.current_player._generate_prompt()
        self.game.current_player.output("\n##### Choose from the options below #####")
        opt = self.game.current_player.user_input(options, prompt)
        return opt
    #####################################################

    def _play_bot_turn(self):
        """Play the entirety of the big moneys bot's turn"""
        player = self.game.current_player
        player.start_turn()
        player.turn_number += 1
        player.phase = Phase.ACTION
        self.debug_output(f"########### BOT TURN ({player.name}) ###########")
        self.debug_output(f"********* ACTION PHASE *********")
        self.debug_output(f"Hand is: {self.game.current_player.piles[Piles.HAND]._cards}")

        # Make choice selection for ACTION and BUY phase based on bot
        while player.phase != Phase.NONE:
            options = player._choice_selection()
            if self.opponent == 'bot':
                opt = self.big_money_strategy(options) # Use Big Money Strategy
            elif self.opponent == 'human':
                opt = self.human_player(options) # Have human play against
            player._perform_action(opt)

            if opt["action"] in ["quit", None]:
                self._advance_phase(player)
            
            if self.game.isGameOver():
                player.phase = Phase.CLEANUP
                self.terminated=True

            if player.phase == Phase.CLEANUP:
                self._complete_cleanup_phase(player)
                break

    def _switch_to_next_player(self):
        """Switch between the players"""
        self.game.current_player = self.game.player_to_left(self.game.current_player)
        self.current_player_index = self.game.player_list().index(self.game.current_player)

    def _complete_cleanup_phase(self, player):
        """Go through the cleanup phase for a player"""
        self.debug_output(f"********* CLEANUP PHASE *********")
        self.debug_output(f"Ending Turn {self.game.current_player.name}")
        player.cleanup_phase()
        player._card_check()
        player.end_turn()
        self.game._validate_cards()
        self.game._turns.append(player.uuid)
        player.phase = Phase.NONE

    def _handle_cleanup_and_bot_turn(self):
        """
        When the agent is in it's cleanup phase, play the cleanup phase
        and play the bots turn
        """
        self._complete_cleanup_phase(self.game.current_player)
        self.game._validate_cards()
        self._switch_to_next_player()

        # Take the bot's full turn
        if self.current_player_index != self.learning_player_index:
            self._play_bot_turn()

        # Switch back to agent's turn
        self._switch_to_next_player()
    
    def _advance_phase(self, player):
        """Switch from action phase to buy phase and buy phase to cleanup phase"""
        # Transition from ACTION to BUY
        if player.phase == Phase.ACTION:
            self.debug_output(f"********* BUY PHASE *********")
            player.phase = Phase.BUY
        # Transition from BUY to CLEANUP
        elif player.phase == Phase.BUY:
            player.hook_end_buy_phase() # Ensure any cards that have effects are triggered
            player.phase = Phase.CLEANUP

    def _start_new_turn_if_needed(self):
        """Handles agent turn set up"""
        # Check if this is a new turn and start it if it is
            #  Phase.NONE to represents when a player ends their turn
        if self.game.current_player.phase == Phase.NONE:
            self.game._validate_cards()
            self.game.current_player.start_turn()
            self.game.current_player.turn_number += 1
            self.debug_output(f"\nTURN {self.game.current_player.turn_number}: Starting new turn")
            # Count the number of used buys and used actions
            self.game.current_player._used_buys = 0
            self.game.current_player._used_actions = 0
            # Change to action phase to start turn
            self.game.current_player.phase = Phase.ACTION
            # Debug and logging output
            self.debug_output(f"********* ACTION PHASE *********")
            self.game.current_player.output(f"{'#' * 30} Turn {self.game.current_player.turn_number} {'#' * 30}")
            hand = self.game.current_player.piles[Piles.HAND]._cards
            self.debug_output(f"Hand is: {hand}")

    def _take_action_check_termination(self, action):
        """
        Have the agent choose what the option for the given phase.
        Also check if the game has been terminated. 
        """
        # Get available options for ACTION and BUY phase
        options = self.game.current_player._choice_selection()
        self.debug_output(f"{len(options)} options: {options}")

        # Choose the option
        opt = options[action]
        self.debug_output(f"Chosen Option: {opt['verb']} ({action})")

        # Update used buys and actions
        prev_num_actions = int(self.game.current_player.actions)
        prev_num_buys = int(self.game.current_player.buys)

        # Perform the action
        self.game.current_player._perform_action(opt)
        self.debug_output(f"Performed action {opt['action']}")

        # Update used buys and actions
        if self.game.current_player.actions < prev_num_actions:
            self.game.current_player._used_actions += 1
        if self.game.current_player.buys < prev_num_buys:
            self.game.current_player._used_buys += 1

        # Check if they buy a card that triggers the end of game
        self.terminated = self.game.isGameOver()
        if self.terminated:
            # They still need to enter cleanup phase for full completion
            self.debug_output(f"Buying a card triggered end of game")
            self.game.current_player.phase = Phase.CLEANUP

        # Check if agent's phase should transition
        if opt["action"] in ["quit", None]:  
            self._advance_phase(self.game.current_player)
