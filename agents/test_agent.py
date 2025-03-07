from dominion import Game
from dominion.Player import Player

# The cards we are choosing to start with
selected_cards = ["Cellar", "Market", "Militia", "Mine", "Moat", "Remodel",
                  "Smithy", "Village", "Woodcutter", "Workshop"]

def setup_game():
    game_args = {
        "initcards": selected_cards,
        "numplayers": 2
    }
    game = Game.Game(**game_args)
    # Add the custom bot (other player is assumed human right now)
    game.players[1] = MonteCarloBotPlayer(game) # Player 1 is bot
    return game


# I think this is how it could work after a cursrary review of the 
#  pydominion code
def run_game_with_agent():
    game = setup_game()
    turn = 0
    while not game.game_over:
        turn += 1
        if isinstance(game.current_player, MonteCarloBotPlayer):
            action = game.current_player.user_input(game.get_valid_actions(), "Choose action:")
            game.play_card(action)
        else:
            # Player turn
            game.play_turn()
        
        # Prevent eternal games
        if turn > 400:
            break
    game.whoWon()


# Make a separate class for each bot
class MonteCarloBotPlayer(Player):
    def __init__(self, game, name="MonteCarloBot", quiet=False, num_simulations=100, **kwargs):
        super().__init__(game, name, **kwargs)
        self.num_simulations = num_simulations
        self.quiet = quiet
    
    # Output for showing what is happening (reger to player implimentation)
    def output(self, msg: str, end: str = "\n") -> None:
        pass

    # I think we need a user_input (refer to player implimentation)
    def user_input(self, options, prompt:str):
        pass

    # Implimentation
    def monte_carlo_simulation(self, game_state):
        pass

    # Functions to help support impliment the MonteCarlo agent
        # Probably need to simulate and evaluate the game state

if __name__ == "__main__":
    run_game_with_agent()
