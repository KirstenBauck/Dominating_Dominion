import os
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3 import DQN

##### CHANGE THIS IF NEEDED #####
log_dir = "logs/dqn_1"
model_name = "dqn_dominion"
#################################

#Register the environment
gym.register(
    id="Dominion-v1",
    entry_point="DominionEnv:DominionEnv"
)

# Make environment space
env = gym.make("Dominion-v1", 
         num_players=2, 
         card_set=["Cellar", "Market", "Militia", "Mine", "Moat", 
                   "Remodel", "Smithy", "Village", "Throne Room", "Workshop"],
         quiet_flag=True,
         debug_flag = True,
         opponent = 'bot'
        )

# Set up logging to CSV and TensorBoard only
os.makedirs(log_dir, exist_ok=True)
logger = configure(log_dir, ["csv", "tensorboard"])

# Create and train the model
model = DQN("MlpPolicy", env, verbose=0)  # Turn off console spam
model.set_logger(logger)
model.learn(total_timesteps=1_000_000)

# Save trained model
model.save(os.path.join(log_dir, model_name))
