import os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

##### CHANGE THIS IF NEEDED #####
log_dir = "logs/Rylan/masked_ppo_2"
model_name = "ppo_masked_dominion"
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
         debug_flag = False
        )

# Mask the environment so that it only includes valid action choices
env = ActionMasker(env, lambda env: env.unwrapped.get_action_mask())


# Set up logging to CSV and TensorBoard only
os.makedirs(log_dir, exist_ok=True)
logger = configure(log_dir, ["csv", "tensorboard"])

# Create and train the model
model = MaskablePPO("MlpPolicy", env, verbose=0)  # Turn off console spam
model.set_logger(logger)
model.learn(total_timesteps=1_000_000)

# Save trained model
model.save(os.path.join(log_dir, model_name))
