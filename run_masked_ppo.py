import os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

##### CHANGE THIS IF NEEDED #####
log_dir = "logs/masked_ppo_5"
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
         debug_flag = True
        )

# Mask the environment so that it only includes valid action choices
env = ActionMasker(env, lambda env: env.unwrapped.get_action_mask())


# Set up logging to CSV and TensorBoard only
os.makedirs(log_dir, exist_ok=True)
logger = configure(log_dir, ["csv"])

# Create and train the model
# If agent is short term greedy: Raise gamma or gaw_lambda
# IF agent always does the same thing: Raise ent_coef
policy_kwargs=dict(net_arch=[256, 256])
model = MaskablePPO("MlpPolicy", env, verbose=0,
                    ent_coef = 0.03, # Encourage more exploration
                    n_steps=4096, # Longer rollouts = More context
                    batch_size=256, # Smooth out the variance
                    gae_lambda=0.98, # More reliance on long-term rewards
                    clip_range=0.3, # Allow for more expressive updates
                    normalize_advantage=True, # Help stabilize updates
                    learning_rate = 2.5e-4, # Make updates finer
                    gamma = 0.995, # Keep long-term consequence relevant
                    policy_kwargs = policy_kwargs 
                    )
model.set_logger(logger)
model.learn(total_timesteps=1_000_000)

# Save trained model
model.save(os.path.join(log_dir, model_name))
