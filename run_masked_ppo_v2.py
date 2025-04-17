import os
import gymnasium as gym
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

##### CHANGE THIS IF NEEDED #####
log_dir = "logs/Rylan/masked_ppo_2_v2-2"
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
model = MaskablePPO("MlpPolicy", 
                    env, 
                    verbose=1, # Turn off console spam
                    # Network structure
                    policy_kwargs=dict(
                        net_arch=dict(
                            pi=[256, 128, 64],  # Policy network
                            vf=[256, 128, 64]   # Value function network
                        ),
                        activation_fn=torch.nn.ReLU
                    ),
                    # PPO parameters
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5) 
 
model.set_logger(logger)
model.learn(total_timesteps=1_000_000)

# Save trained model
model.save(os.path.join(log_dir, model_name))
