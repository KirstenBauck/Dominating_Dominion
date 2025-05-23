import os
import gymnasium as gym
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Where to log, change each time run
log_dir = "logs/masked_ppo_v6"
model_name = "ppo_masked_dominion"

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

# Mask the environment so that it only includes valid action choices
env = ActionMasker(env, lambda env: env.unwrapped.get_action_mask())


# Set up logging to CSV and TensorBoard only
os.makedirs(log_dir, exist_ok=True)
logger = configure(log_dir, ["csv"])

# Create and train the model
model = MaskablePPO("MlpPolicy", env, 
                    verbose=0, # Turn off console spam
                    # Network structure
                    policy_kwargs=dict(
                        net_arch=dict(
                            pi=[512, 256, 128],  # Deeper policy network
                            vf=[512, 256, 128]   # Deeper value network
                        ),
                        activation_fn=torch.nn.ReLU
                    ),
                    # PPO parameters
                    learning_rate=5e-5,  # Lower learning rate for more stability
                    n_steps=4096,        # Longer rollouts = More Context
                    batch_size=256,      # Smooth out the variance
                    n_epochs=15,
                    gamma=0.995,         # Keep long-term consequence relevant
                    gae_lambda=0.98,     # More reliance on long-term rewards
                    clip_range=0.3,      # Allow for more expressive updates
                    ent_coef=0.03,       # Encourage more exploration
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    normalize_advantage=True) # Help stabilize updates)

model.set_logger(logger)
model.learn(total_timesteps=1_000_000)

# Save trained model
model.save(os.path.join(log_dir, model_name))
