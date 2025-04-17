import os
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv
from multiprocessing import freeze_support

# Directories
log_dir = "logs/Rylan/optimized_ppo"
model_name = "ppo_masked_dominion_optimized"
os.makedirs(log_dir, exist_ok=True)

# Register the environment
gym.register(
    id="Dominion-v1",
    entry_point="DominionEnv:DominionEnv"
)

# Create environment function
def make_env(seed=0):
    def _init():
        env = gym.make("Dominion-v1", 
                num_players=2, 
                card_set=["Cellar", "Market", "Militia", "Mine", "Moat", 
                        "Remodel", "Smithy", "Village", "Throne Room", "Workshop"],
                quiet_flag=True,
                debug_flag=False
                )
        env = ActionMasker(env, lambda env: env.unwrapped.get_action_mask())
        env.reset(seed=seed)
        return env
    return _init

def main():
    # Number of parallel environments
    n_envs = 4
    
    # Create a list of environment creation functions with different seeds
    env_fns = [make_env(seed=i) for i in range(n_envs)]
    
    # Create vectorized environment with SubprocVecEnv for true parallelism
    envs = SubprocVecEnv(env_fns)
    
    # Setup evaluation environment
    eval_env = gym.make("Dominion-v1", 
                num_players=2, 
                card_set=["Cellar", "Market", "Militia", "Mine", "Moat", 
                        "Remodel", "Smithy", "Village", "Throne Room", "Workshop"],
                quiet_flag=True,
                debug_flag=False
               )
    eval_env = ActionMasker(eval_env, lambda env: env.unwrapped.get_action_mask())
    eval_env = Monitor(eval_env)
    
    # Set up logging
    logger = configure(log_dir, ["csv", "tensorboard"])
    
    # Callbacks for evaluation and checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50k steps
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="ppo_dominion"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=10000,   # Evaluate every 10k steps
        n_eval_episodes=5,
        deterministic=True
    )
    
    # Optimized PPO hyperparameters for card games
    model = MaskablePPO(
        "MlpPolicy", 
        envs,
        verbose=1,
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
        max_grad_norm=0.5
    )
    
    # Set logger
    model.set_logger(logger)
    
    # Train with callbacks
    model.learn(
        total_timesteps=1000000,  # Increased timesteps since we're using multiple environments
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save trained model
    model.save(os.path.join(log_dir, model_name))
    
    # Close environments properly
    envs.close()

if __name__ == "__main__":
    # This is required for Windows multiprocessing
    freeze_support()
    main()