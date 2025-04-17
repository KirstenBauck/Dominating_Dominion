import os
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from multiprocessing import freeze_support

##### CHANGE THIS IF NEEDED #####
log_dir = "logs/Rylan/masked_ppo_2_v3-2"
model_name = "ppo_masked_dominion"
#################################

# Register the environment
gym.register(
    id="Dominion-v1",
    entry_point="DominionEnv:DominionEnv"
)

# Make environment space
def make_env(rank=0):
    """Create a callable function for environment creation"""
    def _init():
        try:
            # Create the environment
            env = gym.make("Dominion-v1", 
                     num_players=2, 
                     card_set=["Cellar", "Market", "Militia", "Mine", "Moat", 
                               "Remodel", "Smithy", "Village", "Throne Room", "Workshop"],
                     quiet_flag=True,
                     debug_flag=False
                    )
            
            # Apply action masking wrapper
            env = ActionMasker(env, lambda env: env.unwrapped.get_action_mask())
            
            # Initialize the environment with seed
            # Use the modern method for gymnasium
            env.reset(seed=rank)
            
            return env
        except Exception as e:
            print(f"Error in worker process {rank}: {e}")
            raise
    return _init

if __name__ == "__main__":
    # This is crucial for multiprocessing to work correctly
    freeze_support()
    
    # Set up logging
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["csv", "tensorboard"])
    
    # First test the environment creation to catch any errors
    try:
        test_env = make_env(0)()
        test_obs, _ = test_env.reset()
        print(f"Environment initialization successful. Observation shape: {test_obs.shape}")
        del test_env
    except Exception as e:
        print(f"Error during environment test: {e}")
        raise
    
    # Create parallel environments
    num_envs = 4  # Start with a modest number
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # Set start method for multiprocessing if needed
    # On Windows, this should be 'spawn'
    try:
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
        vec_env = VecMonitor(vec_env, log_dir)
    except Exception as e:
        print(f"Error creating vectorized environments: {e}")
        # Fall back to single environment if parallelism fails
        print("Falling back to single environment")
        single_env = make_env(0)()
        single_env = Monitor(single_env, log_dir)
        vec_env = single_env
    
    # Create and train the model
    model = MaskablePPO("MlpPolicy", 
                        vec_env, 
                        verbose=1,
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
                        n_steps=1024,        # Reduced per-env steps (total = n_steps * num_envs)
                        batch_size=128,
                        n_epochs=15,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        ent_coef=0.02,       # Slightly higher entropy for more exploration
                        vf_coef=0.5,
                        max_grad_norm=0.5)
    
    model.set_logger(logger)
    model.learn(total_timesteps=1_000_000)
    
    # Save trained model
    model.save(os.path.join(log_dir, model_name))