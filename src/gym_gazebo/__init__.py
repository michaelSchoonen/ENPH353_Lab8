from .core.gazebo_env import GazeboEnv
from .envs.linefollower_env import LineFollowerEnv
from gymnasium.envs.registration import register

register(
    id='CartpoleEnv-v0',                                      
    entry_point='gym_gazebo.envs.cartpole_env:CartpoleEnv', # Registers the environment
    max_episode_steps=3000, # Value that set truncated to true. The maximum number of steps before reset                                 
)

