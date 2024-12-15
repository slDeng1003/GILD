# Enhancing Online Reinforcement Learning with Meta-Learned Objective from Offline Data

Official code for *Enhancing Online Reinforcement Learning with Meta-Learned Objective from Offline Data, AAAI 2025 (accept)*

GILD is intended for three  vanilla off-policy RL methods, which are DDPG, TD3 and SAC.

To run experiments, you will need to install the following packages preferably in a Anaconda virtual environment:

- python >= 3.7
- pytorch 1.10.0
- gym 0.21.0
- mujoco-py 2.1.2.14
- mujoco-210
- learn2learn

To install the required packages, you can simply execute the following command:

```
pip install -r requirements.txt
```

To run the code with the default parameters and environments, simply execute the following commands for DDPG+GILD, TD3+GILD and SAC+GILD:

To reduce run time while remaining performance, we highly recommend an 1,000 warm-start (1% of total training) steps for GILD.

```
# For vanilla  DDPG  method
python main_DDPG.py --method=DDPG_GILD_ws

# For vanilla TD3 method
python main_TD3.py --method=TD3_GILD_ws

# For vanilla SAC method
python main_SAC.py --method=SAC_GILD_ws
```

For GILD without warm-start, run commands like:

```
# For vanilla DDPG method
python main_DDPG.py --method=DDPG_GILD

# For vanilla TD3 method
python main_TD3.py --method=TD3_GILD

# For vanilla SAC method
python main_SAC.py --method=SAC_GILD
```

For more environments including: Hopper-v2 (default), Walker2d-v2, HalfCheetah-v2 and Ant-v2, run a single line of command (without comments) as follows:

```
# ------------- For vanilla DDPG method -------------
# For DDPG+GILD+ws in Hopper-v2
python main_DDPG.py --method=DDPG_GILD_ws --env_name=Hopper-v2

# For DDPG+GILD+ws in Walker2d-v2
python main_DDPG.py --method=DDPG_GILD_ws --env_name=Walker2d-v2

# For DDPG+GILD+ws in HalfCheetah-v2
python main_DDPG.py --method=DDPG_GILD_ws --env_name=HalfCheetah-v2

# For DDPG+GILD+ws in Ant-v2
python main_DDPG.py --method=DDPG_GILD_ws --env_name=Ant-v2

# ------------- For vanilla TD3 method -------------
# For TD3+GILD+ws in Hopper-v2
python main_TD3.py --method=TD3_GILD_ws --env_name=Hopper-v2

# For TD3+GILD+ws in Walker2d-v2
python main_TD3.py --method=TD3_GILD_ws --env_name=Walker2d-v2

# For TD3+GILD+ws in HalfCheetah-v2
python main_TD3.py --method=TD3_GILD_ws --env_name=HalfCheetah-v2

# For TD3+GILD+ws in in Ant-v2
python main_TD3.py --method=TD3_GILD_ws --env_name=Ant-v2

# ------------- For vanilla SAC method -------------
# For SAC+GILD+ws in Hopper-v2
python main_SAC.py --method=SAC_GILD_ws --env_name=Hopper-v2

# For SAC+GILD+ws in Walker2d-v2
python main_SAC.py --method=SAC_GILD_ws --env_name=Walker2d-v2

# For SAC+GILD+ws in HalfCheetah-v2
python main_SAC.py --method=SAC_GILD_ws --env_name=HalfCheetah-v2

# For SAC+GILD+ws in Ant-v2
python main_SAC.py --method=SAC_GILD_ws --env_name=Ant-v2
```

