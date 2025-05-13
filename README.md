# Dominating Dominion: Training AI Agents to Play Dominion

# Pydominion
This project builds upon [pydominion](https://github.com/dwagon/pydominion), originally developed by Dougal Scott. The original repository is licensed under the MIT License.

## Base cards used
Cellar, Market, Militia, Mine, Moat, Remodel, Smithy, Village, Throne Room, and Workshop

# Training and versions
As much as possible, we have tried to document the process of building this environment. To view past versions of the agent look into the logs, see an overview of important changes in [log configurations](logs/log_configurations.md). One can play with an old agent using the associated `DominionEnv-v#.py` file using `test_env.ipynb`. (Note that the Dominion environment for versions 1 and 2 were not saved)

# References
- [pydominion](https://github.com/dwagon/pydominion)
- [Stable Baseline](https://stable-baselines3.readthedocs.io/en/master/)
- [Gymnasium Environment](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Dominion Rulebook](https://cdn.1j1ju.com/medias/59/e6/c2-dominion-rulebook.pdf)

# Diagrams
![Capstone Poster](diagrams/Poster.png?raw=true)
***Figure 1**: Capstone Poster*

![Environment Representation](diagrams/state_representation.png?raw=true)
***Figure 2**: Environment/State Representation (how the agent sees the game)*

![Action Space](diagrams/action_representation.png?raw=true)
***Figure 3**: Action Space Representation (how the agent plays the game)*

![PPO Architecture](diagrams/PPO.png?raw=true)
***Figure 4**: PPO Architecture*

![DQN Architecture](diagrams/DQN.png?raw=true)
***Figure 5**: DQN Architecture*