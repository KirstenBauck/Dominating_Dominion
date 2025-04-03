# Dominating Dominion: Training AI Agents to Play Dominion

# Pydominion
This project builds upon [pydominion](https://github.com/dwagon/pydominion), originally developed by Dougal Scott. The original repository is licensed under the MIT License.

# Install Environment
1. Clone the repository: `git clone --recursive https://github.com/KirstenBauck/Dominating_Dominion.git`
    - If cloned without  `--recursive` just run `git submodule update --init --recursive` to be able to import pydominion
2. Create conda environment: `conda env create -f environment.yml`
3. Activate conda environment: `conda activate RL_dominion`
4. Check correct installation by running: `python -c "import dominion; print('pydominion is correctly installed!')"`

FYI: Code Kirsten ran to setup environment:
- `git submodule add https://github.com/dwagon/pydominion.git external/pydominion`
- `git submodule update --init --recursive`
- `pip install -e external/pydominion`

## Base cards used
Cellar, Market, Militia, Mine, Moat, Remodel, Smithy, Village, Throne Room, and Workshop

# References
- [pydominion](https://github.com/dwagon/pydominion)
- [Stable Baseline](https://stable-baselines3.readthedocs.io/en/master/)
- [Dominion Rulebook](https://cdn.1j1ju.com/medias/59/e6/c2-dominion-rulebook.pdf)

# Diagrams
![alt text](diagrams/state_representation.png?raw=true)
***Figure 1**: Environment/State Representation (how the agent sees the game)*

![DQN Architecture](diagrams/DQN.png?raw=true)
***Figure 2**: DQN Architecture*