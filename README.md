# Dominating Dominion: Training AI Agents to Play Dominion

# Pydominion
This project builds upon [pydominion](https://github.com/dwagon/pydominion), originally developed by Dougal Scott. The original repository is licensed under the MIT License.

# Install Environment
1. Clone the repository: `git clone --recursive https://github.com/KirstenBauck/Dominating_Dominion.git`
    - If cloned without  `--recursive` just run `git submodule update --init --recursive` to be able to import pydominion
2. Create conda environment: `conda env create -f environment.yml`
3. Activate conda environment: `conda activate RL_dominion`

FYI: Code Kirsten ran to setup environment:
- `git submodule add https://github.com/dwagon/pydominion.git external/pydominion`
- `git submodule update --init --recursive`
- `pip install -e external/pydominion`

# Base 10 cards to use (this is the recommended first set)
Cellar, Market, Militia, Mine, Moat, Remodel, Smithy, Village, Woodcutter, and Workshop

# Notes/articles
- [Dominion Rulebook](https://cdn.1j1ju.com/medias/59/e6/c2-dominion-rulebook.pdf)
- [Walkthrough](https://ianwdavis.com/dominion.html) of how one person went about AI Dominion implementation
- [Research Paper 1](https://cs230.stanford.edu/projects_fall_2019/reports/26260348.pdf)
- [Recent Gradiant Descent](https://johnchenresearch.github.io/demon/?ref=ruder.io) <-- Look into using YellowFin

# TODO
- Morning of 3/19: Each person tries to get PyDominon to work (Claire, Kirsten)
- 3/21: Register for conference
- Visualize environment
- Diagram of agent
