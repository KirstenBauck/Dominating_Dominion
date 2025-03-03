# From Shuffling to Ruling: An AI’s Journey to Dominion Domination

# Pydominion
This project builds upon [pydominion](https://github.com/dwagon/pydominion), originally developed by Dougal Scott. The original repository is licensed under the MIT License.

Need to run:
`git submodule update --init --recursive` to be able ti import pydominion


Code Kirsten ran
`git submodule add https://github.com/dwagon/pydominion.git external/pydominion`
`git submodule update --init --recursive`
`pip install -e external/pydominion`


# Install Environment
`conda env create -f environment.yml`

**Goal:** Get the most points

**End Condition:** Run out of provinces

**Challenges**:

# Reward Function
- Each turn is a malus
- Victory points are bonus

# One player
- Want to maximise the points

# Multi-player
- Want to maximize the difference between players

# Base 10 cards to use (this is the recommended first set)
Cellar, Market, Militia, Mine, Moat, Remodel, Smithy, Village, Woodcutter, and Workshop


# Notes/articles
- [Dominion Rulebook](https://cdn.1j1ju.com/medias/59/e6/c2-dominion-rulebook.pdf)
- [Github Possible Implimentation](https://github.com/dwagon/pydominion/tree/main/dominion) to look into
- [Walkthrough](https://ianwdavis.com/dominion.html) of how one person went about AI Dominion implimentation
- [Research Paper 1](https://cs230.stanford.edu/projects_fall_2019/reports/26260348.pdf)
- [Recent Gradiant Descent](https://johnchenresearch.github.io/demon/?ref=ruder.io) <-- Look into using YellowFin

# Gymnasium
- [Creating an environment](https://gymnasium.farama.org/introduction/create_custom_env/)
  - [Spaces](https://gymnasium.farama.org/api/spaces/)
