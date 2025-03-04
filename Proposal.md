## Outline

### Title and Abstract.

The title should be informative based on the project you intend to do. The abstract should contain 1) the data you are working with and how it is going to be collected, 2) the goal for how you will use the data if you collect everything you are hoping for, and 3) the goals for the project - what do you expect to accomplish? One paragraph summary (maximum 250 words, although I won't check).

### Introduction.

What data did you collect, why are you interested in it? If you have the data collected to some extent, show some examples of the data.frame in a properly formatted table (with legend) and indicate to me you have successfully collected data!

### Methods.

Discuss what you need to do to scrape/collect the data. You may explore any difficulties you think you may encounter here. Then pivot to what you plan to do with the data! What types of analysis do you want to try and use? What type of advanced statistical modeling may be necessary? If modeling is out of reach, when other ways do you plan to analyze the data?

### Expected Outcomes.

If everything goes correctly, what do you expect to find from this study?

### Group Considerations.

Provide an additional statement of expected work. What are the different tasks involved in this project and how do you see the work being distributed? In grants, this typically looks something like this. This work involves data collection (RB - 25%), data wrangling (RB - 100%), data modeling (RB - 75%), figure and table creation (RB - 40%), and final poster presentation (RB - 40%). Please work with your group members to determine a rough outline of who will do what work, and what contributions are expected to those sections. This does not have to be 100% accurate, but should reflect the relative workload of each student on the project, and should be evenly dispersed between students.

### Other things to consider.

If you have some data, you may prepare a preliminary figure. These are always very helpful when we submit grants, so if you have the time, think of one nice visualization that might help understand the problem at hand. I also encourage making a small table displaying a data.frame if you have data available. You could also consider a table of variables describing what you have collected and what the variables represent. If you do not have any data yet, you should spend significant time making sure you can get the data. These projects will not make it far without the data collection step being successful! Finally - if you get motivated - consider starting your GitHub repository for Project 2. This repository will show a lot of diversity in your work, and it is never to early to start! With group projects, you will be required to setup and share these GitHub repositories between group members.

## Actual Proposal

## **Dominating Dominion**

### **Abstract**

Our project's intent is to develop a reinforcement learning (RL) agent capable of effectively playing the board game *Dominion*. To achieve this, we will use *PyDominion* and *Gymnasium* to establish an environment where the agent can learn. The dataset consists of the game’s structured rules and objects which mainly comes from *PyDominion*. The primary goal is to develop a functional RL agent that can make strategic decisions within the game. Secondarily, we hope to employ different RL methodologies and compare their performance. This study will require developing an appropriate reward system and addressing integration challenges between *PyDominion* and the RL framework.

------------------------------------------------------------------------

### **Introduction**

The primary focus of this project is to explore reinforcement learning techniques by training an agent to play *Dominion*, a popular deck-building card game. *PyDominion* provides an implementation of the game in our language of choice, which serves as our foundation for creating a learning environment. By using *Gymnasium*, we can construct an RL-compatible system that allows the agent to iteratively improve its performance.

With the idea of developing an AI agent to play a complex board game, we settled on *Dominion* for it's strategic depth, complexity, and stochastic nature. The features come together to make a game that appropriately pushes the limits of our ability while staying within a what we think is a reasonable scope. Specifically, the non-determinism of the *Dominion*'s design adds an especially interesting element that makes for a more interesting problem. Our goal is to create an RL model that learns optimal strategies through repeated gameplay.

------------------------------------------------------------------------

### **Methods**

To build an RL agent for *Dominion*, we plan to approach it in these stages:

1.  **Environment Setup**: We will integrate *PyDominion* with *Gymnasium*, to create an environment for reinforcement learning frameworks. This requires defining the observation space (game-state representation) and the action space (available moves).\
2.  **Reward System Design**: Creating a robust reward function is integral to guiding the agent’s learning. While the exact structure is yet to be determined, and will likely need tuning along the way, potential reward mechanisms could involve victory points, deck composition, and game progression.\
3.  **Implement the RL Agent**: We will explore RL methodologies to find good fits for our project. Currently, Deep Q-Networks (DQN) and Monte Carlo Tree Search (MCTS) are appealing but others may be selected depending on our findings. We will then implement these methods as an actor in our environment.\
4.  **Training the RL Agent**: We will run simulations in the environment to allow the RL agent to learn the game. This will likely involve tweaking our agent implementations and reward systems.\
5.  **Evaluation and Analysis**: The agent’s performance will be measured by its win rate against baseline strategies (e.g., random moves). If time permits, we will compare the effectiveness of different RL approaches.

**Challenges we anticipate include:**\
- Correctly modeling *Dominion* as an RL environment.\
- Properly establishing effective training given the game’s complexity.\
- Tuning the reward systems to encourage optimal play.

------------------------------------------------------------------------

### **Expected Outcomes**

If the project is successful, we expect to develop an RL agent capable of playing *Dominion* at a competitive level. The agent should demonstrate an understanding of strategic decision-making, such as optimizing deck composition and selecting strong card combinations. Ideally, we will also be able to compare different RL methods to determine their effectiveness in a game like *Dominion*.

------------------------------------------------------------------------

### **Group Considerations**

(*TBD*)

------------------------------------------------------------------------

### **Other things to consider**

One important factor in our developing this model is the game-state representation. The table below provides an example of how the agent will perceive and interact with the game environment:

(*NEEDS REFINING, VERY IMPORTANT*)

| Feature | Example Value | Description |
|------------------|---------------------|---------------------------------|
| Entire Deck | 5 Estates, 7 Coppers | Cards in the player's entire collection |
| Current Deck | 2 Estates, 4 Coppers | Cards currently in the player's deck |
| Hand | Smithy, Silver, Copper | Cards available for play this turn |
| Supply Cards | Village (8), Market (10) | Remaining cards available for purchase |
| Victory Points | 3 VP | Player’s current victory point total |
| Actions/Draws Left | 1 action, 2 draws | Available turn actions |

This structured format ensures that the RL agent receives well-defined inputs, facilitating effective decision-making.

(*Could potentially create a preliminary reward function chart in a similar manner*)

<https://github.com/KirstenBauck/Dominating_Dominion>
