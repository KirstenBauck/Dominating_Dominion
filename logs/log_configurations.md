|  | masked_ppo_v1 | masked_ppo_v2 | masked_ppo_v3 | masked_ppo_v4 | masked_ppo_v5
| --- | --- | --- | --- | --- | --- |
| Agent vs __ | Agent | Agent | Agent | Agent | Bot |
| Observation Space | hand pile, duration pile, defer pile, deck pile, played pile, discard pile, supply pile, trash pile | hand pile, duration pile, defer pile, deck pile, played pile, discard pile, supply pile, trash pile | actions, buys, coins, phase, supply pile, played pile, discard pile, deck size, current score, turn number | actions, buys, coins, phase, supply pile, played pile, discard pile, deck size, current score, turn number | actions, buys, coins, phase, deck_size, score, turn_number, current_player_index, agent_index, hand pile, duration pile, defer pile, deck pile, played pile, discard pile, supply pile, trash pile |
| Reward Function | +/- 100 for win/loss, + agent points * 10, victory points, per turn penalty, end turn with 0 actions | win/loss (+/- win_reward) = ((victory_points_gained + win_reward) + (final_score * 10) - (n_turns * 2)), victory points gained, encourage buying power, using up actions and buys | +/- 10 for win/loss, final_score * 0.5, actions_used * 0.02, buys_used * 0.03 | +/- 10 for win/loss, final_score * 0.5, 0.3 for buying Gold, 0.15 for buying Silver | +/- 50 for win/loss, score differences between agent and opponent * 2, Reward engine for last card bought (Village,Market,Smith + 0.4, Throne Room + 0.5, Provinces valuable later, duchies valuable mid-to-late game) | win/loss = ((victory_points_gained - win_reward) + (final_score_agent) - (n_turns*4) + ((final_score_agent - final_score_bot)*2)), victory points gained in turn, reward for buying silver and gold, reward using more actions and buys |
| Other things implemented  | - | - | ? | If no provinces bought after 30 turns ended the game | Played against Big Money Bot (51 pts) |
| Agent Points | 3 | 10 | 7 | 3 | 11 |
| Num turns | 20 | 69 (50% reaches 75 game limit) | 40 | 18 | 25 |
| Game Over | Cellar, Moat, Remodel piles empty |  Cellar, Estate, Silver piles empty | Copper, Estate, Cellar piles empty | Cellar, Moat, Remodel piles empty | Province pile empty |
| Game observations | - | End up with 60% of deck being copper and silver | | | |
| Notes on graphs | ✅ | ⚠️ Loss and Entropy loss graphs not the best | ⚠️ Loss graph went straight to 0, KL Divergence didn't quite converge to 0, otherwise ok | ⚠️ Strange convergence in KL Divergence | ✅ But variable mean episode length |
| Network structure | - | - | [256, 128, 64], activation_fn=torch.nn.ReLU | [512, 256, 128], activation_fn=torch.nn.ReLU | [256, 256] |
| ent_coef | - | - | 0.01 | 0.02 | 0.03 |
| learning_rate | - | - | 3e-4 | 5e-5 | 2.5e-4 |
| batch_size | - | - | 64 | 128 | 256 |
| gamma | - | - | 0.99 | 0.99 | 0.995 |
| gae_lambda | - | - | 0.95 | 0.95 | 0.98 |
| n_steps | - | - | 2048 | 1024 | 4096 |
| n_epochs | - | - | 10 | 15 | - |
| clip_range | - | - | 0.2 | 0.2 | - |
| vf_coef | - | - | 0.5 | 0.5 | - |
| max_grad_norm | - | - | 0.5 | 0.5 | - |
| normalize_advantage | - | - | - | - | True |
| Parallelized? | No | No | No | No | No |