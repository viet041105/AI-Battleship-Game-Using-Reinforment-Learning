# AI-Battleship-Game-Using-Reinforment-Learning

Battleship AI: A Deep Reinforcement Learning Approach
This repository contains an advanced AI agent designed to master the classic game of Battleship. The agent is built upon a Deep Q-Network (DQN), a cornerstone of modern Deep Reinforcement Learning. It learns optimal strategies entirely through self-play, leveraging a Convolutional Neural Network (CNN) to interpret the game board spatially.

The project demonstrates key competencies in Reinforcement Learning, Neural Network design, and strategic AI development.

🎥 Demonstration

https://github.com/user-attachments/assets/863b6fb4-22ed-44ae-be8c-434faae9d4dc


Technical Approach
This project is not a simple rules-based bot. It's a learning agent that develops its own understanding of the game. The following concepts were central to its design.

Core Algorithm: Deep Q-Network (DQN)
The foundation of the AI is the Deep Q-Network (DQN). Instead of using a simple table to store the value of each action in each state (a Q-table), which is infeasible for complex games, we use a neural network to approximate this Q-function.

Function: The network takes the current game state as input and outputs a vector of Q-values, one for each possible move (each cell on the board).

Decision Making: The AI selects the action with the highest Q-value, effectively choosing the move it predicts will lead to the best long-term reward.

Spatial Analysis with CNNs
The game of Battleship is inherently spatial. The position of ships and the patterns of hits and misses are crucial. To capture these spatial relationships, a Convolutional Neural Network (CNN) was chosen over a simple dense network.

Architecture: The network uses Conv2D layers that act as learnable feature detectors. They can identify patterns like lines of misses, clusters of hits, or areas of the board that are empty, processing the board as if it were an image.

Advantage: This allows the AI to develop a more intuitive, human-like understanding of the board's topology, leading to more intelligent move predictions.

Advanced State Representation
The input to the neural network is a carefully engineered multi-channel tensor, not just a simple grid. Each channel provides a different layer of information:

Main Board Channel: Represents the player's view of the opponent's grid (-1 for a miss, 1 for a hit, 0 for an unknown cell).

Ship Status Channels: For each ship in the game, a separate channel is dedicated to indicating its status. The entire channel is filled with 1 if the ship is still afloat and 0 if it has been sunk.

This rich, multi-channel representation gives the network all the context it needs to make highly informed decisions.

Hybrid AI Strategy: Search & Hunt
A key innovation in this project is a hybrid strategy that combines the learned intelligence of the DQN with a deterministic, rule-based heuristic for efficiency.

Search Mode (DQN-driven): By default, the AI is in "Search Mode." It uses the full power of the DQN to predict the most probable location of an enemy ship on the board.

Hunt Mode (Heuristic-driven): Once the AI scores a hit, it transitions to "Hunt Mode." In this mode, it temporarily ignores the DQN's predictions and focuses exclusively on attacking the cells adjacent (horizontally and vertically) to the last hit. This dramatically increases the efficiency of sinking a discovered ship.

Return to Search: As soon as a ship is fully sunk, the AI reverts to Search Mode to find the next target.

This hybrid model mirrors expert human play: a broad search followed by a focused attack.

Custom Reward Engineering
The reward function is critical in shaping the AI's behavior. A simple +1 for a hit is insufficient as it doesn't encourage efficiency. A more sophisticated reward calculation was designed:

reward = (is_hit - (remaining_ship_parts / remaining_cells)) * (gamma ** move_index)

Baseline Subtraction: The term (remaining_ship_parts / remaining_cells) represents the baseline probability of hitting a ship part with a random guess. By subtracting this, we reward the AI only when its performance is better than random chance. This incentivizes the AI to make intelligent, high-probability shots rather than just any hit.

Discount Factor (gamma): Future rewards are discounted, encouraging the AI to finish the game as quickly as possible.


System Architecture
Technology Stack
+)Core Language: Python

+)Reinforcement Learning & NN: TensorFlow (Keras)

+)Numerical Operations: NumPy

+)Game Interface (GUI): Pygame

+)Data Visualization: Matplotlib (for plotting training progress)
Usage
Training the Model
To train the AI agent from scratch using the self-play mechanism:

Bash

python train.py
This will run thousands of simulated games. The trained model will be saved periodically to the ./models/ directory as mymodel.keras.

Playing Against the AI
Once a model is trained, you can play against it using the Pygame interface.

Important: Make sure to update the model path in HumanPlay.py to point to your trained .keras file.

Python

# In HumanPlay.py, locate this line and modify the path if necessary:
game = BattleshipGUI('models/mymodel.keras')
Then, run the game:

Bash

python HumanPlay.py



