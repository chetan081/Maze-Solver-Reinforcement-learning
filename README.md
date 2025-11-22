Maze Solver using Reinforcement Learning (Q-Learning)

This project demonstrates Reinforcement Learning through an interactive maze environment built with Python + Pygame.

You can draw your own maze, set Start & Goal, train an RL agent, and watch:

✅ How the agent learns (training visualization)
✅ How the agent solves the maze (final Q-policy execution)

Project Files
🔹 maze_qlearn_demo.py — Maze Solving Demo

This script contains the main game where:

You draw a maze

Set Start (S) and Goal (G)
Press T to train the agent
Press R to watch the agent solve the maze
This is the main solver.
Perfect for presentations where the agent shows its final learned behavior.


maze_qlearn_final.py — Training Visualization
This script visually shows how Q-learning updates during training:
Agent explores the maze
Q-values update step-by-step
You can see learning in real-time

✨ Best for explaining RL concepts like exploration, Q-values, and policy improvement.

Controls (Both Scripts)
Left-click        : Toggle wall (drag supported)
S + Left-click    : Set Start cell
G + Left-click    : Set Goal cell
T                 : Train Q-learning
R                 : Run the learned policy
C                 : Clear walls
M                 : Random maze generator
Esc               : Quit


How Reinforcement Learning Works

Each grid cell = State
Agent actions = Up, Down, Left, Right

 Rewards
Event	         Reward
Normal step	    -1
Hit wall	      -5
Reach goal	   +100


Q-Learning Update Formula
Q(s, a) = Q(s, a) + α [ r + γ * max(Q(s', ·)) – Q(s, a) ]
α (alpha) = learning rate
γ (gamma) = discount factor
ε (epsilon-greedy) = controls exploration


How to Run

Install dependencies:

pip install pygame numpy


Run the maze solver:

python maze_qlearn_demo.py


Run the training visualizer:

python maze_qlearn_final.py



