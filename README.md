
# 🧠 Maze Solver using Reinforcement Learning (Q-Learning)

A clean and interactive implementation of **Reinforcement Learning** using **Q-Learning** in a custom maze environment built with Pygame.  
This project beautifully demonstrates both the **training process** and the **final maze-solving behavior**.

---

# 📌 Project Files

## 🔹 `maze_qlearn_demo.py`  
This script shows the **final working RL agent** solving the maze.

✔ Draw maze  
✔ Set Start (S) & Goal (G)  
✔ Train the model  
✔ Watch how agent finds the optimal path  

---

## 🔹 `maze_qlearn_final.py`  
This script shows **visual training** — how the model actually learns step-by-step.  
You can **see** exploration, movement, updates, and progression of reinforcement learning.

---

# 🎮 Controls (Both Scripts)

```
Left-click        : Toggle wall (drag supported)
S + Left-click    : Set Start position
G + Left-click    : Set Goal position
T                 : Train Q-learning
R                 : Run learned policy (visual)
C                 : Clear all walls
M                 : Generate random maze
Esc               : Quit the program
```

---

# 🖼️ Screenshots (Replace with your own)

### ✔ Training Process
![Training Demo](media/training_demo.png)

### ✔ Agent Solving Maze
![Solving Demo](media/solving_demo.png)

---

# 🧠 How Reinforcement Learning Works

The maze is treated as a grid of **states**.  
The agent has **4 actions**: Up, Right, Down, Left.

### Reward system:
- **-1** per step → encourages shorter path  
- **-5** for hitting a wall  
- **+100** for reaching the goal  

### Q-Learning Update:
```
Q(s, a) = Q(s, a) + α * [ r + γ * max Q(s', :) - Q(s, a) ]
```

Over time the agent learns the best action for every state.

---

# ▶️ Running the Project

Install required modules:
```
pip install pygame numpy
```

Run scripts:
```
python maze_qlearn_demo.py
python maze_qlearn_final.py
```

---

# ⭐ Features

- Interactive maze builder  
- Visual RL training animation  
- Q-Learning implementation  
- Final policy playback  
- Clear demonstration for learning RL concepts  
- Beginner friendly  

---

# 📜 License
You may freely use this project for learning, teaching, or academic submissions.
