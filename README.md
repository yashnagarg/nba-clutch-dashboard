<h2> 💻 Landing page:</h2>

<img width="1908" height="1012" alt="image" src="https://github.com/user-attachments/assets/ac303075-c894-48d6-afe8-a9795a9cc06e" />

This project is an NBA analytics dashboard focused on clutch performance
and hot streaks. It is built around Fire Score, a custom weighted metric 
I developed to identify players who consistently heat up when the pressure is highest.

The goal is to explore which players perform best and most consistently
in clutch situations, and how well they sustain hot streaks when the
pressure is on.

Q. What are hot streaks?                               
A. A hot streak refers to a sequence of consecutive made field goals. Players enter a hot state after 3 consecutive made field goals, with the streak resetting after a miss. The dashboard examines the frequency and length of these streaks throughout the game, especially how often players get hot specifically during clutch situations.



A custom weighted Fire Score combines multiple clutch and streak-based
metrics to create an overall ranking of player performance.

**Weighted Model:**

 | Metric   | Weight | 
|--------|-----|
| Clutch FG%  | 30%  | 
| Clutch Hot Rate	 | 25%  |
| Clutch Average Streak  | 25%  |
| Clutch Points    | 20%  | 

Each component is normalized to a 0–1 scale before the weighted score is calculated.
Players must have at least 15 clutch field-goal attempts to appear in the final rankings,
reducing the impact of extremely small samples.

<h2>📶 Why these metrics?</h2>
The model is designed to capture different aspects of clutch performance:

Clutch FG% → Can the player actually convert shots under pressure?<br>Clutch Hot Rate → How frequently are they shooting while in a hot state?<br>
Clutch Average Streak → How long do their hot streaks tend to last?<br>Clutch Points → How much scoring production are they generating?<br>
Together, these features attempt to capture not just how much a player scores, but how they respond when the game gets tight.

<h2>🛠️ Tech Stack</h2>

Frontend:<br>
React,
Vite,
JavaScript / JSX,
CSS,
Framer Motion<br>

Backend:<br>
Python,
FastAPI,
Pandas,
NumPy<br>

Development:<br>
Git / GitHub,
REST API
