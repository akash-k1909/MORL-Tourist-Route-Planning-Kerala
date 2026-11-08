# 🗺️ Multi-Objective Reinforcement Learning for Adaptive Tourist Route Planning

> **Course:** Reinforcement Learning (7th Semester)  
> **Institution:** Amrita Vishwa Vidyapeetham, School of Computing  
> **Authors:** Kanuri Akash & Korukunda Krishna Sai  
> **Domain:** Kerala Tourism Benchmark — 56 Real-World POIs across 6 Clusters

---

## 📌 Project Overview

This project implements a **Multi-Objective Reinforcement Learning (MORL)** system that generates adaptive, personalized tourist itineraries for Kerala, India. The agent simultaneously optimizes three conflicting objectives:

| Objective | Symbol | Direction |
|-----------|--------|-----------|
| Tourist Satisfaction (ratings + diversity) | R₁ | Maximize ↑ |
| Total Transit Time | R₂ | Minimize ↓ |
| Total Financial Cost (INR) | R₃ | Minimize ↓ |

A **Preference-Conditioned Dueling Double DQN** with **Dynamic Action Masking** ensures:
- ✅ Zero constraint violations (never stranded without time/budget to return)
- ✅ Real-time mid-trip re-planning in **0.84 ms**
- ✅ Personalized Pareto-optimal routes for any tourist persona

---

## 🏗️ Architecture

```
User Input (Budget, Time, Origin, Weights)
           ↓
Custom Gymnasium Environment
  ├── POI Static Data (56 Kerala Nodes: GPS, fees, ratings, hours)
  ├── Haversine Transit Matrix (κ=1.35, v̄=32 km/h)
  ├── State Tracker (one-hot location + resources + visited bitmap + weights)
  └── Dynamic Action Masking (time & budget feasibility + operating hours)
           ↓                               ↑
Dueling MO-DQN Agent              Reward Vector [R₁, R₂, R₃]
  ├── Shared Feature Trunk (256→256)
  ├── Value Stream V(s,w)
  ├── Advantage Stream A(s,a,w)
  ├── Mean Subtraction Identifiability
  └── -∞ Masking on Invalid Actions
           ↓
Generated Personalized Itinerary (Sequence, Time, Cost)
```

---

## 📂 Repository Structure

```
RL PROJECCT/
├── run_kerala_rl.py              # Standalone Python implementation (main file)
├── rl_project.ipynb              # Jupyter / Google Colab notebook version
├── morl_tourist_app.html         # Interactive Web App Demo (open in browser)
├── Launch_Kerala_Tour_Planner.bat# Windows launcher for the web app
├── RL_Term_Project_Presentation.pptx  # 15-slide academic presentation
├── RL_Final_Report_Overleaf_Submission.zip  # IEEE-format LaTeX report (18 pages)
└── README.md                     # This file
```

---

## 🗺️ Kerala Tourism Dataset (56 Nodes)

Real GPS coordinates, dwell times, admission fees, ratings, and operating hours across **6 regional clusters**:

| Cluster | Nodes | Key Attractions |
|---------|-------|-----------------|
| **Trivandrum** (Origin Hub) | 12 | Padmanabhaswamy Temple, Kovalam Beach, Varkala Cliff, Poovar Island |
| **Kochi** | 8 | Fort Kochi Fishing Nets, Mattancherry Palace, Jewish Synagogue, Cherai Beach |
| **Munnar** | 6 | Eravikulam National Park, KDHP Tea Museum, Mattupetty Dam, Top Station |
| **Alleppey** | 4 | Punnamada Lake Houseboats, Marari Beach, Kumarakom Bird Sanctuary |
| **Wayanad** | 4 | Edakkal Caves, Banasura Sagar Dam, Pookode Lake, Soochipara Falls |
| **Thrissur** | 4 | Athirapally Waterfalls, Guruvayur Temple, Vadakkunnathan Temple |

**Distance Engine:** Spherical Haversine formula with terrain detour factor κ = 1.35 and average transit velocity v̄ = 32.0 km/h matching Kerala highway conditions.

---

## ⚙️ MDP Formulation

| Component | Definition |
|-----------|------------|
| **State** `s` | `[loc_onehot(N) ∣ T_rem ∣ B_rem ∣ clock ∣ visited(N) ∣ weights(3)]` |
| **Action** `a` | Next POI index ∈ {0, 1, ..., N−1} (filtered by action mask) |
| **Reward** `r` | Scalarized: `r = w·[R_sat, R_time, R_cost]` |
| **Termination** | Agent returns to origin node (ID 0) or no feasible actions remain |
| **Budget** | Time budget T_max=10h, Financial budget B_max=INR 2000 |

**Reward Components:**
- `R_sat = rating(j) × exp(-0.35 × category_count)` — satiation-discounted satisfaction
- `R_time = -t_used / 2.0` — transit time penalty
- `R_cost = -c_used / 500.0` — normalized cost penalty

---

## 🤖 Algorithm: Preference-Conditioned Dueling Double DQN

### Neural Network
```
Input: State vector [location ∣ resources ∣ history ∣ preference_weights]
  → Linear(state_dim, 256) → LayerNorm → ReLU
  → Linear(256, 256) → ReLU
  → Value Stream:     Linear(256→128→1)       → V(s,w)
  → Advantage Stream: Linear(256→128→N_actions)→ A(s,a,w)
  → Q(s,a,w) = V(s,w) + A(s,a,w) − mean_A  [identifiability]
  → mask illegal actions → −∞                [action masking]
```

### Training
- **Algorithm:** Double Q-Learning with Polyak soft target updates (τ=0.005)
- **Loss:** Smooth L1 (Huber) loss over experience replay mini-batches
- **Optimizer:** AdamW (lr=8e-4, weight_decay=1e-5)
- **Exploration:** Masked ε-greedy (ε: 1.0 → 0.05, decay=0.9985)
- **Buffer:** FIFO Replay Buffer (capacity=50,000 transitions)
- **Batch Size:** 128 | **Discount:** γ=0.98

---

## 📊 Baseline Comparison

| Model | Hypervolume (HV) ↑ | Sparsity ↓ | Pareto Points | Constraint Violations |
|-------|-------------------|------------|---------------|-----------------------|
| Random Walk | 0.0272 | 0.0151 | 16 | 0.0% |
| Nearest-Neighbor Greedy | 0.0059 | 0.0000 | 48 (Degenerate) | 0.0% |
| Single-Objective DQN | 0.0011 | 0.0000 | 48 (Collapsed) | 0.0% |
| Genetic Algorithm (NSGA-II) | 0.0385 | 0.0112 | 22 | 0.0% |
| **Proposed Dueling MO-DQN** | **0.0452** | **0.0095** | **17 (Optimal)** | **0.0%** |

---

## 🚀 Installation & Running

### Prerequisites
```bash
Python 3.8+
```

### Install Dependencies
```bash
pip install torch numpy pandas matplotlib seaborn gymnasium
```

### Option 1: Run Python Script Directly
```bash
python run_kerala_rl.py
```

**Expected Output:**
```
Multi-Objective RL Tourist Trip Planner (Kerala Tourism Benchmark)
Loaded 12 attractions in Trivandrum Cluster...
Starting training on Kerala Tourism Benchmark (600 episodes)...
Episode  200/600 | Avg Return (Last 100): +5.22 | Epsilon: 0.741
Episode  400/600 | Avg Return (Last 100): +7.05 | Epsilon: 0.549
Episode  600/600 | Avg Return (Last 100): +7.30 | Epsilon: 0.406
Training finished successfully!

--- Persona A: Luxury Tourist (Weight: Sat 0.8, Time 0.1, Cost 0.1) ---
Route: Central Station → Napier Museum → Shanghumukham Beach → ...
Satisfaction: 22.4 stars | Time: 6.91 hrs | Cost: INR 793.37

--- Persona B: Budget Backpacker (Weight: Sat 0.2, Time 0.1, Cost 0.7) ---
Route: Central Station → Attukal Temple → Napier Museum → ...
Satisfaction: 17.8 stars | Time: 5.02 hrs | Cost: INR 446.94

Mid-Trip Disruption & Dynamic Online Re-Planning Simulation...
ALERT: [Shanghumukham Beach Promenade] is temporarily CLOSED!
Online Re-Planned Route: [...alternative route without the closed venue...]
Re-planning Latency: 0.84 ms | Constraint Violations: 0.0%
```

### Option 2: Google Colab / Jupyter Notebook
1. Upload `rl_project.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run all cells sequentially (Cell 0 installs all dependencies automatically via `!pip install`)

### Option 3: Interactive Web App Demo
- **Windows:** Double-click `Launch_Kerala_Tour_Planner.bat`
- **Any OS:** Open `morl_tourist_app.html` directly in any modern browser (Chrome/Firefox/Edge)
- No server required — fully self-contained HTML5/JavaScript

---

## 🔧 Customization

### Change Tourist Persona Preferences
In `run_kerala_rl.py`, modify the preference weight vectors:
```python
weights_lux = [0.80, 0.10, 0.10]  # [Satisfaction, Time, Cost] — Luxury Seeker
weights_bud = [0.20, 0.10, 0.70]  # Budget Backpacker
weights_time = [0.10, 0.80, 0.10] # Speed Runner (minimize time)
```
Weights must sum to 1.0.

### Change Active Cluster
```python
# Options: 'trivandrum', 'kochi', 'munnar', 'alleppey', 'wayanad', 'thrissur'
df_kerala = load_kerala_tourism_dataset(region='kochi')
```

### Change Budget Constraints
```python
env = MultiObjectiveKeralaTourEnv(
    df_kerala, dist_mat, time_mat, cost_mat,
    t_max=12.0,     # 12 hours time budget
    b_max=3000.0,   # INR 3000 financial budget
    start_clock=8.0 # 8:00 AM departure
)
```

### Increase Training Episodes (for better convergence)
```python
agent, hist = train_kerala_agent(env, episodes=2000)
```

---

## 📐 Key Mathematical Components

**Haversine Distance:**
$$d_{ij} = 2R \arcsin\left(\sqrt{\sin^2\!\left(\frac{\Delta\phi}{2}\right) + \cos\phi_i\cos\phi_j\sin^2\!\left(\frac{\Delta\lambda}{2}\right)}\right)$$

**Effective Road Distance:** `d_eff = κ × d_ij` where `κ = 1.35` (terrain detour factor)

**Transit Cost:** `C(vᵢ, vⱼ) = F_base + C_rate × d_ij + F_j`

**Dueling Q-value:**
$$Q(s,a,\mathbf{w};\theta) = V(s,\mathbf{w};\theta_V) + A(s,a,\mathbf{w};\theta_A) - \frac{1}{|\mathcal{M}(s)|}\sum_{a'\in\mathcal{M}(s)} A(s,a',\mathbf{w};\theta_A)$$

**Dynamic Action Mask:**
$$m_j = \begin{cases} 1 & \text{if } T_{rem} \geq t_{ij} + t_{j0} \text{ AND } B_{rem} \geq c_{ij} + c_{j0} \text{ AND clock} \leq \text{close}_j \\ 0 & \text{otherwise} \end{cases}$$

---

## 🧪 Mid-Trip Re-Planning

The system handles **3 real-world disruption scenarios**:
1. **Venue Closure** — Masks out the closed POI instantly; agent re-routes in 0.84 ms
2. **Monsoon Traffic Delay** (+1.5 hrs time deduction) — Action mask automatically prunes destinations unreachable within updated time budget
3. **Budget Injection** (+INR 1,000) — Action mask dynamically un-prunes premium venues now within reach

---

## 📋 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Neural network (Dueling DQN) |
| `numpy` | ≥1.24 | Matrix operations, transit matrices |
| `pandas` | ≥2.0 | POI dataset management |
| `matplotlib` | ≥3.7 | Training curves, route visualization |
| `gymnasium` | ≥0.29 | RL environment base class |
| `seaborn` | ≥0.12 | Dashboard plots (optional) |

---

## 📄 Report & Presentation

- **IEEE-format LaTeX Report** (18 pages): `RL_Final_Report_Overleaf_Submission.zip` → Upload to [Overleaf](https://overleaf.com)
- **Presentation Deck** (15 slides, 16:9): `RL_Term_Project_Presentation.pptx`

---

## 📜 References

1. Vansteenwegen et al., "The orienteering problem: A survey," *European Journal of Operational Research*, 2011.
2. Chen et al., "MORL.SGF: A multi-objective RL framework for urban tourism," *IEEE Trans. Intell. Transp. Syst.*, 2024.
3. Nguyen et al., "Three stage optimization for sustainable tourism route planning," *Ann. Oper. Res.*, 2025.
4. Castillo & Gomez, "An agile optimization algorithm for the tourist trip problem," *AIMS Math.*, 2026.
5. Mukherjee et al., "Tourism route recommendation using RL under real-time user feedback," *IEEE PerCom*, 2024.
6. Bello et al., "Neural combinatorial optimization with RL," *arXiv:1611.09940*, 2016.
7. Kitsios & Kamarianakis, "Deep RL planner for city tours," *Expert Syst. Appl.*, 2023.

---

## 👥 Authors

**Kanuri Akash**  
B.Tech Computer Science & Engineering, Amrita Vishwa Vidyapeetham  
Research Interests: Deep Reinforcement Learning, Multi-Objective Optimization, Autonomous Agents  
📧 kanuriakash4@gmail.com

**Korukunda Krishna Sai**  
B.Tech Computer Science & Engineering, Amrita Vishwa Vidyapeetham  
Research Interests: Reinforcement Learning, Combinatorial Optimization, Intelligent Transportation Systems

---

*Capstone Term Project — Reinforcement Learning Course, Amrita Vishwa Vidyapeetham, 2026*
