# Multi-Objective Reinforcement Learning for Adaptive and Personalized Tourist Route Planning
### with Dynamic Action Masking and Mid-Trip Re-planning

> **Course:** Reinforcement Learning — End-Semester Term Project  
> **Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham  
> **Authors:** Kanuri Akash · Korukunda Krishna Sai  
> **Dataset:** Real-World Kerala Tourism Spatial Benchmark (56 Nodes, 6 Regional Clusters)

---

## 🗺️ Project Overview

Traditional tourist route planners optimize a single objective (usually distance or time). Real tourists, however, balance **three competing objectives simultaneously**:

| Objective | Goal |
|-----------|------|
| 🌟 **Satisfaction** | Maximize attraction ratings weighted by category diversity |
| ⏱️ **Time Efficiency** | Minimize total transit and dwell time |
| 💰 **Financial Cost** | Minimize taxi fares + attraction admission fees (INR) |

This project formulates Kerala tourist trip planning as a **Multi-Objective Markov Decision Process (MO-MDP)** and trains a **Preference-Conditioned Dueling Double Deep Q-Network (MO-DQN)** with **Dynamic Action Masking** to generate Pareto-optimal personalized itineraries in real time (< 1 ms re-planning latency).

---

## 📂 Repository Structure

```
RL-MORL-Kerala-Tourism/
│
├── rl_project.ipynb               # Main Jupyter / Google Colab notebook (7 cells)
├── run_kerala_rl.py               # Standalone Python script (no Jupyter needed)
├── morl_tourist_app.html          # Interactive Web App Demo (open in browser)
├── Launch_Kerala_Tour_Planner.bat # One-click browser launcher (Windows)
│
├── RL_Final_Report_Overleaf_Submission.zip  # IEEE-format LaTeX report (18–20 pages)
├── RL_Term_Project_Presentation.pptx        # 15-slide presentation deck (16:9)
│
└── README.md
```

---

## 🧠 Architecture

```
User Input (Tourist Profile: Time Budget, Financial Budget, Origin, Preference Weights)
        │
        ▼
Custom Gymnasium Environment
 ├── Transit Matrix Engine      (Haversine Geodesic Distance + Terrain Detour κ=1.35)
 ├── POI Static Data            (56 Kerala POIs: GPS Coords, Fees, Ratings, Hours)
 ├── State Tracker & Action Mask (Dynamic bitwise feasibility enforcement)
 └── Multi-Objective Reward Block [r_sat, r_time, r_cost]
        │
        ▼
Preference-Conditioned Dueling MO-DQN Agent
 ├── Shared Feature Trunk       (Linear → LayerNorm → ReLU → Linear → ReLU)
 ├── Value Stream V(s, w)       (Linear → ReLU → Linear → scalar)
 ├── Advantage Stream A(s,a,w)  (Linear → ReLU → Linear → N logits)
 ├── Action Masking Layer       (illegal actions → −∞ before argmax)
 └── Experience Replay Buffer   (FIFO, capacity 50,000)
        │
        ▼
Generated Personalized Itinerary (Sequence, Time, Cost, Ratings)
```

---

## 🗺️ Kerala Tourism Dataset (56 Nodes)

The benchmark covers **56 real Points of Interest** across 6 regional clusters:

| Cluster | Nodes | Key Attractions |
|---------|-------|-----------------|
| **Trivandrum** | 12 | Padmanabhaswamy Temple, Kovalam Beach, Varkala Cliff, Poovar Island, Ponmudi |
| **Kochi** | 8 | Fort Kochi Fishing Nets, Mattancherry Palace, Jewish Synagogue, Cherai Beach |
| **Munnar** | 6 | Eravikulam National Park, KDHP Tea Museum, Mattupetty Dam, Top Station |
| **Alleppey** | 4 | Punnamada Lake Houseboats, Marari Beach, Kumarakom Bird Sanctuary |
| **Wayanad** | 4 | Edakkal Caves, Banasura Sagar Dam, Pookode Lake, Soochipara Falls |
| **Thrissur** | 4 | Athirapally Waterfalls, Guruvayur Temple, Vadakkunnathan Temple, Kalamandalam |

### Distance & Cost Model
- **Geodesic Distance:** Spherical Haversine formula
- **Terrain Detour Factor:** κ = 1.35 (Kerala coastal/highland highways)
- **Average Transit Velocity:** v̄ = 32.0 km/h
- **Taxi Fare:** INR 30.00 base + INR 15.00/km
- **Admission Fees:** INR 0 – 600 per attraction

---

## 🤖 MDP Formulation

### State Space
```
s_t = [l_t ∈ {0,1}^N,  T^rem,  B^rem,  clock,  v_t ∈ {0,1}^N,  w ∈ Δ²]
```
- `l_t` — One-hot location indicator (current POI)
- `T^rem` — Normalized remaining time budget
- `B^rem` — Normalized remaining financial budget
- `clock` — Normalized wall-clock time (09:00 AM start)
- `v_t` — Binary visited bitmask (prevents revisits)
- `w` — Active tourist preference weight vector [w_sat, w_time, w_cost]

### Action Space
```
A = {0, 1, 2, ..., N−1}   (next POI to visit; 0 = return to hotel)
```

### Dynamic Action Masking
At each step, an action `j` is **allowed only if**:
1. POI `j` has not been visited
2. Traveler arrives before closing time
3. Sufficient time remains for leg `i→j` **plus** guaranteed return `j→0`
4. Sufficient budget remains for leg `i→j` **plus** guaranteed return `j→0`

### Multi-Objective Reward Vector
```
r(s, a) = [r_sat,  r_time,  r_cost]
         = [rating_j × exp(−0.35 × cat_count_j),  −t_ij/2,  −c_ij/500]
```

Scalarized for training:
```
R_scalar = w · r(s, a)   where w ∈ Δ² (unit 2-simplex)
```

---

## 🏆 Baseline Comparison

| Model | Hypervolume (HV) ↑ | Re-planning Latency | Constraint Violations |
|-------|-------------------|--------------------|-----------------------|
| Random Walk Baseline | 0.0272 | — | 0.0% |
| Nearest-Neighbor Greedy | 0.0315 | — | 0.0% |
| Single-Objective DQN (SO-DQN) | 0.0334 | ~1 ms | 0.0% |
| Genetic Algorithm (NSGA-II) | 0.0405 | 4,000–12,000 ms | 0.0% |
| **Proposed Dueling MO-DQN** | **0.0452** | **0.84 ms** | **0.0%** |

---

## ⚙️ Dependencies

```
Python 3.9+
torch >= 2.0.0
gymnasium >= 0.26.0   (or gym >= 0.21.0 as fallback)
numpy >= 1.24.0
pandas >= 1.5.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

Install all dependencies in one command:

```bash
pip install torch gymnasium numpy pandas matplotlib seaborn
```

For Google Colab (notebook automatically installs on first cell run):
```python
!pip install -q gymnasium torch pandas numpy matplotlib seaborn
```

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Upload `rl_project.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Click **Runtime → Run all**
3. All 7 cells execute sequentially:
   - **Cell 0:** Install dependencies + set seeds
   - **Cell 1:** Load 56-Node Kerala Dataset + build Haversine matrices
   - **Cell 2:** Initialize Multi-Objective Gymnasium Environment
   - **Cell 3:** Define Dueling MO-DQN + Replay Buffer
   - **Cell 4:** Train agent (2000 episodes, ~5–10 min on CPU)
   - **Cell 5:** 4-Panel evaluation dashboard (saves `kerala_morl_dashboard.png`)
   - **Cell 6:** Mid-trip disruption & real-time re-planning demo

### Option 2 — Standalone Python Script

```bash
# Clone the repository
git clone https://github.com/kanuriakash/MORL-Kerala-Tourism.git
cd MORL-Kerala-Tourism

# Install dependencies
pip install torch gymnasium numpy pandas matplotlib seaborn

# Run the complete pipeline
python run_kerala_rl.py
```

**Expected output:**
```
======================================================================
Multi-Objective RL Tourist Trip Planner (Kerala Tourism Benchmark)
======================================================================
Loaded 12 attractions in Trivandrum Cluster:
...
Starting training on Kerala Tourism Benchmark (600 episodes)...
Episode  200/600 | Avg Return (Last 100): +5.22 | Epsilon: 0.741
Episode  400/600 | Avg Return (Last 100): +7.05 | Epsilon: 0.549
Episode  600/600 | Avg Return (Last 100): +7.30 | Epsilon: 0.406
Training finished successfully!

--- Persona A: Luxury Tourist (Weight: Sat 0.8, Time 0.1, Cost 0.1) ---
Route: Central Station Hotel → Napier Museum → Shanghumukham Beach → ...
Satisfaction: 22.4 stars | Time: 6.91 hrs | Cost: INR 793.37

--- Persona B: Budget Backpacker (Weight: Sat 0.2, Time 0.1, Cost 0.7) ---
Route: Central Station Hotel → Attukal Temple → Napier Museum → ...
Satisfaction: 17.8 stars | Time: 5.02 hrs | Cost: INR 446.94

Mid-Trip Disruption & Dynamic Online Re-Planning Simulation
======================================================================
ALERT: Next venue [Shanghumukham Beach Promenade] is temporarily CLOSED!
Online Re-Planned Route: [...bypassed venue, new optimal sequence...]
Re-planning Latency: 0.84 ms | Constraint Violations: 0.0%
```

### Option 3 — Interactive Web App

Open `morl_tourist_app.html` directly in any browser — no server needed. Features:
- Select departure region (Trivandrum, Kochi, Munnar, Alleppey, Wayanad, Thrissur)
- Adjust time budget (hours) and financial budget (INR)
- Drag preference sliders (Satisfaction / Time / Cost)
- Trigger mid-trip disruption and watch real-time re-routing

Or double-click `Launch_Kerala_Tour_Planner.bat` on Windows to auto-open.

### Switching Regions

To run on a different Kerala cluster, change the `region` parameter:

```python
# Options: 'trivandrum', 'kochi', 'munnar', 'alleppey', 'wayanad', 'thrissur'
# Or pass None to load all 56 POIs (full statewide benchmark)
df_kerala = load_kerala_tourism_dataset(region='kochi')
```

---

## 📊 Key Results

### Personalized Routing (Simplex Weight Conditioning)
The agent dynamically adapts its route based on the user's preference vector — **no retraining required**:

| Tourist Persona | w = [sat, time, cost] | Route Characteristics |
|----------------|----------------------|----------------------|
| Luxury Seeker | [0.80, 0.10, 0.10] | High-rated temples + cultural sites, INR ~800 |
| Budget Backpacker | [0.20, 0.10, 0.70] | Free-entry beaches + gardens, INR ~450 |
| Time-Efficient | [0.30, 0.60, 0.10] | Nearby compact cluster, 4–5 hrs total |

### Mid-Trip Resilience
Three real-world disruption scenarios evaluated:
1. **Venue Closure** → Reroutes in 0.84 ms, substitutes equivalent landmark
2. **Monsoon Traffic Delay (+1.5 hrs)** → Prunes distant POIs, return guarantee preserved
3. **Budget Injection (+INR 1,000)** → Dynamically unmasks premium experiences

---

## 🔬 Technical Details

### Neural Network Architecture
```
Input: State vector s ∈ ℝ^(2N+6)   where N = number of POIs

Shared Trunk:
  Linear(2N+6 → 256) → LayerNorm(256) → ReLU
  Linear(256 → 256) → ReLU

Value Stream:   Linear(256→128) → ReLU → Linear(128→1)         → V(s,w)
Advantage Stream: Linear(256→128) → ReLU → Linear(128→N)       → A(s,a,w)

Q(s,a,w) = V(s,w) + [A(s,a,w) − (1/|M(s)|) Σ_{a'∈M(s)} A(s,a',w)]
Q_masked(s,a) = Q(s,a,w)  if m[a]=1
              = −∞          if m[a]=0
```

### Training Hyperparameters
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (lr=8×10⁻⁴, weight_decay=1×10⁻⁵) |
| Discount Factor γ | 0.98 |
| Polyak Update Rate τ | 0.005 |
| Replay Buffer Capacity | 50,000 |
| Batch Size | 128 |
| Epsilon Start → End | 1.0 → 0.05 (decay 0.9985/episode) |
| Training Episodes | 2,000 (Colab) / 600 (quick demo) |

---

## 📄 Report & Presentation

The full **18–20 page IEEE-format research report** (`RL_Final_Report_Overleaf_Submission.zip`) includes:
- Complete MDP formulation with mathematical proofs
- Contraction mapping guarantees under action masking (Banach Fixed-Point Theorem)
- Hypervolume evaluation across the full Pareto simplex
- Cross-platform inference latency benchmarks
- Mid-trip re-planning resilience experiments

The **15-slide presentation deck** (`RL_Term_Project_Presentation.pptx`) covers all 14 required course rubric items + a live demo slide.

---

## 📚 References

1. Vansteenwegen et al. (2011). *The orienteering problem: A survey.* European Journal of Operational Research.
2. Chen et al. (2024). *Multi-objective RL for personalized trip planning.* AAAI.
3. Zhang et al. (2023). *City tour optimization with deep RL and time windows.* MDPI Applied Sciences.
4. Kitsios et al. (2023). *Deep RL for heritage tourism itinerary design.* Expert Systems with Applications.
5. Nguyen et al. (2025). *Sustainable tourism optimization with mandatory cultural POIs.* Annals of Operations Research.
6. Bello et al. (2016). *Neural combinatorial optimization with RL.* ICLR.
7. Castillo et al. (2026). *Agile optimization for tourist trip problem with categorical constraints.* AIMS Mathematics.

---

## 📝 License

This project is submitted as an academic term project at Amrita Vishwa Vidyapeetham.  
Code is released for educational and research purposes.

---

*Built with PyTorch · Gymnasium · NumPy · Pandas · Matplotlib*
