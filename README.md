# simulation_web
Simulation web page in hospital process

## Installation

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
### Running

To run the evaluation:

```
python run.py
```

## Overall Framework
<img width="645" height="581" alt="Image" src="https://github.com/user-attachments/assets/0d2c420e-aa70-4852-a5a4-c698a146bebc" />

### 1. Data Upload
- Event Log (CSV) :
Log data for simulation
- Activity Frequency (CSV) :
Each activity frequency data obtained from historical data (pdf by activity)
- Event Duration (Pickle) :
Timeline for each activity from historical data
- Activity Resource (Pickle) : 
Percentage of resource involvement per activity from historical data
- Transition Time (CSV) : 
Transition time when going from a specific activity to the next activity from historical data

### 2. Default Simulation Results Screen
Results for running simulations without any options
- Number of Doctors
- Number of Nurses
- Number of Doctors in shift_1
- Number of Doctors in shift_2
- Number of Nurses in shift_1
- Number of Nurses in shift_2
- Number of Nurses in shift_3
- Median Waiting Time (minutes)
- Mean Waiting Time (minutes)
- Case Duration (max)
- Case Duration (median)
- Case Duration (mean)

### 3. Scenario Map Screen
Changeable scenario screen extracted from taxonomy
<img width="1911" height="922" alt="Image" src="https://github.com/user-attachments/assets/60952ccf-c0a7-40d3-b005-4b484642c0d1" />
- Run Simulation with All Scenarios :
Use accumulated scenarios to conduct simulations
- Clear Scenario Stack : 
Remove all accumulated scenarios

### 4. Click the Scenario button
### 5. Option screen corresponding to the scenario
### 6. Simulation Results Comparison Table Screen