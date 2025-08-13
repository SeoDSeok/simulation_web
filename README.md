# simulation_web
Simulation web page in hospital process

## Installation

1. Create and activate a Python virtual environment:
```bash
conda create -n simulation_web python=3.11
conda activate simulation_web
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
### Running

To run the simulation web:

```
python run.py
```

## Overall Framework
<img width="645" height="581" alt="Image" src="https://github.com/user-attachments/assets/0d2c420e-aa70-4852-a5a4-c698a146bebc" />

### 1. Data Upload
<img width="726" height="866" alt="Image" src="https://github.com/user-attachments/assets/4aab21b2-5133-4b18-8126-b568fe8b32ed" />

- Event Log (CSV) :
Log data for simulation
- Raw Parameter Data (CSV) [optional] :
Data for extracting parameters for simulation
<br>
When you upload Raw Parameter Data, the following four files are automatically generated.
If you don't upload it, please upload the 4 files below directly.

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
- Doctor Shift
    - Number of Intern in shift_1
    - Number of Intern in shift_2
    - Number of Resident in shift_1
    - Number of Resident in shift_2
    - Number of Specialist in shift_1
    - Number of Specialist in shift_2
- Nurse Shift
    - Number of Junior Nurses in shift_1
    - Number of Junior Nurses in shift_2
    - Number of Junior Nurses in shift_3
    - Number of Senior Nurses in shift_1
    - Number of Senior Nurses in shift_2
    - Number of Senior Nurses in shift_3
- Waiting Time
    - Median Waiting Time (minutes)
    - Mean Waiting Time (minutes)
- Duration Time
    - Case Duration (max)
    - Case Duration (median)
    - Case Duration (mean)

### 3. Scenario Map Screen
Changeable scenario screen extracted from taxonomy
<img width="1907" height="860" alt="Image" src="https://github.com/user-attachments/assets/f35bae9f-8e59-4756-901c-f1ad0f939a24" />
- Run Simulation with All Scenarios :
Use accumulated scenarios to conduct simulations
- Clear Scenario Stack : 
Remove all accumulated scenarios

### 4. Click the Scenario button
Click the scenario button you want to select

### 5. Option screen corresponding to the scenario
The Select changeable parameters screen for the selected scenario appears

### 6. Simulation Results Comparison Table Screen
