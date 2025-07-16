from flask import Blueprint, render_template, request, redirect, url_for, session
import json
import os
from app.simulator import formatter
from app.simulator.scenario import simulate_from_loader
import pandas as pd
import pickle
from copy import deepcopy

scenario = Blueprint('scenario', __name__)

def parse_shift_times(form, default_shift_times):
    def to_hour(time_str):
        try:
            return int(time_str)
        except (ValueError, TypeError):
            return None

    parsed = {}
    for role in ['Doctor', 'Nurse']:
        parsed[role] = {}
        for shift_name, default in default_shift_times[role].items():
            start_key = f'{role.lower()}_{shift_name}_start'
            end_key = f'{role.lower()}_{shift_name}_end'

            start_raw = form.get(start_key)
            end_raw = form.get(end_key)
            
            start_hour = to_hour(start_raw)
            end_hour = to_hour(end_raw)

            if start_hour is None or end_hour is None:
                parsed[role][shift_name] = default
            else:
                parsed[role][shift_name] = (start_hour, end_hour)
    return parsed
def apply_multiplier(base_cases, base_arrivals, multiplier):
    """
    base_cases: 기존 시뮬레이션에서 얻은 case 리스트
    base_arrivals: 기존 시뮬레이션에서 얻은 arrival_times 리스트
    multiplier: 비율 (예: 150 → 150%)

    return: (custom_cases, custom_arrivals)
    """
    if multiplier == 100:
        # 100%면 그대로 반환
        return base_cases, base_arrivals

    # 몇 배로 늘릴 것인지
    times_to_duplicate = multiplier / 100.0
    new_count = int(len(base_cases) * times_to_duplicate)

    # 단순하게 기존 리스트를 반복해서 필요한 개수만큼 자름
    custom_cases = (base_cases * ((new_count // len(base_cases)) + 1))[:new_count]
    custom_arrivals = (base_arrivals * ((new_count // len(base_arrivals)) + 1))[:new_count]

    # arrival_times가 단순 반복이면 같은 시간이 중복될 수 있으니 약간씩 offset을 줄 수도 있음
    # 예시: 반복되는 배수에 따라 도착 시간에 작은 증가량 추가
    offset_arrivals = []
    batch_size = len(base_arrivals)
    for i in range(new_count):
        base_time = custom_arrivals[i % batch_size]
        batch_index = i // batch_size
        offset_time = base_time + batch_index * 0.1  # 0.1초씩 offset
        offset_arrivals.append(offset_time)

    return custom_cases, offset_arrivals
# 시나리오 JSON 파일 로드
def load_scenarios():
    with open(os.path.join('app', 'data', 'scenario_map.json'), 'r', encoding='utf-8') as f:
        return json.load(f)['scenarios']

@scenario.route('/scenario_map')
def scenario_map():
    scenarios = load_scenarios()
    return render_template('scenario_map.html', scenarios=scenarios)

@scenario.route('/simulate/run', methods=['POST'])
def run_scenario():
    scenario_id = request.form.get('scenario_id')

    # 선택한 시나리오 저장 또는 시뮬레이션 적용 로직 연결 가능
    # print(f"[INFO] Scenario selected: {scenario_id}")
    if scenario_id == 'remove_activity':
        return redirect(url_for('scenario.remove_activity_selection'))
    elif scenario_id == 'resequence_activity':
        return redirect(url_for('scenario.vary_sequence2'))
    elif scenario_id == 'reallocating':
        return redirect(url_for('scenario.reallocate_shift_selection'))
    elif scenario_id == 'adding_staff':
        return redirect(url_for('scenario.adding_staff_selection'))
    elif scenario_id == 'rostering':
        return redirect(url_for('scenario.rostering_form'))
    elif scenario_id == 'vary_the_assignment':
        return redirect(url_for('scenario.reassign_staff'))
    elif scenario_id == 'vary_task_comp':
        return redirect(url_for('scenario.reassign_task_composition'))
    elif scenario_id == 'merg_role':
        return redirect(url_for('scenario.merge_shift_roles'))
    elif scenario_id == 'add_new_activity':
        return redirect(url_for('scenario.add_activity'))
    elif scenario_id == 'reduce_time':
        return redirect(url_for('scenario.reduce_lab_boarding'))
    elif scenario_id == 'vary_patient_demand':
        return redirect(url_for('scenario.vary_patient_demand'))
    elif scenario_id == 'vary_patient_sequences':
        return redirect(url_for('scenario.vary_sequence'))
    elif scenario_id == 'vary_queue_discipline':
        return redirect(url_for('scenario.vary_queue'))

    # 구현이 안 된 시나리오에 대해서 단순히 결과 페이지로 리디렉션
    return redirect(url_for('main.index'))

@scenario.route('/simulate/remove_activity_selection', methods=['GET'])
def remove_activity_selection():
    # 시뮬레이션에서 등장하는 액티비티 목록 (현재는 고정, 데이터에 따라 추출하도록도 가능할 듯)
    activities = [
        "접수", "협진연결", "협진진료","초진", "분류", "처방", "퇴실결정",
        "퇴실(입원)", "퇴실(사망)", "퇴실(전원)", "퇴실(귀가)", "입원결정", "입원수속", "수술결정",
        "중환자실입실협진의뢰시간", "중환자실입실협진회신시간"
    ]
    return render_template('remove_activity_selection.html', activities=activities)

@scenario.route('/simulate/remove_activity_compare', methods=['POST'])
def remove_activity_compare():
    activity = request.form.get('activity')

    def safe_diff(x, i1, i2):
        # try:
        return x.iloc[i1] - x.iloc[i2]
        # except (IndexError, KeyError):
        #     return pd.Timedelta(0)

    # 원본 시뮬레이션
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    median_dtp_res_base = base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median()
    mean_dtp_res_base = base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean()
    base_kpis = formatter.format_result("Base", base_df, median_dtp_res_base, mean_dtp_res_base, base_sim.resource_capacities)

    # activity 제거된 시뮬레이션
    mod_sim = simulate_from_loader(remove_activity=activity)
    mod_df = pd.DataFrame(mod_sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    median_dtp_res_mod = mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median()
    mean_dtp_res_mod = mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean()
    mod_kpis = formatter.format_result("Modified", mod_df, median_dtp_res_mod, mean_dtp_res_mod, mod_sim.resource_capacities)

    fixed_role_order = [
        "→ Intern usage",
        "→ Resident usage",
        "→ Specialist usage",
        "→ Junior Nurse usage",
        "→ Senior Nurse usage",
    ]
    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = sorted(set(base_dict.keys()) | set(mod_dict.keys()))
    remaining_labels = sorted(label for label in all_labels_set if label not in fixed_role_order)
    final_order = remaining_labels + fixed_role_order
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in final_order]

    return render_template('comparison2.html', activity=f'remove {activity}', comparison_kpis=comparison_kpis, zip=zip)

@scenario.route('/simulate/reallocate_shift_selection', methods=['GET'])
def reallocate_shift_selection():
    # 기본 인원 수
    doctor_shifts = {'shift_1': 34, 'shift_2': 30}  # total = 64
    nurse_shifts = {'shift_1': 3, 'shift_2': 7, 'shift_3': 3}  # total = 13

    # 기본 인원 수 기준 10% 만큼의 옵션을 제공 (몇 % 로 할지 여기서 수정 가능)
    def generate_options(base):
        delta = max(1, int(base * 0.1))
        return list(range(base - delta, base + delta + 1))
    

    doctor_shift_options = {s: generate_options(n) for s, n in doctor_shifts.items()}
    nurse_shift_options = {s: generate_options(n) for s, n in nurse_shifts.items()}

    return render_template('reallocate_shift_selection.html',
                           doctor_shift_options=doctor_shift_options,
                           nurse_shift_options=nurse_shift_options)

@scenario.route('/scenario/reallocate_staff_compare', methods=['POST'])
def reallocate_staff_compare():
    # Get doctor config
    doctor_shift_1 = int(request.form.get('doctor_shift_0', 0))
    doctor_shift_2 = int(request.form.get('doctor_shift_1', 0))

    # Get nurse config
    nurse_shift_1 = int(request.form.get('nurse_shift_0', 0))
    nurse_shift_2 = int(request.form.get('nurse_shift_1', 0))
    nurse_shift_3 = int(request.form.get('nurse_shift_2', 0))

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    # reallocating
    sim = simulate_from_loader(doc_shift_1=doctor_shift_1, doc_shift_2=doctor_shift_2, nur_shift_1=nurse_shift_1, nur_shift_2=nurse_shift_2, nur_shift_3=nurse_shift_3)
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    mod_kpis = formatter.format_result("Modified", mod_df, 
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )
    # base simulation
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result("Base", base_df, 
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )
    fixed_role_order = [
        "→ Intern usage",
        "→ Resident usage",
        "→ Specialist usage",
        "→ Junior Nurse usage",
        "→ Senior Nurse usage",
    ]
    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = sorted(set(base_dict.keys()) | set(mod_dict.keys()))
    remaining_labels = sorted(label for label in all_labels_set if label not in fixed_role_order)
    final_order = remaining_labels + fixed_role_order
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in final_order]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity=f"Reallocation: [ doctor_shift : {doctor_shift_1}, {doctor_shift_2}], [ nurse_shift : {nurse_shift_1}, {nurse_shift_2}, {nurse_shift_3}]", zip=zip)

@scenario.route('/simulate/adding_staff_selection', methods=['GET'])
def adding_staff_selection():
    # 기본 인원 수
    doctor_shifts = {'shift_1': 34, 'shift_2': 30}  # total = 64
    nurse_shifts = {'shift_1': 3, 'shift_2': 7, 'shift_3': 3}  # total = 13

    # 기본 인원 수 기준 10% 만큼의 옵션을 제공 (몇 % 로 할지 여기서 수정 가능)
    def generate_options(base):
        delta = max(1, int(base * 0.2))
        return list(range(base, base + delta + 1))
    

    doctor_shift_options = {s: generate_options(n) for s, n in doctor_shifts.items()}
    nurse_shift_options = {s: generate_options(n) for s, n in nurse_shifts.items()}

    return render_template('adding_staff_selection.html',
                           doctor_shift_options=doctor_shift_options,
                           nurse_shift_options=nurse_shift_options)

@scenario.route('/scenario/adding_staff_compare', methods=['POST'])
def adding_staff_compare():
    # Get doctor config
    doctor_shift_1 = int(request.form.get('doctor_shift_0', 0))
    doctor_shift_2 = int(request.form.get('doctor_shift_1', 0))

    # Get nurse config
    nurse_shift_1 = int(request.form.get('nurse_shift_0', 0))
    nurse_shift_2 = int(request.form.get('nurse_shift_1', 0))
    nurse_shift_3 = int(request.form.get('nurse_shift_2', 0))

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    # reallocating
    sim = simulate_from_loader(doc_shift_1=doctor_shift_1, doc_shift_2=doctor_shift_2, nur_shift_1=nurse_shift_1, nur_shift_2=nurse_shift_2, nur_shift_3=nurse_shift_3)
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    mod_kpis = formatter.format_result("Modified", mod_df, 
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )
    # base simulation
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result("Base", base_df, 
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    fixed_role_order = [
        "→ Intern usage",
        "→ Resident usage",
        "→ Specialist usage",
        "→ Junior Nurse usage",
        "→ Senior Nurse usage",
    ]
    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = sorted(set(base_dict.keys()) | set(mod_dict.keys()))
    remaining_labels = sorted(label for label in all_labels_set if label not in fixed_role_order)
    final_order = remaining_labels + fixed_role_order
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in final_order]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity=f"Adding Staff : [ doctor_shift : {doctor_shift_1}, {doctor_shift_2}], [ nurse_shift : {nurse_shift_1}, {nurse_shift_2}, {nurse_shift_3}]", zip=zip)

@scenario.route('/simulate/rostering_form', methods=['GET'])
def rostering_form():
    roles = ['Intern', 'Resident', 'Specialist', 'Junior', 'Senior']
    shifts = ['shift_1', 'shift_2', 'shift_3']
    return render_template('rostering_form.html', roles=roles, shifts=shifts)

@scenario.route('/scenario/submit_rostering', methods=['POST'])
def submit_rostering():
    # 역할과 shift 정의
    doctor_roles = ['Intern', 'Resident', 'Specialist']
    nurse_roles = ['Junior', 'Senior']
    doctor_shifts = ['shift_1', 'shift_2']
    nurse_shifts = ['shift_1', 'shift_2', 'shift_3']

    # 사용자 입력값으로부터 리소스 구성 가져오기
    resource_config = {}
    for role in doctor_roles:
        for shift in doctor_shifts:
            field = f"{role}_{shift}"
            resource_config[field] = int(request.form.get(field, 0))
    for role in nurse_roles:
        for shift in nurse_shifts:
            field = f"{role}_{shift}"
            resource_config[field] = int(request.form.get(field, 0))
    # print(resource_config)
    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    # 수정된 구성 기반 시뮬레이션 실행
    sim = simulate_from_loader(**resource_config)
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    mod_kpis = formatter.format_result("Modified", mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )

    # 기본 시뮬레이션 실행
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result("Base", base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    fixed_role_order = [
        "→ Intern usage",
        "→ Resident usage",
        "→ Specialist usage",
        "→ Junior Nurse usage",
        "→ Senior Nurse usage",
    ]
    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = sorted(set(base_dict.keys()) | set(mod_dict.keys()))
    remaining_labels = sorted(label for label in all_labels_set if label not in fixed_role_order)
    final_order = remaining_labels + fixed_role_order
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in final_order]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity=f"Rostering scenario with resource allocation",
                           zip=zip)

@scenario.route('/simulate/reassign_staff', methods=['GET'])
def reassign_staff():
    activities = [
        "접수", "협진연결", "협진진료", "초진", "분류", "처방", "퇴실결정",
        "퇴실(입원)", "퇴실(사망)", "퇴실(전원)", "퇴실(귀가)", "입원결정", "입원수속", "수술결정",
        "중환자실입실협진의뢰시간", "중환자실입실협진회신시간", "visit_start", "visit_end"
    ]
    role_map = {
        "접수": ["Intern", "Resident", "Specialist"],
        "협진연결": ["Intern", "Resident", "Specialist"],
        "협진진료": ["Intern", "Resident", "Specialist"],
        "초진": ["Intern", "Resident", "Specialist"],
        "분류": ["Intern", "Resident", "Specialist", "Junior", "Senior"],
        "처방": ["Intern", "Resident", "Specialist"],
        "퇴실결정": ["Intern", "Resident", "Specialist"],
        "퇴실(입원)": ["Intern", "Resident", "Specialist"],
        "퇴실(사망)": ["Intern", "Resident", "Specialist"],
        "퇴실(전원)": ["Intern", "Resident", "Specialist"],
        "퇴실(귀가)": ["Intern", "Resident", "Specialist"],
        "입원결정": ["Intern", "Resident", "Specialist"],
        "입원수속": ["Intern", "Resident", "Specialist"],
        "수술결정": ["Intern", "Resident", "Specialist"],
        "중환자실입실협진의뢰시간": ["Intern", "Resident", "Specialist"],
        "중환자실입실협진회신시간": ["Intern", "Resident", "Specialist"],
        "visit_start": ["Intern", "Resident", "Specialist"],
        "visit_end" : ["Intern", "Resident", "Specialist", "Junior", "Senior"]
    }
    doctor_roles = ['Intern', 'Resident', 'Specialist']
    nurse_roles = ['Junior', 'Senior']
    return render_template('reassign_staff.html', 
                           activities=activities, 
                           role_map=role_map, 
                           doctor_roles=doctor_roles,
                           nurse_roles=nurse_roles)


@scenario.route('/scenario/submit_reassign_staff', methods=['POST'])
def submit_reassign_staff():
    assignment_map = {}
    for key, value in request.form.items():
        if value:
            assignment_map[key] = value  # 예: {'triage': 'Junior', 'examination': 'Resident'}

    sim = simulate_from_loader(role_override=assignment_map)

    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    mod_kpis = formatter.format_result("Modified", mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )

    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result("Base", base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    fixed_role_order = [
        "→ Intern usage",
        "→ Resident usage",
        "→ Specialist usage",
        "→ Junior Nurse usage",
        "→ Senior Nurse usage",
    ]
    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = sorted(set(base_dict.keys()) | set(mod_dict.keys()))
    remaining_labels = sorted(label for label in all_labels_set if label not in fixed_role_order)
    final_order = remaining_labels + fixed_role_order
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in final_order]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity="Reassigned staff roles",
                           zip=zip)

@scenario.route('/simulate/reassign_task_composition', methods=['GET'])
def reassign_task_composition():
    activities = [
        "접수", "협진연결", "협진진료", "초진", "분류", "처방", "퇴실결정",
        "퇴실(입원)", "퇴실(사망)", "퇴실(전원)", "퇴실(귀가)", "입원결정", "입원수속", "수술결정",
        "중환자실입실협진의뢰시간", "중환자실입실협진회신시간", "visit_start", "visit_end"
    ]
    role_map = {
        "접수": ["Intern", "Resident", "Specialist"],
        "협진연결": ["Intern", "Resident", "Specialist"],
        "협진진료": ["Intern", "Resident", "Specialist"],
        "초진": ["Intern", "Resident", "Specialist"],
        "분류": ["Intern", "Resident", "Specialist", "Junior", "Senior"],
        "처방": ["Intern", "Resident", "Specialist"],
        "퇴실결정": ["Intern", "Resident", "Specialist"],
        "퇴실(입원)": ["Intern", "Resident", "Specialist"],
        "퇴실(사망)": ["Intern", "Resident", "Specialist"],
        "퇴실(전원)": ["Intern", "Resident", "Specialist"],
        "퇴실(귀가)": ["Intern", "Resident", "Specialist"],
        "입원결정": ["Intern", "Resident", "Specialist"],
        "입원수속": ["Intern", "Resident", "Specialist"],
        "수술결정": ["Intern", "Resident", "Specialist"],
        "중환자실입실협진의뢰시간": ["Intern", "Resident", "Specialist"],
        "중환자실입실협진회신시간": ["Intern", "Resident", "Specialist"],
        "visit_start": ["Intern", "Resident", "Specialist"],
        "visit_end" : ["Intern", "Resident", "Specialist", "Junior", "Senior"]
    }
    doctor_roles = ['Intern', 'Resident', 'Specialist']
    nurse_roles = ['Junior', 'Senior']
    return render_template('reassign_task_composition.html', 
                           activities=activities, 
                           role_map=role_map, 
                           doctor_roles=doctor_roles,
                           nurse_roles=nurse_roles)


@scenario.route('/scenario/submit_reassign_task', methods=['POST'])
def submit_reassign_task():
    candidate_activities = [
        "접수", "협진연결", "협진진료", "초진", "분류", "처방", "퇴실결정",
        "퇴실(입원)", "퇴실(사망)", "퇴실(전원)", "퇴실(귀가)", "입원결정", "입원수속", "수술결정",
        "중환자실입실협진의뢰시간", "중환자실입실협진회신시간", "visit_start", "visit_end"
    ]
    task_config = {}
    for activity in candidate_activities:
        val = request.form.get(activity)
        if val:
            # 예: "Doctor:0.2,Nurse:0.8"
            parts = val.split(',')
            role_prob = {}
            for part in parts:
                role, prob = part.split(':')
                role_prob[role] = float(prob)
            task_config[activity] = role_prob
    # print(task_config)
    sim = simulate_from_loader(task_composition=task_config)

    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    mod_kpis = formatter.format_result("Modified", mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )

    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result("Base", base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    fixed_role_order = [
        "→ Intern usage",
        "→ Resident usage",
        "→ Specialist usage",
        "→ Junior Nurse usage",
        "→ Senior Nurse usage",
    ]
    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = sorted(set(base_dict.keys()) | set(mod_dict.keys()))
    remaining_labels = sorted(label for label in all_labels_set if label not in fixed_role_order)
    final_order = remaining_labels + fixed_role_order
    # print(final_order)
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in final_order]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity="Reassigned staff roles",
                           zip=zip)

@scenario.route('/simulate/merge_shift_roles', methods=['GET'])
def merge_shift_roles():
    return render_template("merge_shift_roles.html")


@scenario.route('/scenario/submit_merge_roles', methods=['POST'])
def submit_merge_roles():
    # 1. Merge Roles 설정
    merged_roles = {}
    if request.form.get('merge_doctor'):
        merged_roles.update({'Intern': 'Junior Doctor', 'Resident': 'Junior Doctor'})
    if request.form.get('merge_nurse'):
        merged_roles.update({'Junior': 'Nurse', 'Senior': 'Nurse'})

    base_shift_times = {
            'Doctor': {
                'shift_1' : (8, 20),
                'shift_2' : (20, 8)
            },
            'Nurse': {
                'shift_1' : (6, 14),
                'shift_2' : (14, 22),
                'shift_3' : (22, 6)
            },
        }

    # 2. Shift 시간 구성
    doctor_shifts = {
        'shift_1': (request.form.get('doc_shift_1_start'), request.form.get('doc_shift_1_end')),
        'shift_2': (request.form.get('doc_shift_2_start'), request.form.get('doc_shift_2_end')),
    }
    nurse_shifts = {
        'shift_1': (request.form.get('nurse_shift_1_start'), request.form.get('nurse_shift_1_end')),
        'shift_2': (request.form.get('nurse_shift_2_start'), request.form.get('nurse_shift_2_end')),
        'shift_3': (request.form.get('nurse_shift_3_start'), request.form.get('nurse_shift_3_end')),
    }
    print(doctor_shifts)
    print(nurse_shifts)
    shift_time_config = parse_shift_times(request.form, default_shift_times=base_shift_times)
    print(shift_time_config)
    # 3. 시뮬레이션 실행
    sim = simulate_from_loader(role_merge=merged_roles, shift_times=shift_time_config)

    # KPI 비교
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    mod_kpis = formatter.format_result("Modified", mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities, role_merge=merged_roles, shift_times=shift_time_config)

    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result("Base", base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities, shift_times=base_shift_times)
    
    # 병합 여부 확인
    doctor_merged = 'Intern' in merged_roles or 'Resident' in merged_roles
    nurse_merged = 'Junior' in merged_roles or 'Senior' in merged_roles

    # 고정 KPI 순서 조건 설정
    fixed_role_order = [
        "→ Intern usage",
        "→ Resident usage",
        "→ Specialist usage",
    ]
    if doctor_merged:
        fixed_role_order.append("→ Junior Doctor (merged)")
    fixed_role_order += [
        "→ Junior Nurse usage",
        "→ Senior Nurse usage",
    ]
    if nurse_merged:
        fixed_role_order.append("→ Nurse (merged)")

    # shift time 항목은 항상 추가
    fixed_role_order += [
        "Doctor shift_1 time",
        "Doctor shift_2 time",
        "Nurse shift_1 time",
        "Nurse shift_2 time",
        "Nurse shift_3 time"
    ]
    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = sorted(set(base_dict.keys()) | set(mod_dict.keys()))
    remaining_labels = sorted(label for label in all_labels_set if label not in fixed_role_order)
    final_order = remaining_labels + fixed_role_order
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in final_order]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity="Merged roles and customized shift times",
                           zip=zip)

@scenario.route('/simulate/add_activity', methods=['GET'])
def add_activity():
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    original_variants = df_date_re['variant']
    seen = set()
    unique_variants = []
    for v in original_variants:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_variants.append(v)
    return render_template('adding_new_activity.html', variants=unique_variants, enumerate=enumerate)


@scenario.route('/scenario/submit_add_activity', methods=['POST'])
def submit_add_activity():
    new_activity = request.form.get('activity_name')
    selected_indices = [int(i) for i in request.form.getlist('selected_variants')]

    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    original_variants = df_date_re['variant']

    new_variants = [list(v) for v in original_variants]
    for idx in selected_indices:
        original_variant = list(original_variants[idx])
        pos_key = f'insert_pos_{idx}'
        pos_val = request.form.get(pos_key, '').strip()

        if pos_val == '':
            modified_variant = original_variant + [new_activity]
        else:
            insert_pos = max(0, min(len(original_variant), int(pos_val)))
            modified_variant = original_variant[:insert_pos] + [new_activity] + original_variant[insert_pos:]
        new_variants[idx] = modified_variant

    print(len(new_variants))
    # Run simulation with updated variants
    sim = simulate_from_loader(custom_variants=new_variants)

    def safe_diff(x, i1, i2):
        try:
            return x.iloc[i1] - x.iloc[i2]
        except:
            return pd.Timedelta(0)

    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    mod_kpis = formatter.format_result(
        "Modified", mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result(
        "Base", base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels = sorted(set(base_dict.keys()) | set(mod_dict.keys()))
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in all_labels]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity=f"Added '{new_activity}' to {len(selected_indices)} variants",
                           zip=zip)

@scenario.route('/simulate/reduce_lab_boarding', methods=['GET'])
def reduce_lab_boarding():
    # GET으로 화면 표시
    return render_template('reducing_lab_boarding.html')


@scenario.route('/scenario/submit_reduce_lab_boarding', methods=['POST'])
def submit_reduce_lab_boarding():
    # 1. 사용자 입력값 읽기
    lab_ratio = float(request.form.get('lab_reduce_ratio', 0)) / 100.0  # 0.0 ~ 1.0
    boarding_ratio = float(request.form.get('boarding_reduce_ratio', 0)) / 100.0

    # 2. 시뮬레이션에 적용
    #    여기서 simulate_from_loader 호출 시 추가 파라미터로 전달
    sim = simulate_from_loader(
        lab_time_reduce=lab_ratio,
        boarding_time_reduce=boarding_ratio
    )

    # 3. 결과 KPI 계산
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    def safe_diff(x, i1, i2):
        try:
            return x.iloc[i1] - x.iloc[i2]
        except:
            return pd.Timedelta(0)

    mod_kpis = formatter.format_result(
        "Modified",
        mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )

    # baseline
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result(
        "Base",
        base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    comparison_kpis = [(label, dict(base_kpis).get(label, ''), dict(mod_kpis).get(label, ''))
                       for label in sorted(set(dict(base_kpis).keys()) | set(dict(mod_kpis).keys()))]

    return render_template(
        'comparison2.html',
        comparison_kpis=comparison_kpis,
        activity="Reducing Lab & Boarding Times",
        zip=zip
    )

@scenario.route('/simulate/vary_patient_demand', methods=['GET'])
def vary_patient_demand():
    # GET으로 화면 표시
    return render_template('varying_patient_demand.html')

@scenario.route('/scenario/submit_vary_demand', methods=['POST'])
def submit_vary_demand():
    # 입력 값
    multiplier = int(request.form.get('demand_multiplier', 100))
    print(f"[INFO] Patient demand multiplier: {multiplier}%")
    def safe_diff(x, i1, i2):
        try:
            return x.iloc[i1] - x.iloc[i2]
        except:
            return pd.Timedelta(0)

    # 기존 시뮬레이션에서 variants/cases를 얻음
    base_sim = simulate_from_loader()
    base_cases = base_sim.cases
    base_arrivals = base_sim.arrival_times
    variants = base_sim.variants
    base_df = pd.DataFrame(base_sim.results)
    base_kpis = formatter.format_result(
        "Base",
        base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )
    # multiplier 적용
    if multiplier == 100:
        custom_cases = base_cases
        custom_arrivals = base_arrivals
    else:
        # 단순하게 multiplier 배 만큼 복제 (실제 논리 맞게 수정 가능)
        times_to_duplicate = multiplier / 100.0
        new_count = int(len(base_cases) * times_to_duplicate)
        # case 리스트와 arrival time을 늘림
        custom_cases = (base_cases * ((new_count // len(base_cases)) + 1))[:new_count]
        custom_arrivals = (base_arrivals * ((new_count // len(base_arrivals)) + 1))[:new_count]
        # arrival_times를 조금씩 offset을 주고 싶다면 여기에 로직 추가 가능

    # 시뮬레이션 재실행
    sim = simulate_from_loader(custom_cases=custom_cases, custom_arrival_times=custom_arrivals)

    # KPI 추출
    df = pd.DataFrame(sim.results)
    df['Activity'] = df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    

    mod_kpis = formatter.format_result(
        "Demand Modified",
        df,
        df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )
    comparison_kpis = [(label, dict(base_kpis).get(label, ''), dict(mod_kpis).get(label, ''))
                       for label in dict(base_kpis).keys()]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity=f"Varying Patient Demand ({multiplier}%)",
                           zip=zip)

@scenario.route('/simulate/vary_queue', methods=['GET'])
def vary_queue():
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    original_variants = df_date_re['variant']
    seen = set()
    unique_variants = []
    for v in original_variants:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_variants.append(v)
    return render_template('varying_queue_discipline.html', variants=unique_variants, enumerate=enumerate)

@scenario.route('/scenario/submit_vary_queue', methods=['POST'])
def submit_vary_queue():
    discipline = request.form.get('queue_discipline', 'FIFO')

    variant_priority_map = None
    if discipline == 'PRIORITY':
        df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
        original_variants = df_date_re['variant']
        seen = set()
        unique_variants = []
        for v in original_variants:
            t = tuple(v)
            if t not in seen:
                seen.add(t)
                unique_variants.append(v)

        variant_priority_map = {}
        # HTML에서 priority_0, priority_1 ... 형식으로 넘어온 값을 읽기
        for idx in range(len(unique_variants)):
            priority_val = request.form.get(f'priority_{idx}', '')
            if priority_val.strip() != '':
                try:
                    priority_val = int(priority_val)
                except:
                    priority_val = 5  # 기본값
            else:
                priority_val = 5
            # tuple로 변환한 variant를 key로 사용
            variant_priority_map[tuple(unique_variants[idx])] = priority_val

    # 시뮬레이션 실행 (우선순위 맵 전달)
    sim = simulate_from_loader(
        queue_discipline=discipline,
        variant_priority_map=variant_priority_map
    )

    # --- Modified 시뮬레이션 결과 처리 ---
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    def safe_diff(x, i1, i2):
        try:
            return x.iloc[i1] - x.iloc[i2]
        except:
            return pd.Timedelta(0)

    mod_kpis = formatter.format_result(
        "Modified",
        mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )

    # --- Baseline ---
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result(
        "Base",
        base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    # --- 비교 결과 ---
    comparison_kpis = [(label, dict(base_kpis).get(label, ''), dict(mod_kpis).get(label, ''))
                       for label in dict(base_kpis).keys()]

    return render_template(
        'comparison2.html',
        comparison_kpis=comparison_kpis,
        activity=f"Varying Queue Discipline (Mode: {discipline})",
        zip=zip
    )

@scenario.route('/simulate/vary_sequence', methods=['GET'])
def vary_sequence():
    # 데이터에서 unique variants 추출
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    original_variants = df_date_re['variant']
    seen = set()
    unique_variants = []
    for v in original_variants:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_variants.append(v)
    return render_template('varying_patient_sequence.html', variants=unique_variants, enumerate=enumerate)


@scenario.route('/scenario/submit_vary_sequence', methods=['POST'])
def submit_vary_sequence():
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    base_variants = list(df_date_re['variant']) 

    seen = set()
    unique_variants = []
    for v in base_variants:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_variants.append(v)

    modified_map = {}  
    for idx, uv in enumerate(unique_variants):
        key = f'new_variant_{idx}'
        new_val = request.form.get(key, '').strip()
        if new_val:  # 사용자가 변경 입력했을 때만
            new_variant_list = [x.strip() for x in new_val.split(',')]
            modified_map[tuple(uv)] = new_variant_list

    # 최종 variants 구성 (전체 case 수 동일 유지)
    new_variants = []
    for v in base_variants:
        t = tuple(v)
        if t in modified_map:
            new_variants.append(modified_map[t])
        else:
            new_variants.append(v)

    # 이제 new_variants를 이용하여 시뮬레이션 실행
    sim = simulate_from_loader(custom_variants=new_variants)

    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    def safe_diff(x, i1, i2):
        try:
            return x.iloc[i1] - x.iloc[i2]
        except:
            return pd.Timedelta(0)

    mod_kpis = formatter.format_result(
        "Modified",
        mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )

    # baseline
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result(
        "Base",
        base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    comparison_kpis = [(label, dict(base_kpis).get(label, ''), dict(mod_kpis).get(label, ''))
                       for label in dict(base_kpis).keys()]

    return render_template(
        'comparison2.html',
        comparison_kpis=comparison_kpis,
        activity="Varying Patient Sequence",
        zip=zip
    )

@scenario.route('/scenario/vary_sequence2', methods=['GET', 'POST'])
def vary_sequence2():
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    all_variants = df_date_re['variant'].tolist()
    all_activities = sorted(set(act for v in all_variants for act in v))

    variants_for_drag = []  # 기본값: drag&drop UI는 안 보이도록
    selected_activity = None

    if request.method == 'POST':
        selected_activity = request.form.get('selected_activity')
        if selected_activity:
            # 선택한 activity를 포함하는 variants 필터링
            from collections import Counter
            filtered = [tuple(v) for v in all_variants if selected_activity in v]
            freq = Counter(filtered)
            sorted_variants = sorted(freq.items(), key=lambda x: x[1], reverse=True)

            # drag&drop으로 보여줄 variant 후보들 생성 (빈도순)
            variants_for_drag = [
                {
                    'index': idx,
                    'variant': ' > '.join(var),
                    'activities': list(var),
                    'count': count
                }
                for idx, (var, count) in enumerate(sorted_variants[:5])  # 상위 5개 예시
            ]

    return render_template(
        'vary_sequence.html',
        activities=all_activities,
        variants=variants_for_drag,
        selected_activity=selected_activity
    )


@scenario.route('/scenario/submit_vary_sequence2', methods=['POST'])
def submit_vary_sequence2():
    # 기존 variants 전체 불러오기
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    all_variants = df_date_re['variant'].tolist()

    # drag&drop으로 넘어온 최대 5개 변형 정보 수집
    modified_orders = []
    for idx in range(5):
        order_str = request.form.get(f'new_order_{idx}', '')
        if order_str.strip():
            modified_orders.append([a.strip() for a in order_str.split(',') if a.strip()])
        else:
            modified_orders.append(None)  # None이면 해당 variant는 수정하지 않음

    # 상위 5개 variant를 대표로 뽑아둔 목록 (GET에서 넘겨준 것과 동일한 로직을 사용해야 함)
    seen = set()
    unique_variants = []
    for v in all_variants:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_variants.append(v)
    top5_variants = unique_variants[:5]

    stack = session.get('scenario_stack', [])
    stack.append({
        'type': 'resequence_activity',
        'params': {
            'modified_orders': modified_orders,
            'top5_variants': top5_variants
        }
    })
    print(stack)
    session['scenario_stack'] = stack
    session.modified = True

    # 수정된 variants 리스트 생성
    updated_variants = []
    for v in all_variants:
        # v가 top5_variants 중 어떤 것과 "정확히 같은지" 확인
        matched_index = None
        for idx, base in enumerate(top5_variants):
            if v == base:
                matched_index = idx
                break

        if matched_index is not None and modified_orders[matched_index] is not None:
            # 해당 variant에 drag&drop으로 변경된 순서를 적용
            new_order = modified_orders[matched_index]
            selected = [act for act in new_order if act in v]
            remaining = [act for act in v if act not in new_order]
            updated_variants.append(selected + remaining)
        else:
            # drag&drop으로 변경하지 않았다면 기존 variant 그대로 사용
            updated_variants.append(v)

    # 시뮬레이션 실행
    sim = simulate_from_loader(custom_variants=updated_variants)
    mod_df = pd.DataFrame(sim.results)

    def safe_diff(x, i1, i2):
        try:
            return x.iloc[i1] - x.iloc[i2]
        except:
            return pd.Timedelta(0)

    mod_kpis = formatter.format_result(
        "Modified",
        mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )

    # baseline
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result(
        "Base",
        base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    comparison_kpis = [
        (label, dict(base_kpis).get(label, ''), dict(mod_kpis).get(label, ''))
        for label in dict(base_kpis).keys()
    ]

    return render_template(
        'comparison2.html',
        comparison_kpis=comparison_kpis,
        activity="Resequencing activities",
        zip=zip
    )


# 시나리오 스택 초기화
@scenario.route('/scenario/clear_stack', methods=['GET'])
def clear_stack():
    session['scenario_stack'] = []
    return redirect(url_for('scenario.scenario_map'))

# 세션에 스택 초기화 및 스택에 시나리오 추가하는 함수 추가
@scenario.route('/scenario/add_to_stack', methods=['POST'])
def add_to_stack():
    scenario_id = request.form.get('scenario_id')
    if 'scenario_stack' not in session:
        session['scenario_stack'] = []
    stack = session['scenario_stack']

    params = {}

    if scenario_id == 'remove_activity':
        params['activity'] = request.form.get('activity')
    elif scenario_id == 'resequence_activity':
        # # POST로 넘어온 modified_orders, top5_variants 받아서 세션에 추가
        # modified_orders = json.loads(request.form.get('modified_orders'))  # JS에서 JSON.stringify로 보내기
        # top5_variants = json.loads(request.form.get('top5_variants'))

        # stack = session.get('scenario_stack', [])
        # stack.append({
        #     'type': 'resequence_activity',
        #     'params': {
        #         'modified_orders': modified_orders,
        #         'top5_variants': top5_variants
        #     }
        # })
        # session['scenario_stack'] = stack
        # session.modified = True
        pass
    elif scenario_id == 'reallocating':
        params['doc_shift_1'] = int(request.form.get('doctor_shift_0', 0))
        params['doc_shift_2'] = int(request.form.get('doctor_shift_1', 0))
        params['nur_shift_1'] = int(request.form.get('nurse_shift_0', 0))
        params['nur_shift_2'] = int(request.form.get('nurse_shift_1', 0))
        params['nur_shift_3'] = int(request.form.get('nurse_shift_2', 0))
    elif scenario_id == 'adding_staff':
        params['doc_shift_1'] = int(request.form.get('doctor_shift_0', 0))
        params['doc_shift_2'] = int(request.form.get('doctor_shift_1', 0))
        params['nur_shift_1'] = int(request.form.get('nurse_shift_0', 0))
        params['nur_shift_2'] = int(request.form.get('nurse_shift_1', 0))
        params['nur_shift_3'] = int(request.form.get('nurse_shift_2', 0))
    elif scenario_id == 'rostering':
        doctor_roles = ['Intern', 'Resident', 'Specialist']
        nurse_roles = ['Junior', 'Senior']
        doctor_shifts = ['shift_1', 'shift_2']
        nurse_shifts = ['shift_1', 'shift_2', 'shift_3']

        # params를 dict로 구성
        for role in doctor_roles:
            for shift in doctor_shifts:
                field = f"{role}_{shift}"
                params[field] = int(request.form.get(field, 0))
        for role in nurse_roles:
            for shift in nurse_shifts:
                field = f"{role}_{shift}"
                params[field] = int(request.form.get(field, 0))
    elif scenario_id == 'vary_the_assignment':
        assignment_map = {}
        # 폼에 담긴 activity-role 매핑을 그대로 수집
        for key, value in request.form.items():
            if value:  # 빈 값은 제외
                assignment_map[key] = value
        params['role_override'] = assignment_map
    elif scenario_id == 'vary_task_comp':
        candidate_activities = [
            "접수", "협진연결", "협진진료", "초진", "분류", "처방", "퇴실결정",
            "퇴실(입원)", "퇴실(사망)", "퇴실(전원)", "퇴실(귀가)", "입원결정", "입원수속", "수술결정",
            "중환자실입실협진의뢰시간", "중환자실입실협진회신시간", "visit_start", "visit_end"
        ]
        task_config = {}
        for activity in candidate_activities:
            val = request.form.get(activity)
            if val:
                # 입력 형식: "Doctor:0.2,Nurse:0.8"
                parts = val.split(',')
                role_prob = {}
                for part in parts:
                    role, prob = part.split(':')
                    role_prob[role] = float(prob)
                task_config[activity] = role_prob

        params['task_composition'] = task_config
    elif scenario_id == 'merg_role':
        # 1. role_merge dict 구성
        merged_roles = {}
        if request.form.get('merge_doctor'):
            merged_roles.update({'Intern': 'Junior Doctor', 'Resident': 'Junior Doctor'})
        if request.form.get('merge_nurse'):
            merged_roles.update({'Junior': 'Nurse', 'Senior': 'Nurse'})

        # 2. shift_times dict 구성
        base_shift_times = {
            'Doctor': {
                'shift_1': (8, 20),
                'shift_2': (20, 8)
            },
            'Nurse': {
                'shift_1': (6, 14),
                'shift_2': (14, 22),
                'shift_3': (22, 6)
            }
        }
        shift_time_config = parse_shift_times(request.form, default_shift_times=base_shift_times)

        # 3. params로 묶어 스택에 추가
        params['role_merge'] = merged_roles
        params['shift_times'] = shift_time_config

    elif scenario_id == 'add_new_activity':
        new_activity = request.form.get('activity_name')

        # 선택된 variant index 리스트
        selected_indices = [int(i) for i in request.form.getlist('selected_variants')]

        # 각 variant별 삽입 위치도 같이 담아두기
        insert_positions = {}
        for idx in selected_indices:
            pos_key = f'insert_pos_{idx}'
            pos_val = request.form.get(pos_key, '').strip()
            if pos_val == '':
                insert_positions[idx] = None   # 맨 끝에 삽입
            else:
                insert_positions[idx] = int(pos_val)

        # params로 저장
        params['new_activity'] = new_activity
        params['selected_indices'] = selected_indices
        params['insert_positions'] = insert_positions

    elif scenario_id == 'reduce_time':
        lab_ratio = float(request.form.get('lab_reduce_ratio', 0)) / 100.0
        boarding_ratio = float(request.form.get('boarding_reduce_ratio', 0)) / 100.0

        # 스택에 담을 params
        params['lab_time_reduce'] = lab_ratio
        params['boarding_time_reduce'] = boarding_ratio

    elif scenario_id == 'vary_patient_demand':
        multiplier = int(request.form.get('demand_multiplier', 100))
        params['demand_multiplier'] = multiplier

    elif scenario_id == 'vary_patient_sequences':
        # unique_variants와 매핑된 변경 정보(modified_map)를 구성
        df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
        base_variants = list(df_date_re['variant'])

        seen = set()
        unique_variants = []
        for v in base_variants:
            t = tuple(v)
            if t not in seen:
                seen.add(t)
                unique_variants.append(v)

        modified_map = {}
        for idx, uv in enumerate(unique_variants):
            key = f'new_variant_{idx}'
            new_val = request.form.get(key, '').strip()
            if new_val:  # 사용자가 수정한 variant만
                new_variant_list = [x.strip() for x in new_val.split(',')]
                modified_map[tuple(uv)] = new_variant_list

        # params로 전달 (dict는 JSON 직렬화 가능하도록 list로 변환)
        serializable_map = {','.join(k): v for k, v in modified_map.items()}
        params['modified_map'] = serializable_map

    elif scenario_id == 'vary_queue_discipline':
        # 1. discipline 가져오기
        discipline = request.form.get('queue_discipline', 'FIFO')
        params['queue_discipline'] = discipline

        # 2. priority 모드라면 priority 값들도 수집
        if discipline == 'PRIORITY':
            df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
            original_variants = df_date_re['variant']
            seen = set()
            unique_variants = []
            for v in original_variants:
                t = tuple(v)
                if t not in seen:
                    seen.add(t)
                    unique_variants.append(v)

            variant_priority_map = {}
            for idx, uv in enumerate(unique_variants):
                priority_val = request.form.get(f'priority_{idx}', '')
                if priority_val.strip() != '':
                    try:
                        priority_val = int(priority_val)
                    except:
                        priority_val = 5  # 기본값
                else:
                    priority_val = 5
                variant_priority_map[','.join(uv)] = priority_val  # 세션 저장용 직렬화

            params['variant_priority_map'] = variant_priority_map
    
    stack.append({
        'type': scenario_id,
        'params':params
    })
    session['scenario_stack'] = stack
    session.modified = True
    return redirect(url_for('scenario.scenario_map'))

@scenario.route('/scenario/run_stack', methods=['GET'])
def run_stack():
    stack = session.get('scenario_stack', [])

    # 기본 베이스라인 시뮬레이션
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    def safe_diff(x, i1, i2):
        try:
            return x.iloc[i1] - x.iloc[i2]
        except:
            return pd.Timedelta(0)
        
    base_kpis = formatter.format_result(
        "Base",
        base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    current_variants = base_sim.variants
    current_cases = base_sim.cases
    current_arrivals = base_sim.arrival_times

    step_names = ["Base"]
    kpi_snapshots = [dict(base_kpis)]

    for step in stack:
        sim = None

        if step['type'] == 'remove_activity':
            activity_to_remove = step['params'].get('activity')
            sim = simulate_from_loader(
                remove_activity=activity_to_remove,
                custom_variants=current_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals
            )

        elif step['type'] == 'resequence_activity':
            modified_orders = step['params'].get('modified_orders', [])
            print(modified_orders)
            seen = set()
            unique_variants = []
            for v in current_variants:
                t = tuple(v)
                if t not in seen:
                    seen.add(t)
                    unique_variants.append(v)
            top5_variants = unique_variants[:5]

            updated_variants = []
            for v in current_variants:
                matched_index = None
                for idx, base in enumerate(top5_variants):
                    if v == base:
                        matched_index = idx
                        break
                if matched_index is not None and matched_index < len(modified_orders) and modified_orders[matched_index] is not None:
                    new_order = modified_orders[matched_index]
                    selected = [act for act in new_order if act in v]
                    remaining = [act for act in v if act not in new_order]
                    updated_variants.append(selected + remaining)
                else:
                    updated_variants.append(v)

            # 시뮬레이션 실행 (누적된 current_cases / current_arrivals 사용)
            sim = simulate_from_loader(
                custom_variants=updated_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals
            )

        elif step['type'] == 'reallocating':
            sim = simulate_from_loader(
                custom_variants=current_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals,
                doc_shift_1=step['params'].get('doc_shift_1', 0),
                doc_shift_2=step['params'].get('doc_shift_2', 0),
                nur_shift_1=step['params'].get('nur_shift_1', 0),
                nur_shift_2=step['params'].get('nur_shift_2', 0),
                nur_shift_3=step['params'].get('nur_shift_3', 0),
            )

        elif step['type'] == 'adding_staff':

            sim = simulate_from_loader(
                custom_variants=current_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals,
                doc_shift_1=step['params'].get('doc_shift_1', 0),
                doc_shift_2=step['params'].get('doc_shift_2', 0),
                nur_shift_1=step['params'].get('nur_shift_1', 0),
                nur_shift_2=step['params'].get('nur_shift_2', 0),
                nur_shift_3=step['params'].get('nur_shift_3', 0),
            )

        elif step['type'] == 'rostering':
            sim = simulate_from_loader(
                custom_variants=current_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals,
                **step['params'].get('resource_config', {})
            )

        elif step['type'] == 'vary_the_assignment':
            sim = simulate_from_loader(
                custom_variants=current_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals,
                role_override=step['params'].get('role_override', {})
            )

        elif step['type'] == 'vary_task_comp':
            sim = simulate_from_loader(
                custom_variants=current_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals,
                task_composition=step['params'].get('task_composition', {})
            )

        elif step['type'] == 'merg_role':
            sim = simulate_from_loader(
                custom_variants=current_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals,
                role_merge=step['params'].get('role_merge', {}),
                shift_times=step['params'].get('shift_times', {})
            )

        elif step['type'] == 'add_new_activity':
            new_variants = [list(v) for v in current_variants]
            new_act = step['params'].get('activity_name')
            indices = step['params'].get('selected_indices', [])
            positions = step['params'].get('insert_positions', {})

            for idx in indices:
                if idx < 0 or idx >= len(new_variants):
                    continue
                orig = list(new_variants[idx])
                pos_val = positions.get(str(idx), '')
                if pos_val == '' or pos_val is None:
                    modified = orig + [new_act]
                else:
                    try:
                        insert_pos = int(pos_val)
                    except:
                        insert_pos = len(orig)
                    insert_pos = max(0, min(len(orig), insert_pos))
                    modified = orig[:insert_pos] + [new_act] + orig[insert_pos:]
                new_variants[idx] = modified

            sim = simulate_from_loader(
                custom_variants=new_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals
            )

        elif step['type'] == 'reduce_time':
            sim = simulate_from_loader(
                custom_variants=current_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals,
                lab_time_reduce=step['params'].get('lab_time_reduce', 0),
                boarding_time_reduce=step['params'].get('boarding_time_reduce', 0)
            )

        elif step['type'] == 'vary_patient_demand':
            multiplier = step['params'].get('demand_multiplier', 100)
            custom_cases, custom_arrivals = apply_multiplier(current_cases, current_arrivals, multiplier)
            sim = simulate_from_loader(
                custom_cases=custom_cases,
                custom_arrival_times=custom_arrivals,
                custom_variants=current_variants
            )

        elif step['type'] == 'vary_patient_sequences':
            raw_map = step['params'].get('modified_map', {})
            new_variants = []
            for v in current_variants:
                t = ','.join(v)
                if t in raw_map:
                    new_variants.append(raw_map[t])
                else:
                    new_variants.append(v)
            sim = simulate_from_loader(
                custom_variants=new_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals
            )

        elif step['type'] == 'vary_queue_discipline':
            discipline = step['params'].get('queue_discipline', 'FIFO')
            v_map_raw = step['params'].get('variant_priority_map', None)
            variant_priority_map = None
            if v_map_raw:
                variant_priority_map = {tuple(k.split(',')): v for k, v in v_map_raw.items()}
            sim = simulate_from_loader(
                queue_discipline=discipline,
                variant_priority_map=variant_priority_map,
                custom_variants=current_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals
            )

        if sim:
            current_variants = sim.variants
            current_cases = sim.cases
            current_arrivals = sim.arrival_times

            df = pd.DataFrame(sim.results)
            df['Activity'] = df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

            kpis = formatter.format_result(
                step['type'],
                df,
                df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
                df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
                sim.resource_capacities
            )

            step_names.append(step['type'])
            kpi_snapshots.append(dict(kpis))


    all_labels = sorted(set().union(*[k.keys() for k in kpi_snapshots]))
    comparison_kpis = []

    for label in all_labels:
        row = [label]
        for snap in kpi_snapshots:
            row.append(snap.get(label, ''))
        comparison_kpis.append(row)

    return render_template(
        'comparison.html',
        comparison_kpis=comparison_kpis,
        step_names=step_names,
        activity="Stacked Scenarios",
        zip=zip
    )