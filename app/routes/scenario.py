from flask import Blueprint, render_template, request, redirect, url_for, session
import json
import os
from app.simulator import formatter
from app.simulator.scenario import simulate_from_loader
from urllib.parse import urlencode
import pandas as pd
import pickle
from copy import deepcopy
import re
from datetime import timedelta

scenario = Blueprint('scenario', __name__)

def parse_shift_times(form, default_shift_times):
    def to_hour(time_str):
        try:
            return int(time_str)
        except (ValueError, TypeError):
            return None

    # 리소스 풀 구조에 맞게 role 목록 정의
    roles = [
        ("Intern", ["shift_1", "shift_2"]),
        ("Resident", ["shift_1", "shift_2"]),
        ("Specialist", ["shift_1", "shift_2"]),
        ("Junior", ["shift_1", "shift_2", "shift_3"]),
        ("Senior", ["shift_1", "shift_2", "shift_3"])
    ]

    parsed = {}

    for role_name, shifts in roles:
        parsed[role_name] = {}
        for shift_name in shifts:
            # HTML form의 name 속성과 일치
            start_key = f'{role_name.lower()}_{shift_name}_start'
            end_key = f'{role_name.lower()}_{shift_name}_end'

            start_raw = form.get(start_key)
            end_raw = form.get(end_key)

            start_hour = to_hour(start_raw)
            end_hour = to_hour(end_raw)

            # default_shift_times는 같은 구조로 들어온다고 가정
            default_for_this = default_shift_times.get(role_name, {}).get(shift_name, (0, 0))

            if start_hour is None or end_hour is None:
                parsed[role_name][shift_name] = default_for_this
            else:
                parsed[role_name][shift_name] = (start_hour, end_hour)

    return parsed

def parse_shift(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0
    
def apply_multiplier(base_cases, base_arrivals, base_variants, multiplier):
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
    custom_variants = (base_variants * ((new_count // len(base_variants)) + 1))[:new_count]

    # arrival_times가 단순 반복이면 같은 시간이 중복될 수 있으니 약간씩 offset을 줄 수도 있음
    # 예시: 반복되는 배수에 따라 도착 시간에 작은 증가량 추가
    offset_arrivals = []
    batch_size = len(base_arrivals)
    for i in range(new_count):
        base_time = custom_arrivals[i % batch_size]
        batch_index = i // batch_size
        offset_time = base_time + batch_index * 0.1  # 0.1초씩 offset
        offset_arrivals.append(offset_time)

    return custom_cases, offset_arrivals, custom_variants

def _to_number_or_minutes(v):
    """
    KPI 값이 숫자(str/int/float)거나 '0 days HH:MM:SS(.ms)' 같은 시간일 때
    비교 가능한 수치로 변환. 시간은 '분' 단위 float로 반환.
    변환 불가하면 None.
    """
    if v is None:
        return None

    if isinstance(v, (int, float)):
        return float(v)

    s = str(v).strip()
    try:
        return float(s)
    except Exception:
        pass

    m = re.match(r'(?:(\d+)\s*days?\s*)?(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?$', s)
    if m:
        days = int(m.group(1) or 0)
        h, mi, sec = int(m.group(2)), int(m.group(3)), int(m.group(4))
        td = timedelta(days=days, hours=h, minutes=mi, seconds=sec)
        return td.total_seconds() / 60.0

    return None

def _fmt_delta(delta, pct):
    if delta is None:
        return ""
    cls = "pos" if delta > 0 else ("neg" if delta < 0 else "neu")
    sign = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
    abs_val = abs(delta)
    pct_txt = f" ({pct:+.1f}%)" if pct is not None else ""
    return f"<span class='delta {cls}'>{sign} {abs_val:.2f}{pct_txt}</span>"

# 시나리오 JSON 파일 로드
def load_scenarios():
    with open(os.path.join('app', 'data', 'scenario_map.json'), 'r', encoding='utf-8') as f:
        return json.load(f)['scenarios']

@scenario.route('/scenario_map')
def scenario_map():
    scenarios = load_scenarios()
    current_stack = session.get('scenario_stack', [])
    return render_template('scenario_map.html', scenarios=scenarios, stack=current_stack)

@scenario.route('/simulate/run', methods=['POST'])
def run_scenario():
    scenario_id = request.form.get('scenario_id')

    # 선택한 시나리오 저장 또는 시뮬레이션 적용 로직 연결 가능
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

    return redirect(url_for('main.index'))

# -------------------------------------------
# removing activities
# -------------------------------------------

@scenario.route('/simulate/remove_activity_selection', methods=['GET'])
def remove_activity_selection():
    # 시뮬레이션에서 등장하는 액티비티 목록 (현재는 고정, 데이터에 따라 추출하도록도 가능할 듯)
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    all_variants = df_date_re['variant'].tolist()
    activities = sorted(set(act for v in all_variants for act in v))
    remove_set = {'visit_end', 'visit_start'}

    # 세션에서 이미 제거된 활동들 추출
    removed_activities = [
        step['params']['activity']
        for step in session.get('scenario_stack', [])
        if step.get('type') == 'remove_activity'
    ]

    # 제외 목록에 추가
    remove_set.update(removed_activities)
    
    activities = [act for act in activities if act not in remove_set]
    return render_template('remove_activity_selection.html', activities=activities)

@scenario.route('/scenario/submit_remove_activity', methods=['POST'])
def submit_remove_activity():
    activity = request.form.get('activity')
    return redirect(url_for('scenario.remove_activity_loading', activity=activity))

@scenario.route('/simulate/remove_activity_loading', methods=['GET'])
def remove_activity_loading():
    return render_template('loading.html', next_url=url_for('scenario.remove_activity_compare', **request.args))

@scenario.route('/scenario/compare_remove_activity', methods=['GET'])
def remove_activity_compare():
    activity = request.args.get('activity')

    def safe_diff(x, i1, i2):
        try:
            return x.iloc[i1] - x.iloc[i2]
        except:
            return pd.Timedelta(0)

    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    median_dtp_res_base = base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median()
    mean_dtp_res_base = base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean()
    base_kpis = formatter.format_result("Base", base_df, median_dtp_res_base, mean_dtp_res_base, base_sim.resource_capacities)

    mod_sim = simulate_from_loader(remove_activity=activity)
    mod_df = pd.DataFrame(mod_sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    median_dtp_res_mod = mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median()
    mean_dtp_res_mod = mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean()
    mod_kpis = formatter.format_result("Modified", mod_df, median_dtp_res_mod, mean_dtp_res_mod, mod_sim.resource_capacities)

    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = base_dict.keys()
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in all_labels_set]

    return render_template('comparison2.html', activity=f'remove {activity}', comparison_kpis=comparison_kpis, zip=zip)


# -------------------------------------------------------
# reallocating based on staff type and shift
# -------------------------------------------------------

@scenario.route('/simulate/reallocate_shift_selection', methods=['GET'])
def reallocate_shift_selection():
    # 기본 인원 수
    staff_config = {
        'intern_shift_1': 10, 'intern_shift_2': 10,
        'resident_shift_1': 10, 'resident_shift_2': 10,
        'specialist_shift_1': 14, 'specialist_shift_2': 10,
        'junior_shift_1': 2, 'junior_shift_2': 5, 'junior_shift_3': 2,
        'senior_shift_1': 1, 'senior_shift_2': 2, 'senior_shift_3': 1
    }
    # 기본 인원 수 기준 10% 만큼의 옵션을 제공 (몇 % 로 할지 여기서 수정 가능)
    def generate_options(base):
        delta = max(1, int(base * 0.1))
        return list(range(base - delta, base + delta + 1))
    

    staff_options = {name: generate_options(val) for name, val in staff_config.items()}

    return render_template(
        'reallocate_shift_selection.html',
        staff_options=staff_options
    )
@scenario.route('/scenario/submit_reallocate_staff', methods=['POST'])
def submit_reallocate_staff():
    query_params = request.form.to_dict()
    return redirect(url_for('scenario.reallocate_staff_loading', **query_params))
@scenario.route('/simulate/reallocate_staff_loading', methods=['GET'])
def reallocate_staff_loading():
    return render_template('loading.html', next_url=url_for('scenario.reallocate_staff_compare', **request.args))
@scenario.route('/scenario/reallocate_staff_compare', methods=['GET'])
def reallocate_staff_compare():
    form = request.args

    def get_int(name, default):
        try:
            return int(form.get(name, default))
        except:
            return default

    keys = [
        'intern_shift_1','intern_shift_2',
        'resident_shift_1','resident_shift_2',
        'specialist_shift_1','specialist_shift_2',
        'junior_shift_1','junior_shift_2','junior_shift_3',
        'senior_shift_1','senior_shift_2','senior_shift_3'
    ]

    default_config = {
        'intern_shift_1': 10, 'intern_shift_2': 10,
        'resident_shift_1': 10, 'resident_shift_2': 10,
        'specialist_shift_1': 14, 'specialist_shift_2': 10,
        'junior_shift_1': 2, 'junior_shift_2': 5, 'junior_shift_3': 2,
        'senior_shift_1': 1, 'senior_shift_2': 2, 'senior_shift_3': 1
    }

    resource_config = {k: get_int(k, default_config[k]) for k in keys}

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    # Modified simulation
    sim = simulate_from_loader(
        Intern_shift_1=resource_config['intern_shift_1'], Intern_shift_2=resource_config['intern_shift_2'],
        Resident_shift_1=resource_config['resident_shift_1'], Resident_shift_2=resource_config['resident_shift_2'],
        Specialist_shift_1=resource_config['specialist_shift_1'], Specialist_shift_2=resource_config['specialist_shift_2'],
        Junior_shift_1=resource_config['junior_shift_1'], Junior_shift_2=resource_config['junior_shift_2'], Junior_shift_3=resource_config['junior_shift_3'],
        Senior_shift_1=resource_config['senior_shift_1'], Senior_shift_2=resource_config['senior_shift_2'], Senior_shift_3=resource_config['senior_shift_3']
    )
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    mod_kpis = formatter.format_result("Modified", mod_df, 
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )

    # Base simulation
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result("Base", base_df, 
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )
    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = base_dict.keys()
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in all_labels_set]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity=(
                               f"Reallocation: Interns[{resource_config['intern_shift_1']}, {resource_config['intern_shift_2']}], "
                               f"Residents[{resource_config['resident_shift_1']}, {resource_config['resident_shift_2']}], "
                               f"Specialists[{resource_config['specialist_shift_1']}, {resource_config['specialist_shift_2']}], "
                               f"Juniors[{resource_config['junior_shift_1']}, {resource_config['junior_shift_2']}, {resource_config['junior_shift_3']}], "
                               f"Seniors[{resource_config['senior_shift_1']}, {resource_config['senior_shift_2']}, {resource_config['senior_shift_3']}]"
                           ), zip=zip)


# -----------------------------------------------------
# adding staff
# -----------------------------------------------------

@scenario.route('/simulate/adding_staff_selection', methods=['GET'])
def adding_staff_selection():
    # 기본 인원 수
    base_config = {
        'intern_shift_1': 10, 'intern_shift_2': 10,
        'resident_shift_1': 10, 'resident_shift_2': 10,
        'specialist_shift_1': 14, 'specialist_shift_2': 10,
        'junior_shift_1': 2, 'junior_shift_2': 5, 'junior_shift_3': 2,
        'senior_shift_1': 1, 'senior_shift_2': 2, 'senior_shift_3': 1
    }

    # 기본 인원 수 기준 10% 만큼의 옵션을 제공 (몇 % 로 할지 여기서 수정 가능)
    def generate_options(base):
        delta = max(1, int(base * 0.2))
        return list(range(base, base + delta + 1))
    

    # 역할별 옵션 생성
    intern_opts = {k: generate_options(v) for k, v in base_config.items() if 'intern' in k}
    resident_opts = {k: generate_options(v) for k, v in base_config.items() if 'resident' in k}
    specialist_opts = {k: generate_options(v) for k, v in base_config.items() if 'specialist' in k}
    junior_opts = {k: generate_options(v) for k, v in base_config.items() if 'junior' in k}
    senior_opts = {k: generate_options(v) for k, v in base_config.items() if 'senior' in k}

    return render_template('adding_staff_selection.html',
                           intern_opts=intern_opts,
                            resident_opts=resident_opts,
                            specialist_opts=specialist_opts,
                            junior_opts=junior_opts,
                            senior_opts=senior_opts
                            )

@scenario.route('/scenario/submit_adding_staff', methods=['POST'])
def submit_adding_staff():
    query_params = request.form.to_dict()
    return redirect(url_for('scenario.adding_staff_loading', **query_params))

@scenario.route('/simulate/adding_staff_loading', methods=['GET'])
def adding_staff_loading():
    return render_template('loading.html', next_url=url_for('scenario.adding_staff_compare', **request.args))

@scenario.route('/scenario/adding_staff_compare', methods=['GET'])
def adding_staff_compare():
    form = request.args  # 쿼리스트링에서 파라미터 읽기
    # 폼 값 파싱
    intern_shift_1 = int(form.get('intern_shift_1', 0))
    intern_shift_2 = int(form.get('intern_shift_2', 0))
    resident_shift_1 = int(form.get('resident_shift_1', 0))
    resident_shift_2 = int(form.get('resident_shift_2', 0))
    specialist_shift_1 = int(form.get('specialist_shift_1', 0))
    specialist_shift_2 = int(form.get('specialist_shift_2', 0))
    junior_shift_1 = int(form.get('junior_shift_1', 0))
    junior_shift_2 = int(form.get('junior_shift_2', 0))
    junior_shift_3 = int(form.get('junior_shift_3', 0))
    senior_shift_1 = int(form.get('senior_shift_1', 0))
    senior_shift_2 = int(form.get('senior_shift_2', 0))
    senior_shift_3 = int(form.get('senior_shift_3', 0))

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    # Simulation
    sim = simulate_from_loader(
        Intern_shift_1=intern_shift_1, Intern_shift_2=intern_shift_2,
        Resident_shift_1=resident_shift_1, Resident_shift_2=resident_shift_2,
        Specialist_shift_1=specialist_shift_1, Specialist_shift_2=specialist_shift_2,
        Junior_shift_1=junior_shift_1, Junior_shift_2=junior_shift_2, Junior_shift_3=junior_shift_3,
        Senior_shift_1=senior_shift_1, Senior_shift_2=senior_shift_2, Senior_shift_3=senior_shift_3
    )
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    mod_kpis = formatter.format_result("Modified", mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities)

    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result("Base", base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities)

    base_dict, mod_dict = dict(base_kpis), dict(mod_kpis)
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in base_dict.keys()]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity=(
                               f"Adding Staff : Interns[{intern_shift_1},{intern_shift_2}], "
                               f"Residents[{resident_shift_1},{resident_shift_2}], "
                               f"Specialists[{specialist_shift_1},{specialist_shift_2}], "
                               f"Juniors[{junior_shift_1},{junior_shift_2},{junior_shift_3}], "
                               f"Seniors[{senior_shift_1},{senior_shift_2},{senior_shift_3}]"
                           ), zip=zip)

# -------------------------------------------
# rostering
# -------------------------------------------

@scenario.route('/simulate/rostering_form', methods=['GET'])
def rostering_form():
    roles = ['Intern', 'Resident', 'Specialist', 'Junior', 'Senior']
    shifts = ['shift_1', 'shift_2', 'shift_3']
    return render_template('rostering_form.html', roles=roles, shifts=shifts)

@scenario.route('/scenario/submit_rostering', methods=['POST'])
def submit_rostering():
    # 폼 데이터 → 쿼리 문자열
    params = request.form.to_dict()
    query_string = urlencode(params)

    # redirect to loading screen
    return redirect(url_for('scenario.rostering_loading') + '?' + query_string)

@scenario.route('/simulate/rostering_loading', methods=['GET'])
def rostering_loading():
    return render_template('loading.html', next_url=url_for('scenario.compare_rostering', **request.args))

@scenario.route('/scenario/compare_rostering', methods=['GET'])
def compare_rostering():
    doctor_roles = ['Intern', 'Resident', 'Specialist']
    nurse_roles = ['Junior', 'Senior']
    doctor_shifts = ['shift_1', 'shift_2']
    nurse_shifts = ['shift_1', 'shift_2', 'shift_3']

    # 쿼리 파라미터 → 리소스 구성
    resource_config = {}
    for role in doctor_roles:
        for shift in doctor_shifts:
            key = f"{role}_{shift}"
            resource_config[key] = int(request.args.get(key, 0))
    for role in nurse_roles:
        for shift in nurse_shifts:
            key = f"{role}_{shift}"
            resource_config[key] = int(request.args.get(key, 0))

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    # 수정 구성 기반 시뮬레이션
    sim = simulate_from_loader(**resource_config)
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    mod_kpis = formatter.format_result("Modified", mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        sim.resource_capacities
    )

    # 기본 시뮬레이션
    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    base_kpis = formatter.format_result("Base", base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    # KPI 비교
    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = base_dict.keys()
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in all_labels_set]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity="Rostering scenario with resource allocation",
                           zip=zip)

# -------------------------------------------
# varying the assignment of staff
# -------------------------------------------

@scenario.route('/simulate/reassign_staff', methods=['GET'])
def reassign_staff():
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    all_variants = df_date_re['variant'].tolist()
    activities = sorted(set(act for v in all_variants for act in v))
    activities = [act for act in activities]

    resource_pdf = pickle.load(open('app/data/activity_resource_detailed2.pickle', 'rb'))
    
    all_roles = set()
    for role_dict in resource_pdf.values():
        for r in role_dict.keys():
            if r.endswith("_pdf"):
                all_roles.add(r.replace("_pdf", ""))

    # Nurse/Doctor 계열 분리 (이름에 Nurse가 포함되어 있으면 Nurse 계열)
    doctor_candidates = sorted([r for r in all_roles if "Nurse" not in r])
    nurse_candidates = sorted([r for r in all_roles if "Nurse" in r])

    # 이제 위에서 구한 후보를 이용해서 role_map 구성
    role_map = {}
    for activity, roles in resource_pdf.items():
        # 각 계열별로 실제 0이 아닌 role이 있는지 체크
        doctor_nonzero = any(roles.get(f"{r}_pdf", 0) > 0 for r in doctor_candidates)
        nurse_nonzero = any(roles.get(f"{r}_pdf", 0) > 0 for r in nurse_candidates)

        selected_roles = []

        # Doctor 계열 처리
        if doctor_nonzero:
            selected_roles.extend(doctor_candidates)
        else:
            for r in doctor_candidates:
                if roles.get(f"{r}_pdf", 0) > 0:
                    selected_roles.append(r)

        # Nurse 계열 처리
        if nurse_nonzero:
            selected_roles.extend(nurse_candidates)
        else:
            for r in nurse_candidates:
                if roles.get(f"{r}_pdf", 0) > 0:
                    selected_roles.append(r)

        # 중복 제거
        role_map[activity] = list(dict.fromkeys(selected_roles))

    return render_template('reassign_staff.html', 
                           activities=activities, 
                           role_map=role_map, 
                           doctor_roles=doctor_candidates,
                           nurse_roles=nurse_candidates)

@scenario.route('/scenario/submit_reassign_staff', methods=['POST'])
def submit_reassign_staff():
    # 선택된 활동별 역할을 쿼리로 변환
    assignment_map = {}
    for key, value in request.form.items():
        if value:
            assignment_map[key] = value

    # 쿼리 스트링으로 encode
    query_string = urlencode(assignment_map)
    # 리디렉트 to loading 화면
    return redirect(url_for('scenario.reassign_staff_loading') + '?' + query_string)

@scenario.route('/scenario/reassign_staff_loading', methods=['GET'])
def reassign_staff_loading():
    return render_template('loading.html',
                           next_url=url_for('scenario.compare_reassign_staff', **request.args))

@scenario.route('/scenario/compare_reassign_staff', methods=['GET'])
def compare_reassign_staff():
    # 쿼리 스트링에서 역할 재할당 정보 추출
    assignment_map = request.args.to_dict()
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

    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = base_dict.keys()
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in all_labels_set]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity="Reassigned staff roles",
                           zip=zip)

# -------------------------------------------
# varying task composition
# -------------------------------------------

@scenario.route('/simulate/reassign_task_composition', methods=['GET'])
def reassign_task_composition():
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    all_variants = df_date_re['variant'].tolist()
    activities = sorted(set(act for v in all_variants for act in v))
    activities = [act for act in activities]

    resource_pdf = pickle.load(open('./data/activity_resource_detailed2.pickle', 'rb'))

    all_roles = set()
    for role_dict in resource_pdf.values():
        for r in role_dict.keys():
            if r.endswith("_pdf"):
                all_roles.add(r.replace("_pdf", ""))

    # Nurse/Doctor 계열 분리 (이름에 Nurse가 포함되어 있으면 Nurse 계열)
    doctor_candidates = sorted([r for r in all_roles if "Nurse" not in r])
    nurse_candidates = sorted([r for r in all_roles if "Nurse" in r])

    # 이제 위에서 구한 후보를 이용해서 role_map 구성
    role_map = {}
    for activity, roles in resource_pdf.items():
        # 각 계열별로 실제 0이 아닌 role이 있는지 체크
        doctor_nonzero = any(roles.get(f"{r}_pdf", 0) > 0 for r in doctor_candidates)
        nurse_nonzero = any(roles.get(f"{r}_pdf", 0) > 0 for r in nurse_candidates)

        selected_roles = []

        # Doctor 계열 처리
        if doctor_nonzero:
            selected_roles.extend(doctor_candidates)
        else:
            for r in doctor_candidates:
                if roles.get(f"{r}_pdf", 0) > 0:
                    selected_roles.append(r)

        # Nurse 계열 처리
        if nurse_nonzero:
            selected_roles.extend(nurse_candidates)
        else:
            for r in nurse_candidates:
                if roles.get(f"{r}_pdf", 0) > 0:
                    selected_roles.append(r)

        # 중복 제거
        role_map[activity] = list(dict.fromkeys(selected_roles))
    return render_template('reassign_task_composition.html', 
                           activities=activities, 
                           role_map=role_map, 
                           doctor_roles=doctor_candidates,
                           nurse_roles=nurse_candidates)

@scenario.route('/scenario/submit_reassign_task', methods=['POST'])
def submit_reassign_task():
    # 전달받은 form 데이터를 그대로 redirect에 사용
    args = {key: val for key, val in request.form.items() if val.strip() != ''}
    return redirect(url_for('scenario.loading_reassign_task', **args))

@scenario.route('/simulate/loading_reassign_task', methods=['GET'])
def loading_reassign_task():
    return render_template('loading.html', next_url=url_for('scenario.compare_reassign_task', **request.args))

@scenario.route('/scenario/compare_reassign_task', methods=['GET'])
def compare_reassign_task():
    resource_pdf = pickle.load(open('app/data/activity_resource_detailed2.pickle', 'rb'))

    all_roles = ['Intern', 'Resident', 'Specialist', 'Junior_Nurse', 'Senior_Nurse']

    # 입력값 기반으로 업데이트
    for activity in resource_pdf.keys():
        updated_values = {}
        for r in all_roles:
            field = f"{activity}_{r}"
            if field in request.args and request.args[field] != '':
                updated_values[r + '_pdf'] = float(request.args[field])
        if updated_values:
            for k, v in updated_values.items():
                resource_pdf[activity][k] = v

    sim = simulate_from_loader(task_composition=resource_pdf)
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

    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    all_labels_set = base_dict.keys()
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in all_labels_set]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity="Varying task composition",
                           zip=zip)

# ---------------------------------------------
# merging roles and changing shift times
# ---------------------------------------------

@scenario.route('/simulate/merge_shift_roles', methods=['GET'])
def merge_shift_roles():
    return render_template("merge_shift_roles.html")

@scenario.route('/scenario/submit_merge_roles', methods=['POST'])
def submit_merge_roles():
    merged_roles = {}
    if request.form.get('merge_doctor'):
        merged_roles.update({'Intern': 'Junior Doctor', 'Resident': 'Junior Doctor'})
    if request.form.get('merge_nurse'):
        merged_roles.update({'Junior': 'Nurse', 'Senior': 'Nurse'})

    shift_data = {}
    for role in ['intern', 'resident', 'specialist']:
        for shift in ['shift_1', 'shift_2']:
            shift_data[f'{role}_{shift}_start'] = request.form.get(f'{role}_{shift}_start', '')
            shift_data[f'{role}_{shift}_end'] = request.form.get(f'{role}_{shift}_end', '')
    for role in ['junior', 'senior']:
        for shift in ['shift_1', 'shift_2', 'shift_3']:
            shift_data[f'{role}_{shift}_start'] = request.form.get(f'{role}_{shift}_start', '')
            shift_data[f'{role}_{shift}_end'] = request.form.get(f'{role}_{shift}_end', '')

    # 병합 정보도 redirect에 포함
    return redirect(url_for('scenario.merge_roles_loading', **merged_roles, **shift_data))

@scenario.route('/simulate/merge_roles_loading', methods=['GET'])
def merge_roles_loading():
    return render_template('loading.html', next_url=url_for('scenario.compare_merge_roles', **request.args))

@scenario.route('/scenario/compare_merge_roles', methods=['GET'])
def compare_merge_roles():
    args = request.args
    merged_roles = {}
    if args.get('Intern') == 'Junior Doctor':
        merged_roles['Intern'] = 'Junior Doctor'
    if args.get('Resident') == 'Junior Doctor':
        merged_roles['Resident'] = 'Junior Doctor'
    if args.get('Junior') == 'Nurse':
        merged_roles['Junior'] = 'Nurse'
    if args.get('Senior') == 'Nurse':
        merged_roles['Senior'] = 'Nurse'

    base_shift_times = {
        'intern_shift_1': (8, 20), 'intern_shift_2': (20, 8),
        'resident_shift_1': (8, 20), 'resident_shift_2': (20, 8),
        'specialist_shift_1': (8, 20), 'specialist_shift_2': (20, 8),
        'junior_shift_1': (6, 14), 'junior_shift_2': (14, 22), 'junior_shift_3': (22, 6),
        'senior_shift_1': (6, 14), 'senior_shift_2': (14, 22), 'senior_shift_3': (22, 6),
    }

    shift_time_config = {}
    shift_time_modified = False
    for k in base_shift_times.keys():
        start = args.get(f'{k}_start')
        end = args.get(f'{k}_end')
        if start and end and start.strip() != '' and end.strip() != '':
            shift_time_config[k] = (int(start), int(end))
            shift_time_modified = True
        else:
            shift_time_config[k] = base_shift_times[k]
    if not shift_time_modified:
        shift_time_config = None

    # 시뮬레이션 실행
    sim = simulate_from_loader(role_merge=merged_roles, shift_times=shift_time_config)
    mod_df = pd.DataFrame(sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

    base_sim = simulate_from_loader()
    base_df = pd.DataFrame(base_sim.results)
    base_df['Activity'] = base_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    mod_args = {
        "file_name": "Modified",
        "result": mod_df,
        "median_dtp_res": mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        "mean_dtp_res": mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        "resource_capacities": sim.resource_capacities,
    }
    base_args = {
        "file_name": "Base",
        "result": base_df,
        "median_dtp_res": base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        "mean_dtp_res": base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        "resource_capacities": base_sim.resource_capacities,
    }

    if merged_roles and shift_time_config:
        mod_args.update({"role_merge": merged_roles, "shift_times": shift_time_config})
        base_args.update({"shift_times": base_shift_times})
    elif merged_roles:
        mod_args.update({"role_merge": merged_roles})
    elif shift_time_config:
        mod_args.update({"shift_times": shift_time_config})
        base_args.update({"shift_times": base_shift_times})

    mod_kpis = formatter.format_result(**mod_args)
    base_kpis = formatter.format_result(**base_args)

    base_dict = dict(base_kpis)
    mod_dict = dict(mod_kpis)
    seen = set()
    all_labels_set = []
    for label, *_ in base_kpis + mod_kpis:
        if label not in seen:
            seen.add(label)
            all_labels_set.append(label)

    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in all_labels_set]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity="Merged roles and customized shift times",
                           zip=zip)

# -------------------------------------------
# adding new activities
# -------------------------------------------
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
    args = request.form.to_dict(flat=True)
    selected_indices = request.form.getlist('selected_variants')
    for idx in selected_indices:
        args[f'selected_{idx}'] = '1'
    args['selected_indices'] = ','.join(selected_indices)
    return redirect(url_for('scenario.add_activity_loading', **args))

@scenario.route('/simulate/add_activity_loading', methods=['GET'])
def add_activity_loading():
    return render_template('loading.html', next_url=url_for('scenario.compare_add_activity', **request.args))

@scenario.route('/scenario/compare_add_activity', methods=['GET'])
def compare_add_activity():
    args = request.args
    new_activity = args.get('activity_name')
    selected_indices = list(map(int, args.get('selected_indices', '').split(',')))

    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    original_variants = df_date_re['variant']
    new_variants = [list(v) for v in original_variants]

    for idx in selected_indices:
        original_variant = list(original_variants[idx])
        pos_val = args.get(f'insert_pos_{idx}', '').strip()
        if pos_val == '':
            modified_variant = original_variant + [new_activity]
        else:
            insert_pos = max(0, min(len(original_variant), int(pos_val)))
            modified_variant = original_variant[:insert_pos] + [new_activity] + original_variant[insert_pos:]
        new_variants[idx] = modified_variant

    def safe_float(val, default=0.0):
        try: return float(val)
        except: return default

    duration_values = list(map(int, args.get('duration_values').split(',')))
    duration_probs = list(map(float, args.get('duration_probs').split(',')))
    role_names = ['Intern', 'Resident', 'Specialist', 'Junior_Nurse', 'Senior_Nurse']
    role_probs = [safe_float(args.get(f'role_probs_{r}')) for r in role_names]

    prev_activity = args.get('prev_activity')
    next_activity = args.get('next_activity')
    trans_time_before = safe_float(args.get('trans_time_before'))
    trans_time_after = safe_float(args.get('trans_time_after'))

    new_event_duration = {
        new_activity: {
            'value': duration_values,
            'pdf': [p/sum(duration_probs) for p in duration_probs]
        }
    }
    new_resource_pdf = {
        new_activity: {
            f"{role}_pdf": p for role, p in zip(role_names, role_probs)
        }
    }
    new_trans_time = {}
    if prev_activity:
        new_trans_time[(prev_activity, new_activity)] = {'trans_time': trans_time_before}
    if next_activity:
        new_trans_time[(new_activity, next_activity)] = {'trans_time': trans_time_after}

    sim = simulate_from_loader(
        custom_variants=new_variants,
        custom_cases=None,
        custom_arrival_times=None,
        event_duration=new_event_duration,
        resource_pdf=new_resource_pdf,
        trans_time=new_trans_time
    )

    def safe_diff(x, i1, i2):
        try: return x.iloc[i1] - x.iloc[i2]
        except: return pd.Timedelta(0)

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
    all_labels = base_dict.keys()
    comparison_kpis = [(label, base_dict.get(label, ''), mod_dict.get(label, '')) for label in all_labels]

    return render_template('comparison2.html',
                           comparison_kpis=comparison_kpis,
                           activity=f"Added '{new_activity}' to {len(selected_indices)} variants",
                           zip=zip)

# -------------------------------------------
# reducing_lab_boarding
# -------------------------------------------

@scenario.route('/simulate/reduce_lab_boarding', methods=['GET'])
def reduce_lab_boarding():
    df = pd.read_csv('data/transition_time_generated.csv')
    transition_df = df.dropna()
    transition_time_map = {}
    for _, row in transition_df.iterrows():
        prev, next_, time = row['new_activity'], row['next_act'], row['trans_time']
        if prev not in transition_time_map:
            transition_time_map[prev] = {}
        transition_time_map[prev][next_] = time

    return render_template('reducing_lab_boarding.html',
                           activities=sorted(transition_time_map.keys()),
                           transition_time_map=transition_time_map)

@scenario.route('/scenario/submit_reduce_lab_boarding', methods=['POST'])
def submit_reduce_lab_boarding():
    from_act = request.form.get('prev_activity')
    to_act = request.form.get('next_activity')
    new_time = request.form.get('new_trans_time')

    return redirect(url_for('scenario.loading_reducing_lab_boarding_simulation',
                            from_act=from_act,
                            to_act=to_act,
                            new_time=new_time))

@scenario.route('/scenario/loading_reducing_lab_boarding_simulation')
def loading_reducing_lab_boarding_simulation():
    next_url = url_for('scenario.run_reduce_lab_boarding_simulation',
                       from_act=request.args.get('from_act'),
                       to_act=request.args.get('to_act'),
                       new_time=request.args.get('new_time'))

    return render_template('loading.html', next_url=next_url)

@scenario.route('/scenario/run_reduce_lab_boarding_simulation')
def run_reduce_lab_boarding_simulation():
    from_act = request.args.get('from_act')
    to_act = request.args.get('to_act')
    new_time = request.args.get('new_time')

    try:
        new_time = int(new_time)
    except (ValueError, TypeError):
        new_time = None

    modified_transitions = {}
    if from_act and to_act and new_time is not None:
        modified_transitions[(from_act, to_act)] = {'trans_time': new_time}

    # 시뮬레이션 실행
    sim = simulate_from_loader(modified_transition_time=modified_transitions)
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
        activity="Reducing Lab & Boarding Times",
        zip=zip
    )

# -------------------------------------------
# varying patient demand
# -------------------------------------------

@scenario.route('/simulate/vary_patient_demand', methods=['GET'])
def vary_patient_demand():
    # GET으로 화면 표시
    return render_template('varying_patient_demand.html')

@scenario.route('/scenario/submit_vary_demand', methods=['POST'])
def submit_vary_demand():
    multiplier = int(request.form.get('demand_multiplier', 100))
    return redirect(url_for('scenario.vary_demand_loading', multiplier=multiplier))

@scenario.route('/simulate/vary_demand_loading', methods=['GET'])
def vary_demand_loading():
    return render_template('loading.html', next_url=url_for('scenario.compare_vary_demand', **request.args))

@scenario.route('/scenario/compare_vary_demand', methods=['GET'])
def compare_vary_demand():
    multiplier = int(request.args.get('multiplier', 100))

    def safe_diff(x, i1, i2):
        try:
            return x.iloc[i1] - x.iloc[i2]
        except:
            return pd.Timedelta(0)

    # Baseline
    base_sim = simulate_from_loader()
    base_cases = base_sim.cases
    base_arrivals = base_sim.arrival_times
    base_df = pd.DataFrame(base_sim.results)
    base_kpis = formatter.format_result(
        "Base",
        base_df,
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        base_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        base_sim.resource_capacities
    )

    # Varying Demand
    if multiplier == 100:
        custom_cases = base_cases
        custom_arrivals = base_arrivals
        custom_variants = base_sim.variants
    else:
        ratio = multiplier / 100.0
        new_count = int(len(base_cases) * ratio)
        custom_cases = (base_cases * ((new_count // len(base_cases)) + 1))[:new_count]
        custom_arrivals = (base_arrivals * ((new_count // len(base_arrivals)) + 1))[:new_count]
        custom_variants = (base_sim.variants * ((new_count // len(base_sim.variants)) + 1))[:new_count]

    mod_sim = simulate_from_loader(custom_cases=custom_cases, custom_arrival_times=custom_arrivals, custom_variants=custom_variants)
    mod_df = pd.DataFrame(mod_sim.results)
    mod_df['Activity'] = mod_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

    mod_kpis = formatter.format_result(
        "Demand Modified",
        mod_df,
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
        mod_df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
        mod_sim.resource_capacities
    )

    comparison_kpis = [
        (label, dict(base_kpis).get(label, ''), dict(mod_kpis).get(label, ''))
        for label in dict(base_kpis).keys()
    ]

    return render_template(
        'comparison2.html',
        comparison_kpis=comparison_kpis,
        activity=f"Varying Patient Demand ({multiplier}%)",
        zip=zip
    )

# -------------------------------------------
# varying queue discipline
# -------------------------------------------

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

    args = {'discipline': discipline}

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

        for idx, variant in enumerate(unique_variants):
            key = f'priority_{idx}'
            val = request.form.get(key, '').strip()
            args[f'priority_{idx}'] = val
            args[f'variant_{idx}'] = str(tuple(variant))

    return redirect(url_for('scenario.vary_queue_loading', **args))

@scenario.route('/simulate/vary_queue_loading', methods=['GET'])
def vary_queue_loading():
    return render_template(
        'loading.html',
        next_url=url_for('scenario.compare_vary_queue', **request.args)
    )

@scenario.route('/scenario/compare_vary_queue', methods=['GET'])
def compare_vary_queue():
    from ast import literal_eval

    discipline = request.args.get('discipline', 'FIFO')
    variant_priority_map = None

    if discipline == 'PRIORITY':
        variant_priority_map = {}
        for key, val in request.args.items():
            if key.startswith('variant_'):
                idx = key.split('_')[1]
                variant = literal_eval(val)
                priority = int(request.args.get(f'priority_{idx}', '5'))
                variant_priority_map[variant] = priority

    sim = simulate_from_loader(
        queue_discipline=discipline,
        variant_priority_map=variant_priority_map
    )

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
        activity=f"Varying Queue Discipline (Mode: {discipline})",
        zip=zip
    )


# -------------------------------------------
# varying patient sequences
# -------------------------------------------

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
        if new_val:
            modified_map[json.dumps(uv)] = new_val

    return redirect(url_for('scenario.vary_sequence_loading', **modified_map))

@scenario.route('/simulate/vary_sequence_loading', methods=['GET'])
def vary_sequence_loading():
    return render_template(
        'loading.html',
        next_url=url_for('scenario.compare_vary_sequence', **request.args)
    )

@scenario.route('/scenario/compare_vary_sequence', methods=['GET'])
def compare_vary_sequence():
    from ast import literal_eval

    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    base_variants = list(df_date_re['variant'])

    seen = set()
    unique_variants = []
    for v in base_variants:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_variants.append(v)

    modified_map = {
        tuple(literal_eval(k)): [x.strip() for x in v.split(',')]
        for k, v in request.args.items()
    }

    new_variants = []
    for v in base_variants:
        t = tuple(v)
        if t in modified_map:
            new_variants.append(modified_map[t])
        else:
            new_variants.append(v)

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


# -------------------------------------------
# resequencing activities
# -------------------------------------------

@scenario.route('/scenario/vary_sequence2', methods=['GET', 'POST'])
def vary_sequence2():
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    all_variants = df_date_re['variant'].tolist()
    all_activities = sorted(set(act for v in all_variants for act in v))

    variants_for_drag = []  # 기본값: drag&drop UI는 안 보이도록
    selected_activity = None
    top_n = 5

    total_count = len(all_variants) # 전체 고유 variant 개수
    shown_count = 0
    shown_percentage = 0.0

    if request.method == 'POST':
        # 사용자로부터 상위 개수(top_n) 선택 받기
        try:
            top_n = int(request.form.get('top_n', 5))
            if top_n <= 0:
                top_n = 5
        except ValueError:
            top_n = 5

        selected_activity = request.form.get('selected_activity')
        if selected_activity:
            # 선택한 activity를 포함하는 variants 필터링
            from collections import Counter
            filtered = [tuple(v) for v in all_variants if selected_activity in v]
            freq = Counter(filtered)
            sorted_variants = sorted(freq.items(), key=lambda x: x[1], reverse=True)

            shown_variants = sorted_variants[:top_n]
            shown_count = sum(count for _, count in shown_variants)
            shown_percentage = round((shown_count / total_count) * 100, 1) if total_count > 0 else 0.0

            # drag&drop으로 보여줄 variant 후보들 생성 (빈도순)
            variants_for_drag = [
                {
                    'index': idx,
                    'variant': ' > '.join(var),
                    'activities': list(var),
                    'count': count
                }
                for idx, (var, count) in enumerate(sorted_variants[:top_n]) 
            ]

    return render_template(
        'vary_sequence.html',
        activities=all_activities,
        variants=variants_for_drag,
        selected_activity=selected_activity,
        selected_top_n=top_n,
        shown_count=shown_count,
        total_count=total_count,
        shown_percentage=shown_percentage
    )

@scenario.route('/scenario/submit_vary_sequence2', methods=['POST'])
def submit_vary_sequence2():
    query_args = {}
    top_n = int(request.form.get('top_n', 5))
    for i in range(top_n):
        val = request.form.get(f'new_order_{i}', '')
        if val.strip():
            query_args[f'new_order_{i}'] = val
    return redirect(url_for('scenario.loading_vary_sequence', **query_args))

@scenario.route('/simulate/loading_vary_sequence', methods=['GET'])
def loading_vary_sequence():
    return render_template('loading.html', next_url=url_for('scenario.compare_vary_sequence2', **request.args))

@scenario.route('/scenario/compare_vary_sequence2', methods=['GET'])
def compare_vary_sequence2():
    df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
    all_variants = df_date_re['variant'].tolist()

    top_n = int(request.args.get('top_n', 5))

    modified_orders = []
    for idx in range(top_n):
        order_str = request.args.get(f'new_order_{idx}', '')
        if order_str.strip():
            modified_orders.append([a.strip() for a in order_str.split(',') if a.strip()])
        else:
            modified_orders.append(None)

    seen = set()
    unique_variants = []
    for v in all_variants:
        t = tuple(v)
        if t not in seen:
            seen.add(t)
            unique_variants.append(v)
    topn_variants = unique_variants[:top_n]

    updated_variants = []
    for v in all_variants:
        matched_index = None
        for idx, base in enumerate(topn_variants):
            if v == base:
                matched_index = idx
                break

        if matched_index is not None and modified_orders[matched_index] is not None:
            new_order = modified_orders[matched_index]
            selected = [act for act in new_order if act in v]
            remaining = [act for act in v if act not in new_order]
            updated_variants.append(selected + remaining)
        else:
            updated_variants.append(v)

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
        print(request.form)
        # 1. 선택한 activity
        selected_activity = request.form.get('selected_activity')

        # 2. 변경된 순서들
        new_orders = {}
        for key in request.form:
            if key.startswith('new_order_'):
                idx = key[len('new_order_'):]
                order = request.form.get(key)
                if order:
                    new_orders[int(idx)] = [act.strip() for act in order.split(',')]

        # Stack에 저장할 파라미터 구성
        params['selected_activity'] = selected_activity
        params['new_orders'] = new_orders

    elif scenario_id in ('reallocating', 'adding_staff'):
        # Intern
        params['intern_shift_1'] = int(request.form.get('intern_shift_1', 0))
        params['intern_shift_2'] = int(request.form.get('intern_shift_2', 0))
        # Resident
        params['resident_shift_1'] = int(request.form.get('resident_shift_1', 0))
        params['resident_shift_2'] = int(request.form.get('resident_shift_2', 0))
        # Specialist
        params['specialist_shift_1'] = int(request.form.get('specialist_shift_1', 0))
        params['specialist_shift_2'] = int(request.form.get('specialist_shift_2', 0))
        # Junior Nurse
        params['junior_shift_1'] = int(request.form.get('junior_shift_1', 0))
        params['junior_shift_2'] = int(request.form.get('junior_shift_2', 0))
        params['junior_shift_3'] = int(request.form.get('junior_shift_3', 0))
        # Senior Nurse
        params['senior_shift_1'] = int(request.form.get('senior_shift_1', 0))
        params['senior_shift_2'] = int(request.form.get('senior_shift_2', 0))
        params['senior_shift_3'] = int(request.form.get('senior_shift_3', 0))
    elif scenario_id == 'rostering':
        print(request.form)
        # Intern
        params['intern_shift_1'] = int(request.form.get('Intern_shift_1', 0))
        params['intern_shift_2'] = int(request.form.get('Intern_shift_2', 0))
        # Resident
        params['resident_shift_1'] = int(request.form.get('Resident_shift_1', 0))
        params['resident_shift_2'] = int(request.form.get('Resident_shift_2', 0))
        # Specialist
        params['specialist_shift_1'] = int(request.form.get('Specialist_shift_1', 0))
        params['specialist_shift_2'] = int(request.form.get('Specialist_shift_2', 0))
        # Junior Nurse
        params['junior_shift_1'] = int(request.form.get('Junior_shift_1', 0))
        params['junior_shift_2'] = int(request.form.get('Junior_shift_2', 0))
        params['junior_shift_3'] = int(request.form.get('Junior_shift_3', 0))
        # Senior Nurse
        params['senior_shift_1'] = int(request.form.get('Senior_shift_1', 0))
        params['senior_shift_2'] = int(request.form.get('Senior_shift_2', 0))
        params['senior_shift_3'] = int(request.form.get('Senior_shift_3', 0))
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
        print(request.form)
        # 1. 역할 병합 설정
        merged_roles = {}
        if request.form.get('merge_doctor'):
            merged_roles.update({'Intern': 'Junior Doctor', 'Resident': 'Junior Doctor'})
        if request.form.get('merge_nurse'):
            merged_roles.update({'Junior': 'Nurse', 'Senior': 'Nurse'})

        # 2. 시간 설정 수집 (flat 형태로)
        shift_prefixes = [
            'intern_shift_1', 'intern_shift_2',
            'resident_shift_1', 'resident_shift_2',
            'specialist_shift_1', 'specialist_shift_2',
            'junior_shift_1', 'junior_shift_2', 'junior_shift_3',
            'senior_shift_1', 'senior_shift_2', 'senior_shift_3'
        ]
        shift_times = {}
        for prefix in shift_prefixes:
            start = request.form.get(f'{prefix}_start')
            end = request.form.get(f'{prefix}_end')
            if start and end:
                shift_times[prefix] = (int(start), int(end))

        # 3. params로 저장
        params['role_merge'] = merged_roles
        params['shift_times'] = shift_times

    elif scenario_id == 'add_new_activity':
        new_activity = request.form.get('activity_name')
        selected_indices = [int(i) for i in request.form.getlist('selected_variants')]

        insert_positions = {}
        for idx in selected_indices:
            pos_key = f'insert_pos_{idx}'
            pos_val = request.form.get(pos_key, '').strip()
            insert_positions[idx] = None if pos_val == '' else int(pos_val)

        # Duration values and probs
        duration_values = [int(v) for v in request.form.getlist('duration_values') if v.strip() != '']
        duration_probs = [float(p) for p in request.form.getlist('duration_probs') if p.strip() != '']

        # Role assignment probabilities
        roles = ['Intern', 'Resident', 'Specialist', 'Junior_Nurse', 'Senior_Nurse']
        role_probs = {}
        for role in roles:
            val = request.form.get(f'role_probs_{role}', '').strip()
            if val != '':
                role_probs[role] = float(val)

        # Transition time
        trans_time_before = request.form.get('trans_time_before')
        trans_time_after = request.form.get('trans_time_after')
        trans_time_before = float(trans_time_before) if trans_time_before and trans_time_before.strip() != '' else None
        trans_time_after = float(trans_time_after) if trans_time_after and trans_time_after.strip() != '' else None

        # Neighboring activities
        prev_activity = request.form.get('prev_activity', '').strip() or None
        next_activity = request.form.get('next_activity', '').strip() or None

        # Save to params
        params['new_activity'] = new_activity
        params['selected_indices'] = selected_indices
        params['insert_positions'] = insert_positions
        params['duration_values'] = duration_values
        params['duration_probs'] = duration_probs
        params['role_probs'] = role_probs
        params['trans_time_before'] = trans_time_before
        params['trans_time_after'] = trans_time_after
        params['prev_activity'] = prev_activity
        params['next_activity'] = next_activity


    elif scenario_id == 'reduce_time':
        prev_activity = request.form.get('prev_activity')
        next_activity = request.form.get('next_activity')
        new_trans_time = request.form.get('new_trans_time')

        params['prev_activity'] = prev_activity
        params['next_activity'] = next_activity
        try:
            params['new_trans_time'] = float(new_trans_time)
        except:
            params['new_trans_time'] = None  # 혹은 기본값 설정

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

@scenario.route('/scenario/run_stack_loading', methods=['GET'])
def run_stack_loading():
    return render_template('loading2.html', redirect_url=url_for('scenario.run_stack'))

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

    staff_keys = [
        'Intern_shift_1','Intern_shift_2',
        'Resident_shift_1','Resident_shift_2',
        'Specialist_shift_1','Specialist_shift_2',
        'Junior_shift_1','Junior_shift_2','Junior_shift_3',
        'Senior_shift_1','Senior_shift_2','Senior_shift_3',
    ]
    current_staff_overrides: dict[str, int] = {}
    current_role_merge: dict[str, str] = {}
    current_shift_times: dict[str, tuple[int, int]] | None = None
    
    def _extract_staff_overrides(params: dict) -> dict[str, int]:
        """
        form/stack에 들어온 키가 intern_shift_1 또는 Intern_shift_1 둘 다 오더라도
        simulate_from_loader가 기대하는 TitleCase 키로 통일해서 반환.
        """
        result = {}
        roles = [
            ('intern',     [1, 2]),
            ('resident',   [1, 2]),
            ('specialist', [1, 2]),
            ('junior',     [1, 2, 3]),
            ('senior',     [1, 2, 3]),
        ]
        for role, shifts in roles:
            for s in shifts:
                lower = f'{role}_shift_{s}'
                title = f'{role.capitalize()}_shift_{s}'
                key_present = None
                if lower in params: key_present = lower
                if title in params: key_present = title if key_present is None else key_present
                if key_present is not None:
                    try:
                        val = int(params[key_present])
                    except (ValueError, TypeError):
                        continue
                    result[title] = val   # 반드시 TitleCase로 보관
        return result

    def call_sim_with_overrides(**kwargs):
        """항상 누적된 overrides를 함께 전달"""
        base_kwargs = dict(
            custom_variants=current_variants,
            custom_cases=current_cases,
            custom_arrival_times=current_arrivals,
            **current_staff_overrides
        )
        # role_merge/shift_times는 설정된 경우에만 전달
        if current_role_merge:
            base_kwargs['role_merge'] = current_role_merge
        if current_shift_times:
            base_kwargs['shift_times'] = current_shift_times
        base_kwargs.update(kwargs)
        return simulate_from_loader(**base_kwargs)


    step_names = ["Base"]
    kpi_snapshots = [dict(base_kpis)]

    for step in stack:
        stype = step['type']
        params = step.get('params', {})
        sim = None

        if stype == 'remove_activity':
            sim = call_sim_with_overrides(remove_activity=params.get('activity'))

        elif stype == 'resequence_activity':
            print(step['params'])
            # 1. 전체 variant 불러오기
            df_date_re = pickle.load(open('app/data/df_date_re.pkl', 'rb'))
            all_variants = df_date_re['variant'].tolist()

            # 2. Stack에서 전달된 파라미터 꺼내기
            new_orders = step['params'].get('new_orders', {})
            top_n = len(new_orders)

            # 3. 전체 variant 중 상위 top_n unique variant 추출
            seen = set()
            unique_variants = []
            for v in all_variants:
                t = tuple(v)
                if t not in seen:
                    seen.add(t)
                    unique_variants.append(v)
            topn_variants = unique_variants[:top_n]

            # 4. 사용자가 지정한 순서대로 variant 재정렬
            updated_variants = []
            for v in all_variants:
                matched_index = None
                for idx, base in enumerate(topn_variants):
                    if v == base:
                        matched_index = idx
                        break

                if matched_index is not None and new_orders.get(matched_index) is not None:
                    new_order = new_orders[matched_index]
                    selected = [act for act in new_order if act in v]
                    remaining = [act for act in v if act not in new_order]
                    updated_variants.append(selected + remaining)
                else:
                    updated_variants.append(v)

            # 시뮬레이션 실행 (누적된 current_cases / current_arrivals 사용)
            sim = call_sim_with_overrides(custom_variants=updated_variants)

        elif stype in ('reallocating', 'adding_staff', 'rostering'):
            step_staff = _extract_staff_overrides(params)
            if step_staff:
                current_staff_overrides.update(step_staff)
            print(current_staff_overrides)
            sim = call_sim_with_overrides()

        elif stype == 'vary_the_assignment':
            sim = call_sim_with_overrides(role_override=params.get('role_override', {}))

        elif stype == 'vary_task_comp':
            sim = call_sim_with_overrides(task_composition=params.get('task_composition', {}))

        elif stype == 'merg_role':
            merged_roles = step['params'].get('role_merge', {})

            # shift_times가 None이면 기본값 사용
            default_shift_times = {
                'intern_shift_1': (8, 20), 'intern_shift_2': (20, 8),
                'resident_shift_1': (8, 20), 'resident_shift_2': (20, 8),
                'specialist_shift_1': (8, 20), 'specialist_shift_2': (20, 8),
                'junior_shift_1': (6, 14), 'junior_shift_2': (14, 22), 'junior_shift_3': (22, 6),
                'senior_shift_1': (6, 14), 'senior_shift_2': (14, 22), 'senior_shift_3': (22, 6),
            }

            shift_times = step['params'].get('shift_times')
            if not shift_times:
                shift_times = default_shift_times

            # 시간 문자열일 경우 정수 튜플로 변환 (예: "8,20" → (8, 20))
            parsed_shift_times = {
                k: (parse_shift(v[0]), parse_shift(v[1])) if isinstance(v, tuple) else (parse_shift(v.split(',')[0]), parse_shift(v.split(',')[1]))
                for k, v in shift_times.items()
            }

            # sim = simulate_from_loader(role_merge=merged_roles, shift_times=parsed_shift_times)
            sim = call_sim_with_overrides()

        elif stype == 'add_new_activity':

            new_variants = [list(v) for v in current_variants]
            new_act = step['params'].get('new_activity')
            indices = step['params'].get('selected_indices', [])
            positions = step['params'].get('insert_positions', {})

            for idx in indices:
                if idx < 0 or idx >= len(new_variants):
                    continue
                orig = list(new_variants[idx])
                pos_val = positions.get(str(idx)) or positions.get(idx) or ''
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

            # duration
            duration_values = step['params'].get('duration_values', [])
            duration_probs = step['params'].get('duration_probs', [])
            event_duration = {}
            if duration_values and duration_probs:
                try:
                    norm_probs = [p / sum(duration_probs) for p in duration_probs]
                    event_duration = {
                        new_act: {
                            'value': duration_values,
                            'pdf': norm_probs
                        }
                    }
                except:
                    pass

            # role probs
            role_keys = ['Intern', 'Resident', 'Specialist', 'Junior_Nurse', 'Senior_Nurse']
            raw_role_probs = step['params'].get('role_probs', {})
            resource_pdf = {
                new_act: {
                    f"{role}_pdf": raw_role_probs.get(role, 0.0) for role in role_keys
                }
            }

            # transition time
            trans_time = {}
            before = step['params'].get('trans_time_before')
            after = step['params'].get('trans_time_after')
            prev = step['params'].get('prev_activity')
            next_ = step['params'].get('next_activity')
            if prev and before is not None:
                trans_time[(prev, new_act)] = {'trans_time': before}
            if next_ and after is not None:
                trans_time[(new_act, next_)] = {'trans_time': after}

            sim = simulate_from_loader(
                custom_variants=new_variants,
                custom_cases=current_cases,
                custom_arrival_times=current_arrivals,
                event_duration=event_duration,
                resource_pdf=resource_pdf,
                trans_time=trans_time
            )

        elif stype == 'reduce_time':
            prev = step['params'].get('prev_activity')
            next_ = step['params'].get('next_activity')
            new_time = step['params'].get('new_trans_time')

            # 유효성 검사 및 딕셔너리 구성
            modified_transitions = {}
            if prev and next_ and new_time is not None:
                try:
                    new_time = int(new_time)
                    modified_transitions[(prev, next_)] = {'trans_time': new_time}
                except ValueError:
                    pass

            sim = call_sim_with_overrides(trans_time=modified_transitions)

        elif stype == 'vary_patient_demand':
            multiplier = step['params'].get('demand_multiplier', 100)
            custom_cases, custom_arrivals, custom_variants = apply_multiplier(current_cases, current_arrivals, current_variants, multiplier)
            sim = call_sim_with_overrides(custom_cases=custom_cases, custom_arrival_times=custom_arrivals, custom_variants=custom_variants)

        elif stype == 'vary_patient_sequences':
            raw_map = step['params'].get('modified_map', {})
            new_variants = []
            for v in current_variants:
                t = ','.join(v)
                if t in raw_map:
                    new_variants.append(raw_map[t])
                else:
                    new_variants.append(v)
            sim = call_sim_with_overrides(custom_variants=new_variants)

        elif stype == 'vary_queue_discipline':
            discipline = step['params'].get('queue_discipline', 'FIFO')
            v_map_raw = step['params'].get('variant_priority_map', None)
            variant_priority_map = None
            if v_map_raw:
                variant_priority_map = {tuple(k.split(',')): v for k, v in v_map_raw.items()}
            sim = call_sim_with_overrides(queue_discipline=discipline, variant_priority_map=variant_priority_map)

        if sim:
            print(step['params'])
            current_variants = sim.variants
            current_cases = sim.cases
            current_arrivals = sim.arrival_times

            df = pd.DataFrame(sim.results)
            df['Activity'] = df['Activity'].str.replace(r'(_\d+)$', '', regex=True)

            kpis = formatter.format_result(
                stype,
                df,
                df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).median(),
                df.groupby('Patient_id')['Start_time'].apply(lambda x: safe_diff(x, 2, 1)).mean(),
                sim.resource_capacities,
                role_merge=step.get('params', {}).get('role_merge'),
                shift_times=step.get('params', {}).get('shift_times')
            )

            step_names.append(stype)
            kpi_snapshots.append(dict(kpis))

    preferred_order = [
        # Doctor Shift
        "Number of Intern Shift 1", "Number of Intern Shift 2",
        "Number of Resident Shift 1", "Number of Resident Shift 2",
        "Number of Specialist Shift 1", "Number of Specialist Shift 2",

        # Nurse Shift
        "Number of Junior Shift 1",
        "Number of Junior Shift 2",
        "Number of Junior Shift 3",
        "Number of Senior Shift 1",
        "Number of Senior Shift 2",
        "Number of Senior Shift 3",

        # Waiting Time
        "Median Waiting Time (minutes)",
        "Mean Waiting Time (minutes)",

        # Duration Time
        "Case Duration (max)",
        "Case Duration (median)",
        "Case Duration (mean)"
    ]

    section_headers = {
        "Number of Intern Shift 1": "Doctor Shift",
        "Number of Junior Shift 1": "Nurse Shift",
        "Median Waiting Time (minutes)": "Waiting Time",
        "Case Duration (max)": "Duration Time"
    }

    base_map = kpi_snapshots[0]  # "Base"가 0번째

    comparison_kpis = []
    prev_section = None

    for label in preferred_order:
        current_section = section_headers.get(label)
        if current_section and current_section != prev_section:
            comparison_kpis.append([f"<b>{current_section}</b>"] + [""] * len(kpi_snapshots))
            prev_section = current_section

        row = [label]

        for i, snap in enumerate(kpi_snapshots):
            val = snap.get(label, '')
            cell = str(val)

            if i == 0:
                row.append(cell)
            else:
                prev_val = kpi_snapshots[i-1].get(label, '')
                prev_num = _to_number_or_minutes(prev_val)
                curr_num = _to_number_or_minutes(val)

                delta_txt = ""
                if (prev_num is not None) and (curr_num is not None):
                    delta = curr_num - prev_num
                    pct = (delta / prev_num * 100.0) if prev_num not in (0, None) else None
                    delta_txt = f" <small class='delta'>{_fmt_delta(delta, pct)}</small>"

                row.append(cell + delta_txt)

        comparison_kpis.append(row)

    return render_template(
        'comparison.html',
        comparison_kpis=comparison_kpis,
        step_names=step_names,
        activity="Stacked Scenarios",
        zip=zip
    )