from flask import Blueprint, render_template, request, flash, redirect, url_for
from app.simulator.loader import prepare_simulation_data
from app.simulator.engine import Simulator, Timer
from app.simulator.formatter import format_result
import pandas as pd
import numpy as np
import json
from collections import Counter
import pickle
import os

main = Blueprint('main', __name__)

UPLOAD_FOLDER = "./data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@main.route('/')
def index():
    return redirect(url_for('main.upload_inputs'))

def generate_parameter_files(raw_param_path):
    df = pd.read_csv(raw_param_path)

    # activity frequency 생성
    group_cols = ['주상병_truc', '통합_진료구역']
    activities = df['new_activity'].dropna().unique()

    result_rows = []

    for (disease, dept), g_df in df.groupby(group_cols):
        row = {group_cols[0]: disease, group_cols[1]: dept}

        case_groups = g_df.groupby('case_id')['new_activity'].apply(list)

        for act in activities:
            counts = [acts.count(act) for acts in case_groups]
            if len(counts) == 0:
                value_dict = {0: 1.0}
            else:
                total_cases = len(counts)
                counter = Counter(counts)
                value_dict = {cnt: round(count/total_cases, 6) for cnt, count in counter.items()}

            row[f"{act}_pdf"] = json.dumps(value_dict, ensure_ascii=False)

        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)
    activity_frequency_path = "./data/activity_fre-hussh_generated.csv"
    result_df.to_csv(activity_frequency_path, index=False, encoding="utf-8-sig")

    # event duration 
    activity_col = 'new_activity'
    duration_col = 'case_duration'  

    # 제외할 activity 목록
    exclude_activities = ['visit_end', 'visit_start']

    event_duration_dict = {}

    for act, group in df.groupby(activity_col):
        if act in exclude_activities:
            continue

        durations = group[duration_col].dropna().values

        durations = durations[durations >= 0]
        if len(durations) == 0:
            continue

        mean_val = durations.mean()
        if mean_val <= 0:
            continue
        lam = 1.0 / mean_val

        max_val = durations.max()
        value = np.linspace(0, max_val, 1000)

        # pdf 계산
        pdf = lam * np.exp(-lam * value)

        # 결과 저장
        event_duration_dict[act] = {
            'value': value,
            'pdf': pdf
        }
    event_duration_path = "./data/event_duration_generated.pickle"
    with open(event_duration_path, "wb") as f:
        pickle.dump(event_duration_dict, f)

    # activity resource 
    # 활동별-역할별 count 계산
    grouped = df.groupby(['new_activity', 'provider_role']).size().reset_index(name='count')

    # 전체를 담을 dict
    resource_pdf = {}

    for activity, sub_df in grouped.groupby('new_activity'):
        total = sub_df['count'].sum()
        role_dict = {}
        for _, row in sub_df.iterrows():
            role = row['provider_role']
            count = row['count']
            if role == 'Intern':
                key = 'Intern_pdf'
            elif role == 'Resident':
                key = 'Resident_pdf'
            elif role == 'Specialist':
                key = 'Specialist_pdf'
            elif role == 'Junior_Nurse':
                key = 'Junior_Nurse_pdf'
            elif role == 'Senior_Nurse':
                key = 'Senior_Nurse_pdf'
            else:
                key = f'{role}_pdf'
            role_dict[key] = round(count / total, 4)  

        for default_role in ['Intern_pdf','Resident_pdf','Specialist_pdf','Junior_Nurse_pdf','Senior_Nurse_pdf']:
            role_dict.setdefault(default_role, 0.0)

        resource_pdf[activity] = role_dict


    activity_resource_path = './data/activity_resource_generated.pickle'
    with open(activity_resource_path, 'wb') as f:
        pickle.dump(resource_pdf, f)

    # transition time
    case_col = 'case_id'
    start_col = 'starttime'
    act_col = 'new_activity'
    trans_col = 'transition_time'

    df[start_col] = pd.to_datetime(df[start_col])

    records = []
    for cid, group in df.sort_values(by=[case_col, start_col]).groupby(case_col):
        acts = group[act_col].tolist()
        trans_times = group[trans_col].tolist()
        for i in range(len(acts)-1):
            records.append({
                'new_activity': acts[i],
                'next_act': acts[i+1],
                'trans_time': trans_times[i] 
            })

    trans_df = pd.DataFrame(records)

    result_df = trans_df.groupby(['new_activity', 'next_act'], as_index=False)['trans_time'].mean()

    transition_time_path = "./data/transition_time_generated.csv"
    result_df.to_csv(transition_time_path, index=False, encoding='utf-8-sig')

    return activity_frequency_path, event_duration_path, activity_resource_path, transition_time_path

@main.route('/upload_inputs', methods=['GET', 'POST'])
def upload_inputs():
    if request.method == 'POST':
        event_log = request.files.get('event_log')
        raw_param_data = request.files.get('raw_param_data')

        # 저장 경로 설정
        event_log_path = os.path.join(UPLOAD_FOLDER, event_log.filename)
        event_log.save(event_log_path)

        if raw_param_data and raw_param_data.filename != '':
            # raw parameter data를 이용해 파일 생성
            raw_param_path = os.path.join(UPLOAD_FOLDER, raw_param_data.filename)
            raw_param_data.save(raw_param_path)

            # 아래는 예시로 만든 생성 함수 (실제 로직에 맞게 수정)
            activity_freq_path, event_duration_path, activity_resource_path, transition_time_path = generate_parameter_files(raw_param_path)

        else:
            # 기존처럼 각 파일을 직접 사용
            activity_freq = request.files.get('activity_freq')
            event_duration = request.files.get('event_duration')
            activity_resource = request.files.get('activity_resource')
            transition_time = request.files.get('transition_time')

            # 업로드된 파일 저장
            activity_freq_path = os.path.join(UPLOAD_FOLDER, activity_freq.filename)
            event_duration_path = os.path.join(UPLOAD_FOLDER, event_duration.filename)
            activity_resource_path = os.path.join(UPLOAD_FOLDER, activity_resource.filename)
            transition_time_path = os.path.join(UPLOAD_FOLDER, transition_time.filename)

            activity_freq.save(activity_freq_path)
            event_duration.save(event_duration_path)
            activity_resource.save(activity_resource_path)
            transition_time.save(transition_time_path)

        # 이후 시뮬레이션 엔진으로 넘기기
        

        return run_simulation()

    return render_template('upload_inputs.html')

@main.route('/run_simulation')
def run_simulation():
    cases, start_date, arrival_times, variants, event_duration, resource_pdf, trans_time = prepare_simulation_data()

    simulator = Simulator(
        name="ER Simulation",
        cases=cases,
        start_date=start_date,
        arrival_times=arrival_times,
        event_duration=event_duration,
        variants=variants,
        resource_pdf=resource_pdf,
        trans_time=trans_time,
        resource_config=None
    )

    with Timer():
        simulator.run()

    result_df = pd.DataFrame(simulator.results).sort_values(['Patient_id', 'Start_time']).reset_index(drop=True)
    result_df['Activity'] = result_df['Activity'].str.replace(r'(_\d+)$', '', regex=True)
    median_dtp_res = result_df[['Patient_id', 'Start_time']].groupby('Patient_id')['Start_time'].apply(list).apply(lambda x: x[2] - x[1]).median()
    mean_dtp_res = result_df[['Patient_id', 'Start_time']].groupby('Patient_id')['Start_time'].apply(list).apply(lambda x: x[2] - x[1]).mean()
    kpis = format_result("ER Simulation", result_df, median_dtp_res, mean_dtp_res, simulator.resource_capacities)

    return render_template('results.html', results=kpis)