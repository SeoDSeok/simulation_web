from app.simulator.loader import load_and_prepare_data
from app.simulator.engine import Simulator
import pandas as pd
import pickle

def simulate_from_loader(
    remove_activity=None, resequence_pair=None,
    Intern_shift_1=None, Intern_shift_2=None,
    Resident_shift_1=None, Resident_shift_2=None,
    Specialist_shift_1=None, Specialist_shift_2=None,
    Junior_shift_1=None, Junior_shift_2=None, Junior_shift_3=None,
    Senior_shift_1=None, Senior_shift_2=None, Senior_shift_3=None,
    role_override=None, task_composition=None, role_merge=None,
    shift_times=None, custom_variants=None, custom_cases=None, custom_arrival_times=None,
    queue_discipline='FIFO', variant_priority_map=None,
    event_duration = None, resource_pdf = None, trans_time = None, modified_transition_time=None
):
    # ================================
    # 1) 데이터 로딩
    # ================================
    _, df_date_re, start_date, base_event_duration, base_resource_pdf, base_trans_time = load_and_prepare_data(
        remove_activity=remove_activity, resequence_pair=resequence_pair
    )

    # 새로 전달된 것 있으면 추가하기
    if event_duration:
        base_event_duration.update(event_duration)
    if resource_pdf:
        base_resource_pdf.update(resource_pdf)
    if trans_time:
        for k,v in trans_time.items():
            base_trans_time.loc[k, 'trans_time'] = v['trans_time']
    if modified_transition_time:
        for (from_act, to_act), time_info in modified_transition_time.items():
            base_trans_time.loc[(from_act, to_act), 'trans_time'] = time_info['trans_time']

    # ================================
    # 2) 기본 리소스 capacity 설정
    # ================================
    default_config = {
        'intern_shift_1': 10, 'intern_shift_2': 10,
        'resident_shift_1': 10, 'resident_shift_2': 10,
        'specialist_shift_1': 14, 'specialist_shift_2': 10,
        'junior_shift_1': 2, 'junior_shift_2': 5, 'junior_shift_3': 2,
        'senior_shift_1': 1, 'senior_shift_2': 2, 'senior_shift_3': 1
    }
    resource_config = default_config.copy()

    # 사용자 지정 shift별 인원수 반영
    def set_cap(key, value):
        if value is not None:
            resource_config[key] = int(value)
    set_cap('intern_shift_1', Intern_shift_1)
    set_cap('intern_shift_2', Intern_shift_2)
    set_cap('resident_shift_1', Resident_shift_1)
    set_cap('resident_shift_2', Resident_shift_2)
    set_cap('specialist_shift_1', Specialist_shift_1)
    set_cap('specialist_shift_2', Specialist_shift_2)
    set_cap('junior_shift_1', Junior_shift_1)
    set_cap('junior_shift_2', Junior_shift_2)
    set_cap('junior_shift_3', Junior_shift_3)
    set_cap('senior_shift_1', Senior_shift_1)
    set_cap('senior_shift_2', Senior_shift_2)
    set_cap('senior_shift_3', Senior_shift_3)

    # ================================
    # 3) role_override 반영
    # ================================
    doctor_roles = ['Intern', 'Resident', 'Specialist']
    nurse_roles = ['Junior_Nurse', 'Senior_Nurse']
    if role_override:
        for activity, new_role in role_override.items():
            if activity not in base_resource_pdf:
                continue
            # 모든 기존 역할 0으로 초기화
            for r in doctor_roles: base_resource_pdf[activity][f"{r}_pdf"] = 0.0
            for r in nurse_roles: base_resource_pdf[activity][f"{r}_pdf"] = 0.0
            # 선택된 역할만 1.0
            if new_role in doctor_roles:
                base_resource_pdf[activity][f"{new_role}_pdf"] = 1.0
            elif new_role == 'Junior':
                base_resource_pdf[activity]["Junior_Nurse_pdf"] = 1.0
            elif new_role == 'Senior':
                base_resource_pdf[activity]["Senior_Nurse_pdf"] = 1.0

    # ================================
    # 4) merged role 반영
    # ================================
    if role_merge:
        updated_pdf = {}
        for act, pdf in base_resource_pdf.items():
            merged_roles_pdf = {}
            for role_key, prob in pdf.items():
                base_role = role_key.replace("_pdf", "")
                if base_role in role_merge:
                    merged_name = role_merge[base_role]
                    merged_roles_pdf[merged_name] = merged_roles_pdf.get(merged_name, 0) + prob
                else:
                    merged_roles_pdf[base_role] = merged_roles_pdf.get(base_role, 0) + prob
            total = sum(merged_roles_pdf.values())
            if total > 0:
                merged_roles_pdf = {f"{r}_pdf": v/total for r,v in merged_roles_pdf.items()}
            updated_pdf[act] = merged_roles_pdf
        base_resource_pdf = updated_pdf
        # print(resource_pdf)
        # 리소스 capacity와 shift_times에 동적 추가
        if shift_times is None:
            shift_times = {}
        for orig, merged in role_merge.items():
            merged_prefix = merged.lower().replace(' ', '_')
            if orig.lower() in ['intern','resident','specialist']:
                for s in ['shift_1','shift_2']:
                    key = f"{merged_prefix}_{s}"
                    resource_config.setdefault(key, 20)
                    shift_times.setdefault(key, (8,20))
            elif orig.lower() in ['junior','senior']:
                for s in ['shift_1','shift_2','shift_3']:
                    key = f"{merged_prefix}_{s}"
                    resource_config.setdefault(key, 13)
                    shift_times.setdefault(key, (6,14))
    # ================================
    # 5) task_composition 반영
    # ================================
    if task_composition:
        for activity, new_probs in task_composition.items():
            if activity in base_resource_pdf:
                total = sum(new_probs.values())
                if total > 0:
                    normalized = {k: v/total for k,v in new_probs.items()}
                    base_resource_pdf[activity].update(normalized)

    # ================================
    # 6) 이벤트 데이터 구성
    # ================================
    if resequence_pair:
        df_new = pickle.load(open('app/data/df_modified.pkl','rb'))
    else:
        df_new = df_date_re.copy()
    df_new['starttime'] = pd.to_datetime(df_new['starttime'])
    df_new = df_new.sort_values(by='starttime')
    cases = list(df_new.case_id)
    arrival_times = [(t - start_date).total_seconds() for t in df_new['starttime']]
    variants = list(df_new.variant)

    if custom_variants is not None:
        variants = custom_variants
        if remove_activity:
            variants = [[act for act in v if act != remove_activity] for v in variants]
    if custom_cases is not None:
        cases = custom_cases
    if custom_arrival_times is not None:
        arrival_times = custom_arrival_times

    # ================================
    # 7) shift_times 파싱
    # ================================
    def parse_shift(t, default=0):
        try: return int(t)
        except: return default

    parsed_shift_times = {k: (parse_shift(v[0]), parse_shift(v[1])) for k, v in (shift_times or {}).items()}
    # parsed_shift_times = {
    #     role: {
    #         shift: (parse_shift(t[0]), parse_shift(t[1]))
    #         for shift, t in shifts.items()
    #     }
    #     for role, shifts in (shift_times or {}).items()
    # }
    # print(parsed_shift_times)
    # print(resource_pdf)
    # ================================
    # 8) 시뮬레이터 실행
    # ================================
    sim = Simulator(
        name="ScenarioSimulation",
        cases=cases,
        start_date=start_date,
        arrival_times=arrival_times,
        event_duration=base_event_duration,
        variants=variants,
        resource_pdf=base_resource_pdf,
        trans_time=base_trans_time,
        resource_config=resource_config,
        shift_times=parsed_shift_times if parsed_shift_times else None,
        queue_discipline=queue_discipline,
        variant_priority_map=variant_priority_map
    )
    sim.run()
    return sim
