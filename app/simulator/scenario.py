from app.simulator.loader import load_and_prepare_data
from app.simulator.engine import Simulator
import pandas as pd
import pickle
from datetime import datetime

def simulate_from_loader(remove_activity=None, resequence_pair=None,  
                         doc_shift_1=None, doc_shift_2=None, 
                         nur_shift_1=None, nur_shift_2=None, nur_shift_3=None,
                           role_override=None, task_composition=None, role_merge=None,
                            shift_times=None, custom_variants=None, 
                            custom_cases=None, custom_arrival_times=None, 
                            queue_discipline='FiFO', variant_priority_map=None, **kwargs):
    # 데이터 로딩 및 전처리
    _, df_date_re, start_date, event_duration, resource_pdf, trans_time = load_and_prepare_data(remove_activity=remove_activity, resequence_pair=resequence_pair)

    
    default_config = {
        'Doctor_shift_1': 34,
        'Doctor_shift_2': 30,
        'Nurse_shift_1': 3,
        'Nurse_shift_2': 7,
        'Nurse_shift_3': 3
    }

    # 개별 입력을 통해 각 역할별 인원 수를 합산
    role_keys = ['Intern', 'Resident', 'Specialist', 'Junior', 'Senior']
    shift_keys = ['shift_1', 'shift_2', 'shift_3']

    # 초기화
    full_config = {
        'Doctor_shift_1': 0, 'Doctor_shift_2': 0,
        'Nurse_shift_1': 0, 'Nurse_shift_2': 0, 'Nurse_shift_3': 0
    }

    for role in role_keys:
        for shift in shift_keys:
            key = f"{role}_{shift}"
            val = int(kwargs.get(key, 0))
            if role in ['Intern', 'Resident', 'Specialist']:
                if shift in ['shift_1', 'shift_2']:
                    full_config[f'Doctor_{shift}'] += val
            else:
                full_config[f'Nurse_{shift}'] += val

    resource_config = full_config if any(full_config.values()) else default_config
    # print(resource_config)

    if doc_shift_1 is not None:
        resource_config['Doctor_shift_1'] = doc_shift_1
    if doc_shift_2 is not None:
        resource_config['Doctor_shift_2'] = doc_shift_2
    if nur_shift_1 is not None:
        resource_config['Nurse_shift_1'] = nur_shift_1
    if nur_shift_2 is not None:
        resource_config['Nurse_shift_2'] = nur_shift_2
    if nur_shift_3 is not None:
        resource_config['Nurse_shift_3'] = nur_shift_3

    if role_override:
        for activity, new_role in role_override.items():
            if new_role in ['Intern', 'Resident', 'Specialist']:  # Doctor role
                resource_pdf[activity]['Doctor_pdf'] = {
                    role: 1.0 if role == new_role else 0.0 for role in ['Intern', 'Resident', 'Specialist']
                }
                resource_pdf[activity]['Nurse_pdf'] = {
                    role: 0.0 for role in ['Junior', 'Senior']
                }
                resource_pdf[activity]['Role_prob'] = {
                    'Doctor': 1.0,
                    'Nurse': 0.0
                }
            elif new_role in ['Junior', 'Senior']:  # Nurse role
                resource_pdf[activity]['Nurse_pdf'] = {
                    role: 1.0 if role == new_role else 0.0 for role in ['Junior', 'Senior']
                }
                resource_pdf[activity]['Doctor_pdf'] = {
                    role: 0.0 for role in ['Intern', 'Resident', 'Specialist']
                }
                resource_pdf[activity]['Role_prob'] = {
                    'Doctor': 0.0,
                    'Nurse': 1.0
                }
            else:
                print(f"[Warning] Unknown role '{new_role}' for activity '{activity}'")


    if role_merge:
        for activity, config in resource_pdf.items():
            # Role_prob 업데이트
            old_role_prob = config['Role_prob']
            new_role_prob = {}
            for role, prob in old_role_prob.items():
                merged = role_merge.get(role, role)
                new_role_prob[merged] = new_role_prob.get(merged, 0) + prob
            config['Role_prob'] = new_role_prob

            # Doctor_pdf 업데이트
            if 'Doctor_pdf' in config:
                old_doc_pdf = config['Doctor_pdf']
                new_doc_pdf = {}
                for role, prob in old_doc_pdf.items():
                    merged = role_merge.get(role, role)
                    new_doc_pdf[merged] = new_doc_pdf.get(merged, 0) + prob
                config['Doctor_pdf'] = new_doc_pdf

            # Nurse_pdf 업데이트
            if 'Nurse_pdf' in config:
                old_nur_pdf = config['Nurse_pdf']
                new_nur_pdf = {}
                for role, prob in old_nur_pdf.items():
                    merged = role_merge.get(role, role)
                    new_nur_pdf[merged] = new_nur_pdf.get(merged, 0) + prob
                config['Nurse_pdf'] = new_nur_pdf
    
    if resequence_pair:
        df_new = pickle.load(open('app/data/df_modified.pkl', 'rb'))
    else:
        df_new = df_date_re.copy()

    df_new['starttime'] = pd.to_datetime(df_new['starttime'])
    df_new = df_new.sort_values(by='starttime')
    cases = list(df_new.case_id)
    arrival_times = [(i - start_date).total_seconds() for i in sorted(list(df_new['starttime']))]
    variants = list(df_new.variant)

    if task_composition:
        for activity, new_probs in task_composition.items():
            if activity in resource_pdf:
                resource_pdf[activity]['Role_prob'] = new_probs

    if custom_variants is not None:
        variants = custom_variants

    if custom_cases is not None and custom_arrival_times is not None:
        cases = custom_cases
    else:
        cases = list(df_new.case_id)
        arrival_times = [(i - start_date).total_seconds() for i in sorted(list(df_new['starttime']))]
        

    def parse_shift(t, default=0):
        try:
            return int(t)
        except:
            return default

    if shift_times:
        parsed_shift_times = {
            'Doctor_shift_1': (parse_shift(shift_times['Doctor']['shift_1'][0]), parse_shift(shift_times['Doctor']['shift_1'][1])),
            'Doctor_shift_2': (parse_shift(shift_times['Doctor']['shift_2'][0]), parse_shift(shift_times['Doctor']['shift_2'][1])),
            'Nurse_shift_1': (parse_shift(shift_times['Nurse']['shift_1'][0]), parse_shift(shift_times['Nurse']['shift_1'][1])),
            'Nurse_shift_2': (parse_shift(shift_times['Nurse']['shift_2'][0]), parse_shift(shift_times['Nurse']['shift_2'][1])),
            'Nurse_shift_3': (parse_shift(shift_times['Nurse']['shift_3'][0]), parse_shift(shift_times['Nurse']['shift_3'][1]))
        }
        simulator = Simulator(
            name="ScenarioSimulation",
            cases=cases,
            start_date=start_date,
            arrival_times=arrival_times,
            event_duration=event_duration,
            variants=variants,
            resource_pdf=resource_pdf,
            trans_time=trans_time,
            resource_config=resource_config,
            shift_times=parsed_shift_times,
            queue_discipline=queue_discipline,
            variant_priority_map=variant_priority_map
        )
        
    else:
        simulator = Simulator(
            name="ScenarioSimulation",
            cases=cases,
            start_date=start_date,
            arrival_times=arrival_times,
            event_duration=event_duration,
            variants=variants,
            resource_pdf=resource_pdf,
            trans_time=trans_time,
            resource_config=resource_config,
            queue_discipline=queue_discipline,
            variant_priority_map=variant_priority_map
        )

    simulator.run()
    return simulator
