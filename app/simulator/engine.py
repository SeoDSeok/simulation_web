import simpy
import random
import numpy as np

class Simulator:
    def __init__(self, name, cases, start_date, arrival_times, event_duration, variants,
                 resource_pdf, trans_time, resource_config=None, shift_times=None,
                 queue_discipline='FIFO', variant_priority_map=None):
        self.name = name
        self.cases = cases
        self.start_date = start_date
        self.arrival_times = arrival_times
        self.event_duration = event_duration
        self.variants = variants
        self.resource_pdf = resource_pdf
        self.trans_time = trans_time
        self.queue_discipline = queue_discipline
        self.variant_priority_map = variant_priority_map or {}
        self.env = simpy.Environment()

        # capacity 초기값
        default_config = {
            'intern_shift_1': 10, 'intern_shift_2': 10,
            'resident_shift_1': 10, 'resident_shift_2': 10,
            'specialist_shift_1': 14, 'specialist_shift_2': 10,
            'junior_shift_1': 2, 'junior_shift_2': 5, 'junior_shift_3': 2,
            'senior_shift_1': 1, 'senior_shift_2': 2, 'senior_shift_3': 1
        }
        self.resource_capacities = resource_config if resource_config else default_config.copy()
        # print(self.resource_capacities)
        # shift times
        default_shift_times = {
            'intern_shift_1': (8, 20),
            'intern_shift_2': (20, 8),
            'resident_shift_1': (8, 20),
            'resident_shift_2': (20, 8),
            'specialist_shift_1': (8, 20),
            'specialist_shift_2': (20, 8),
            'junior_shift_1': (6, 14),
            'junior_shift_2': (14, 22),
            'junior_shift_3': (22, 6),
            'senior_shift_1': (6, 14),
            'senior_shift_2': (14, 22),
            'senior_shift_3': (22, 6),
        }
        self.shift_times = shift_times if shift_times else default_shift_times.copy()
        # print(self.shift_times)
        # === 동적으로 Store 생성 ===
        self.store_map = {}
        for key, cap in self.resource_capacities.items():
            # key 예: intern_shift_1, junior_doctor_shift_1 등
            self.store_map[key] = simpy.PriorityStore(self.env, capacity=cap)

        self.results = []

    def MakeStore(self, store, name, capacity):
        for c in range(capacity):
            priority = -c if self.queue_discipline == 'LIFO' else 0
            yield store.put((priority, f"{name}_{c}"))

    def run(self):
        # 모든 store 초기화
        for key, store in self.store_map.items():
            cap = self.resource_capacities.get(key, 0)
            self.env.process(self.MakeStore(store, key, cap))
        # print(len(self.cases), len(self.variants))
        # 케이스 실행
        for idx, case in enumerate(self.cases):
            self.env.process(
                Activity(self.env, self, idx, self.variants[idx])
                .run_activity(self.arrival_times[idx] - self.env.now, case, self.variants[idx], 0)
            )
        self.env.run()

    def record_result(self, patient_id, event, start_time, end_time, resource):
        self.results.append({
            'Patient_id': patient_id,
            'Start_time': start_time,
            'End_time': end_time,
            'Activity': event,
            'Resource': resource
        })


class Activity:
    def __init__(self, env, simulator, case_id, variant):
        self.env = env
        self.simulator = simulator
        self.case_id = case_id
        self.variant = tuple(variant)
        self.ratio = 0.1

    def now_time(self):
        return self.simulator.start_date + np.timedelta64(int(self.env.now), 's')

    def determine_priority(self):
        if self.variant in self.simulator.variant_priority_map:
            return self.simulator.variant_priority_map[self.variant]
        return 0

    def find_store_for_role(self, role, hour):
        """
        role 예: Intern, Resident, Specialist, Junior Doctor(merged) 등
        """
        role_key = role.lower().replace(' ', '_')  # Intern -> intern, Junior Doctor -> junior_doctor
        # 시프트 시간들을 확인
        matches = [k for k in self.simulator.shift_times.keys() if k.startswith(role_key)]
        if not matches:
            # 기본 Intern 로직 fallback
            if role_key.startswith('intern'):
                return self.simulator.store_map['intern_shift_1'] if 8 <= hour < 20 else self.simulator.store_map['intern_shift_2']
            if role_key.startswith('resident'):
                return self.simulator.store_map['resident_shift_1'] if 8 <= hour < 20 else self.simulator.store_map['resident_shift_2']
            if role_key.startswith('specialist'):
                return self.simulator.store_map['specialist_shift_1'] if 8 <= hour < 20 else self.simulator.store_map['specialist_shift_2']
            if role_key.startswith('junior'):
                if 6 <= hour < 14: return self.simulator.store_map['junior_shift_1']
                if 14 <= hour < 22: return self.simulator.store_map['junior_shift_2']
                return self.simulator.store_map['junior_shift_3']
            if role_key.startswith('senior'):
                if 6 <= hour < 14: return self.simulator.store_map['senior_shift_1']
                if 14 <= hour < 22: return self.simulator.store_map['senior_shift_2']
                return self.simulator.store_map['senior_shift_3']
        else:
            # 동적 shift
            for shift in matches:
                start, end = self.simulator.shift_times[shift]
                # 기본적으로 start<end 형태
                if start < end:
                    if start <= hour < end:
                        return self.simulator.store_map[shift]
                else:
                    # 야간 교대(예: 20~8)
                    if hour >= start or hour < end:
                        return self.simulator.store_map[shift]
            # 기본 첫 번째 시프트
            return self.simulator.store_map[matches[0]]

    def run_activity(self, time_diff, case, variant, n_index):
        if time_diff > 0:
            yield self.env.timeout(time_diff)

        # 다음 이벤트 없으면 종료
        if n_index >= len(variant):
            return

        event = variant[n_index]
        try:
            previous_event = variant[n_index - 1]
            transition_time = self.simulator.trans_time.loc[(previous_event, event)]['trans_time']
        except Exception:
            transition_time = 0.0
        yield self.env.timeout(self.ratio * transition_time)

        # 역할 할당
        pdf_info = self.simulator.resource_pdf[event]
        roles = list(pdf_info.keys())
        weights = list(pdf_info.values())
        selected_role = random.choices(roles, weights=weights, k=1)[0].replace('_pdf', '')

        current_time = self.now_time()  # datetime 또는 Timestamp
        hour = current_time.hour 
        store = self.find_store_for_role(selected_role, hour)

        priority, resource = yield store.get()
        duration = random.choices(
            self.simulator.event_duration[event]['value'],
            self.simulator.event_duration[event]['pdf'],
            k=1
        )[0]
        start_time = self.now_time()
        yield self.env.timeout(duration)
        self.simulator.record_result(case, event, start_time, self.now_time(), f"{resource} ({selected_role})")

        # 다시 store에 반환
        if self.simulator.queue_discipline == 'LIFO':
            yield store.put((-priority, resource))
        elif self.simulator.queue_discipline == 'PRIORITY':
            yield store.put((self.determine_priority(), resource))
        else:
            yield store.put((priority, resource))

        # 다음 이벤트로 진행
        return self.env.process(self.run_activity(0, case, variant, n_index + 1))

    
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, *args):
        import time
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Elapsed_time: {self.interval} seconds")
