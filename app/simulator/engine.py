import simpy
import random
import numpy as np

class Simulator:
    def __init__(self, name, cases, start_date, arrival_times, event_duration, variants, 
    resource_pdf, trans_time, resource_config=None, shift_times=None, queue_discipline='FIFO', variant_priority_map=None):
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
        print(self.queue_discipline)
        print(self.variant_priority_map)
        # 기본 리소스 설정
        default_config = {
            'Doctor_shift_1': 34,
            'Doctor_shift_2': 30,
            'Nurse_shift_1': 3,
            'Nurse_shift_2': 7,
            'Nurse_shift_3': 3
        }
        
        self.resource_capacities = resource_config if resource_config else default_config

        # 기본 shift 시간 설정 (24시간 기준)
        default_shift_times = {
            'Doctor_shift_1': (8, 20),
            'Doctor_shift_2': (20, 8), 
            'Nurse_shift_1': (6, 14),
            'Nurse_shift_2': (14, 22),
            'Nurse_shift_3': (22, 6)
        }
        self.shift_times = shift_times or default_shift_times
        

        self.doctor_shift_1 = simpy.PriorityStore(self.env, capacity=self.resource_capacities['Doctor_shift_1'])
        self.doctor_shift_2 = simpy.PriorityStore(self.env, capacity=self.resource_capacities['Doctor_shift_2'])
        self.nurse_shift_1 = simpy.PriorityStore(self.env, capacity=self.resource_capacities['Nurse_shift_1'])
        self.nurse_shift_2 = simpy.PriorityStore(self.env, capacity=self.resource_capacities['Nurse_shift_2'])
        self.nurse_shift_3 = simpy.PriorityStore(self.env, capacity=self.resource_capacities['Nurse_shift_3'])

        self.doctor_roles = ['Intern', 'Resident', 'Specialist']
        self.nurse_roles = ['Junior', 'Senior']
        self.results = []

    def run(self):
        print(self.name)
        print(self.queue_discipline)
        print(self.variant_priority_map)
        self.env.process(self.MakeStore(self.doctor_shift_1, 'Doctor_shift_1', self.resource_capacities['Doctor_shift_1']))
        self.env.process(self.MakeStore(self.doctor_shift_2, 'Doctor_shift_2', self.resource_capacities['Doctor_shift_2']))
        self.env.process(self.MakeStore(self.nurse_shift_1, 'Nurse_shift_1', self.resource_capacities['Nurse_shift_1']))
        self.env.process(self.MakeStore(self.nurse_shift_2, 'Nurse_shift_2', self.resource_capacities['Nurse_shift_2']))
        self.env.process(self.MakeStore(self.nurse_shift_3, 'Nurse_shift_3', self.resource_capacities['Nurse_shift_3']))

        for case in self.cases:
            case_index = self.cases.index(case)
            time_diff = self.arrival_times[case_index] - self.env.now
            variant = self.variants[case_index]
            self.env.process(Activity(self.env, self.doctor_shift_1, self.doctor_shift_2,
                                      self.nurse_shift_1, self.nurse_shift_2, self.nurse_shift_3,
                                      self, case_index, variant).run_activity(time_diff, case, variant, 0))

        self.env.run()

    # def MakeStore(self, store, name, capacity):
    #     for capa in range(capacity):
    #         yield store.put(f"{name} {capa}")
    def MakeStore(self, store, name, capacity):
        for capa in range(capacity):
            if self.queue_discipline == 'LIFO':
                # 음수 우선순위로 넣기 (나중에 넣은 게 먼저 나감)
                yield store.put((-capa, f"{name} {capa}"))
            else:
                # 기본 FIFO → 우선순위도 기본값 0으로
                yield store.put((0, f"{name} {capa}"))


    def record_result(self, patient_id, event, start_time, end_time, resource):
        self.results.append({
            'Patient_id': patient_id,
            'Start_time': start_time,
            'End_time': end_time,
            'Activity': event,
            'Resource': resource
        })

class Activity:
    def __init__(self, env, doctor_shift_1, doctor_shift_2, nurse_shift_1, nurse_shift_2, nurse_shift_3, simulator, case_id, variant):
        self.env = env
        self.doctor_shift_1 = doctor_shift_1
        self.doctor_shift_2 = doctor_shift_2
        self.nurse_shift_1 = nurse_shift_1
        self.nurse_shift_2 = nurse_shift_2
        self.nurse_shift_3 = nurse_shift_3
        self.simulator = simulator
        self.case_id = case_id 
        self.variant = tuple(variant)
        self.ratio = 0.1

    def determine_priority(self):
        # (원하는 로직으로 변경 가능: triage 점수 등)
        if self.variant in self.simulator.variant_priority_map:
            return self.simulator.variant_priority_map[self.variant]
        return 0

    def now_time(self):
        return self.simulator.start_date + np.timedelta64(int(self.env.now), 's')

    # def determine_shift(self, current_time):
    #     hour = current_time.hour
    #     doctor_store = self.doctor_shift_1 if 8 <= hour < 20 else self.doctor_shift_2
    #     if 6 <= hour < 14:
    #         nurse_store = self.nurse_shift_1
    #     elif 14 <= hour < 22:
    #         nurse_store = self.nurse_shift_2
    #     else:
    #         nurse_store = self.nurse_shift_3
    #     return doctor_store, nurse_store
    
    def determine_shift(self, current_time):
        hour = current_time.hour
        st = self.simulator.shift_times

        doc_start1, doc_end1 = st['Doctor_shift_1']
        doctor_store = self.doctor_shift_1 if (
            doc_start1 < doc_end1 and doc_start1 <= hour < doc_end1 or
            doc_start1 > doc_end1 and (hour >= doc_start1 or hour < doc_end1)
        ) else self.doctor_shift_2

        for shift_name, store in [('Nurse_shift_1', self.nurse_shift_1),
                                ('Nurse_shift_2', self.nurse_shift_2),
                                ('Nurse_shift_3', self.nurse_shift_3)]:
            start, end = st[shift_name]
            if start < end and start <= hour < end:
                return doctor_store, store
            elif start > end and (hour >= start or hour < end):  # 예: 22~6
                return doctor_store, store

        # fallback
        return doctor_store, self.nurse_shift_1


    def run_activity(self, time_diff, case, variant, n_index):
        if time_diff > 0:
            yield self.env.timeout(time_diff)
        try:
            event = variant[n_index]
        except:
            return
        try:
            previous_event = variant[n_index - 1]
            transition_time = self.simulator.trans_time.loc[(previous_event, event)]['trans_time']
        except:
            transition_time = 0.0

        yield self.env.timeout(self.ratio * transition_time)

        resource_allocated = False
        while not resource_allocated:
            current_time = self.now_time()
            doctor_store, nurse_store = self.determine_shift(current_time)
            pdf_info = self.simulator.resource_pdf[event]
            role_prob = pdf_info['Role_prob']
            # role = random.choices(['Doctor', 'Nurse'], weights=[role_prob['Doctor'], role_prob['Nurse']], k=1)[0]
            role = random.choices(['Doctor', 'Nurse'],
                      weights=[role_prob.get('Doctor', 0.0), role_prob.get('Nurse', 0.0)],
                      k=1)[0]
            doc_prob = role_prob.get('Doctor', 0)
            nurse_prob = role_prob.get('Nurse', 0)
            if doc_prob > 0 and nurse_prob == 0:
                role = 'Doctor'
            elif nurse_prob > 0 and doc_prob == 0:
                role = 'Nurse'
            else:
                # role = random.choices(['Doctor', 'Nurse'], weights=[doc_prob, nurse_prob], k=1)[0]
                role = random.choices(['Doctor', 'Nurse'],
                      weights=[role_prob.get('Doctor', 0.0), role_prob.get('Nurse', 0.0)],
                      k=1)[0]
            if role == 'Doctor':
                doc_weights = list(pdf_info['Doctor_pdf'].values())
                if sum(doc_weights) == 0:
                    raise ValueError(f"[ERROR] All Doctor weights are zero in event '{event}'. Cannot select role.")
                selected_role = random.choices(
                    population=list(pdf_info['Doctor_pdf'].keys()),
                    weights=list(pdf_info['Doctor_pdf'].values()), k=1
                )[0]
                chosen_store = doctor_store
            else:
                nur_weights = list(pdf_info['Nurse_pdf'].values())
                if sum(nur_weights) == 0:
                    raise ValueError(f"[ERROR] All Nurse weights are zero in event '{event}'. Cannot select role.")
                selected_role = random.choices(
                    population=list(pdf_info['Nurse_pdf'].keys()),
                    weights=list(pdf_info['Nurse_pdf'].values()), k=1
                )[0]
                chosen_store = nurse_store

            resource_allocated = True

        # resource = yield chosen_store.get()
        priority, resource = yield chosen_store.get()

        event_duration = random.choices(self.simulator.event_duration[event]['value'],
                                        self.simulator.event_duration[event]['pdf'], k=1)[0]
        start_time = self.now_time()
        yield self.env.timeout(event_duration)
        self.simulator.record_result(
            case, event, start_time, self.now_time(),
            f"{resource} ({selected_role})"
        )
        # yield chosen_store.put(resource)
        if self.simulator.queue_discipline == 'LIFO':
            yield chosen_store.put((-priority, resource))
        elif self.simulator.queue_discipline == 'PRIORITY':
            # 새로운 priority 계산 가능
            yield chosen_store.put((self.determine_priority(), resource))
        else:
            yield chosen_store.put((priority, resource))

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
