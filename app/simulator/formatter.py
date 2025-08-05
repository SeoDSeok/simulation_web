from collections import defaultdict

def format_role_usage(df, role_keyword):
    role_df = df[df['Resource'].str.contains(fr"\({role_keyword}\)")]
    used = role_df[['Patient_id','Start_time']].drop_duplicates().shape[0]
    return f"{used}"

def format_resource_usage(df, pattern, capacity):
    usage_df = df.loc[df['Resource'].str.contains(pattern)]
    used = usage_df[['Patient_id','Start_time']].drop_duplicates().shape[0]
    used = min(used, capacity)
    return f"{used}"

def max_median_mean(df):
    df_temp = df.groupby('Patient_id')['End_time'].apply(list).apply(lambda x: x[-1]-x[0]).reset_index()
    return {
        "max": df_temp['End_time'].max(),
        "median": df_temp['End_time'].median(),
        "mean": df_temp['End_time'].mean()
    }


def format_result(file_name, result, median_dtp_res, mean_dtp_res,
                   resource_capacities, role_merge=None, shift_times=None):
    formatted_result = []
    # print(role_merge)
    doctor_merged = any(v.lower() == 'junior doctor' for v in (role_merge or {}).values())
    nurse_merged = any(v.lower() == 'nurse' for v in (role_merge or {}).values())

    # --- 1. Doctor Shifts ---
    if not doctor_merged:
        formatted_result.append(("Doctor Shift", ""))  # header
        formatted_result += [
            ("Number of Intern Shift 1", str(resource_capacities.get('intern_shift_1', 0))),
            ("Number of Intern Shift 2", str(resource_capacities.get('intern_shift_2', 0))),
            ("Number of Resident Shift 1", str(resource_capacities.get('resident_shift_1', 0))),
            ("Number of Resident Shift 2", str(resource_capacities.get('resident_shift_2', 0))),
            ("Number of Specialist Shift 1", str(resource_capacities.get('specialist_shift_1', 0))),
            ("Number of Specialist Shift 2", str(resource_capacities.get('specialist_shift_2', 0))),
        ]
    else:
        formatted_result.append(("Doctor Shift (Merged)", ""))
        for k in resource_capacities:
            if k.startswith('junior_doctor_'):
                print(k)
                formatted_result.append((f"Number of {k.replace('_', ' ').title()}", str(resource_capacities[k])))
                print(formatted_result)
        formatted_result += [
            ("Number of Specialist Shift 1", str(resource_capacities.get('specialist_shift_1', 0))),
            ("Number of Specialist Shift 2", str(resource_capacities.get('specialist_shift_2', 0))),
        ]

    # --- 2. Nurse Shifts ---
    if not nurse_merged:
        formatted_result.append(("Nurse Shift", ""))  # header
        formatted_result += [
            ("Number of Junior Shift 1", str(resource_capacities.get('junior_shift_1', 0))),
            ("Number of Junior Shift 2", str(resource_capacities.get('junior_shift_2', 0))),
            ("Number of Junior Shift 3", str(resource_capacities.get('junior_shift_3', 0))),
            ("Number of Senior Shift 1", str(resource_capacities.get('senior_shift_1', 0))),
            ("Number of Senior Shift 2", str(resource_capacities.get('senior_shift_2', 0))),
            ("Number of Senior Shift 3", str(resource_capacities.get('senior_shift_3', 0))),
        ]
    else:
        formatted_result.append(("Nurse Shift (Merged)", ""))  # header
        for k in resource_capacities:
            if k.startswith('nurse_'):
                formatted_result.append((f"Number of {k.replace('_', ' ').title()}", str(resource_capacities[k])))

    # --- 3. Waiting Time ---
    formatted_result.append(("Waiting Time", ""))  # header
    median_waiting = round(median_dtp_res.total_seconds() / 60, 2)
    mean_waiting = round(mean_dtp_res.total_seconds() / 60, 2)
    formatted_result += [
        ("Median Waiting Time (minutes)", median_waiting),
        ("Mean Waiting Time (minutes)", mean_waiting),
    ]

    # --- 4. Duration Time ---
    formatted_result.append(("Duration Time", ""))  # header
    case_duration = max_median_mean(result)
    formatted_result += [
        ("Case Duration (max)", case_duration['max']),
        ("Case Duration (median)", case_duration['median']),
        ("Case Duration (mean)", case_duration['mean']),
    ]
    # ========================
    # 5) Shift time 정보
    # ========================
    # print(shift_times)
    if shift_times:
        # --- Doctor Shifts 출력 여부 ---
        show_doctor_shifts = not doctor_merged and any(
            k.startswith(('intern_', 'resident_', 'specialist_')) for k in shift_times
        )
        if show_doctor_shifts:
            formatted_result.append(("Doctor Shift Time", ""))
            for key, (start, end) in shift_times.items():
                if key.startswith(('intern_', 'resident_', 'specialist_')):
                    formatted_result.append((f"{key} time", f"{start}시~{end}시"))

        # --- Nurse Shifts 출력 여부 ---
        show_nurse_shifts = not nurse_merged and any(
            k.startswith(('junior_', 'senior_')) for k in shift_times
        )
        if show_nurse_shifts:
            formatted_result.append(("Nurse Shift Time", ""))
            for key, (start, end) in shift_times.items():
                if key.startswith(('junior_', 'senior_')):
                    formatted_result.append((f"{key} time", f"{start}시~{end}시"))

        if doctor_merged:
            formatted_result.append(("Doctor Shift Time (Merged)", ""))
            for key in shift_times:
                if key.startswith('junior_doctor_'):
                    start, end = shift_times[key]
                    formatted_result.append((f"{key} time", f"{start}시~{end}시"))

        if nurse_merged:
            formatted_result.append(("Nurse Shift Time (Merged)", ""))
            for key in shift_times:
                if key.startswith('nurse_'):
                    start, end = shift_times[key]
                    formatted_result.append((f"{key} time", f"{start}시~{end}시"))


    return formatted_result




