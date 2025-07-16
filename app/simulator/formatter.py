from collections import defaultdict
def max_median_mean(df):
    '''
    각 환자에 대해 체류 시간(마지막 이벤트 종료 시각 - 첫 이벤트 시작 시각)을 계산하고,
    그에 대한 최대/중앙/평균 값을 반환한다.
    '''
    df_temp = df.groupby('Patient_id')['End_time'].apply(list).apply(lambda x: x[-1] - x[0]).reset_index()
    max_val = df_temp['End_time'].max()
    median_val = df_temp['End_time'].median()
    mean_val = df_temp['End_time'].mean()

    return {
        "max": max_val,
        "median": median_val,
        "mean": mean_val
    }

# def format_resource_usage(df, label, pattern, capacity):
#     used = df.loc[df.Resource.str.contains(pattern), 'Resource'].nunique()
#     return f"{used} / {capacity}"

def format_resource_usage(df, pattern, capacity):
    """
    리소스 사용량을 고유한 (환자 ID, 시작시간)에 기반하여 계산하고,
    capacity보다 크지 않게 제한하여 표시.
    """
    usage_df = df.loc[df['Resource'].str.contains(pattern)]
    # 실제 중복 제거된 고유 사용 건수 계산
    used = usage_df[['Patient_id', 'Start_time']].drop_duplicates().shape[0]
    used = min(used, capacity)  # capacity 초과 방지
    # return f"{used} / {capacity}"
    return f"{used}"

def format_role_usage(df, role_keyword):
    """
    역할 문자열 (예: 'Resident', 'Senior')이 Resource에 포함된 횟수를 계산
    """
    role_df = df[df['Resource'].str.contains(fr"\({role_keyword}\)")]
    used = role_df[['Patient_id', 'Start_time']].drop_duplicates().shape[0]
    return f"{used}"


def format_result(file_name, result, median_dtp_res, mean_dtp_res, resource_capacities, role_merge=None, shift_times=None):
    '''
    시뮬레이션 결과로부터 KPI 지표들을 계산하고 출력 포맷을 리스트로 구성한다.
    '''
    # num_patients = result.Patient_id.nunique()
    # num_docs = result.loc[result.Resource.str.contains('^Doc')]['Resource'].nunique()
    # num_nurses = result.loc[result.Resource.str.contains('^Nur')]['Resource'].nunique()

    # num_doc_1 = result.loc[(result.Resource.str.contains('^Doc')) & (result.Resource.str.contains('shift_1'))]['Resource'].nunique()
    # num_doc_2 = result.loc[(result.Resource.str.contains('^Doc')) & (result.Resource.str.contains('shift_2'))]['Resource'].nunique()

    # num_nur_1 = result.loc[(result.Resource.str.contains('^Nur')) & (result.Resource.str.contains('shift_1'))]['Resource'].nunique()
    # num_nur_2 = result.loc[(result.Resource.str.contains('^Nur')) & (result.Resource.str.contains('shift_2'))]['Resource'].nunique()
    # num_nur_3 = result.loc[(result.Resource.str.contains('^Nur')) & (result.Resource.str.contains('shift_3'))]['Resource'].nunique()

    median_waiting = round(median_dtp_res.total_seconds() / 60, 2)
    mean_waiting = round(mean_dtp_res.total_seconds() / 60, 2)

    case_duration = max_median_mean(result)

    # formatted_result = [
    #     ("File Name", file_name),
    #     ("Number of Patients", num_patients),
    #     ("Number of Doctors", num_docs),
    #     ("Number of Nurses", num_nurses),

    #     ("Number of Doctors in shift_1", num_doc_1),
    #     ("Number of Doctors in shift_2", num_doc_2),

    #     ("Number of Nurses in shift_1", num_nur_1),
    #     ("Number of Nurses in shift_2", num_nur_2),
    #     ("Number of Nurses in shift_3", num_nur_3),

    #     ("Median Waiting Time (minutes)", median_waiting),
    #     ("Mean Waiting Time (minutes)", mean_waiting),

    #     ("Case Duration (max)", case_duration['max']),
    #     ("Case Duration (median)", case_duration['median']),
    #     ("Case Duration (mean)", case_duration['mean'])
    # ]
    # formatted_result = [
    #     ("Number of Doctors", format_resource_usage(result, 'Doctor', '^Doc', 64)),
    #     ("Number of Nurses", format_resource_usage(result, 'Nurse', '^Nur', 13)),
    #     ("Number of Doctors in shift_1", format_resource_usage(result, 'Doctor_shift_1', 'Doc.*shift_1', 34)),
    #     ("Number of Doctors in shift_2", format_resource_usage(result, 'Doctor_shift_2', 'Doc.*shift_2', 30)),
    #     ("Number of Nurses in shift_1", format_resource_usage(result, 'Nurse_shift_1', 'Nur.*shift_1', 3)),
    #     ("Number of Nurses in shift_2", format_resource_usage(result, 'Nurse_shift_2', 'Nur.*shift_2', 7)),
    #     ("Number of Nurses in shift_3", format_resource_usage(result, 'Nurse_shift_3', 'Nur.*shift_3', 3)),

    #     ("Median Waiting Time (minutes)", median_waiting),
    #     ("Mean Waiting Time (minutes)", mean_waiting),

    #     ("Case Duration (max)", case_duration['max']),
    #     ("Case Duration (median)", case_duration['median']),
    #     ("Case Duration (mean)", case_duration['mean'])
    # ]

    formatted_result = [
        ("Number of Doctors", format_resource_usage(result, '^Doc', resource_capacities['Doctor_shift_1'] + resource_capacities['Doctor_shift_2'])),
        ("Number of Nurses", format_resource_usage(result, '^Nur', resource_capacities['Nurse_shift_1'] + resource_capacities['Nurse_shift_2'] + resource_capacities['Nurse_shift_3'])),
        ("Number of Doctors in shift_1", format_resource_usage(result, 'Doc.*shift_1', resource_capacities['Doctor_shift_1'])),
        ("Number of Doctors in shift_2", format_resource_usage(result, 'Doc.*shift_2', resource_capacities['Doctor_shift_2'])),
        ("Number of Nurses in shift_1", format_resource_usage(result, 'Nur.*shift_1', resource_capacities['Nurse_shift_1'])),
        ("Number of Nurses in shift_2", format_resource_usage(result, 'Nur.*shift_2', resource_capacities['Nurse_shift_2'])),
        ("Number of Nurses in shift_3", format_resource_usage(result, 'Nur.*shift_3', resource_capacities['Nurse_shift_3'])),
        # 역할별 의사
        ("→ Intern usage", format_role_usage(result, 'Intern')),
        ("→ Resident usage", format_role_usage(result, 'Resident')),
        ("→ Specialist usage", format_role_usage(result, 'Specialist')),

        # 역할별 간호사
        ("→ Junior Nurse usage", format_role_usage(result, 'Junior')),
        ("→ Senior Nurse usage", format_role_usage(result, 'Senior')),
    ]
    if role_merge:
        reverse_merge = defaultdict(list)
        for original, merged in role_merge.items():
            reverse_merge[merged].append(original)

        for merged_role, _ in reverse_merge.items():
            merged_usage = format_role_usage(result, merged_role)
            formatted_result.append((f"→ {merged_role} (merged)", merged_usage))

    formatted_result += [
        ("Median Waiting Time (minutes)", median_waiting),
        ("Mean Waiting Time (minutes)", mean_waiting),
        ("Case Duration (max)", case_duration['max']),
        ("Case Duration (median)", case_duration['median']),
        ("Case Duration (mean)", case_duration['mean'])
    ]
    if shift_times:
        for role, shifts in shift_times.items():
            for shift_name, (start, end) in shifts.items():
                formatted_result.append((
                    f"{role} {shift_name} time",
                    f"{start}시 ~ {end}시"
                ))
    return formatted_result