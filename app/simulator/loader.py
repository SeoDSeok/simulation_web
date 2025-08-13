import pandas as pd
import numpy as np
import pickle
import random
import os

def load_inputs(data_dir = './app/data'):
    df = pd.read_csv(os.path.join(data_dir, 'masked_log.csv'))
    df = df.drop(['Unnamed: 0'], axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['starttime'] = pd.to_datetime(df['starttime'])
    df['endtime'] = pd.to_datetime(df['endtime'])
    df['주상병_two'] = df['주상병'].str[:1]
    df['주상병_three'] = df['주상병'].str[:2]

    df_date = df[(df['new_activity']=='visit_start')&(~pd.isnull(df['주상병_truc']))][['case_id', 'starttime', '주상병_truc', '통합_진료구역']].reset_index()
    df_pdf = pd.read_csv(os.path.join(data_dir, 'activity_fre-hussh.csv'), index_col = ['주상병_truc', '통합_진료구역'])
    df_date = pd.concat([df_date,df_date.apply(lambda x: df_pdf.loc[x['주상병_truc'], x['통합_진료구역']], axis=1)], axis=1)
    df_date[[column for column in df_date.columns if '_pdf' in column]] = df_date[[column for column in df_date.columns if '_pdf' in column]].apply(lambda x: [eval(i) for i in x])

    for column in df_date.columns:
        if '_pdf' in column:
            df_date[column] = df_date[column].apply(lambda x: random.choices(list(x.keys()), list(x.values()), k=1)[0])
    df_v = df.groupby('case_id')['new_activity'].apply(list).reset_index()
    df_v.index = df_v['case_id']
    df_v = df_v.drop('case_id', axis=1)
    df_v = df_v.rename(columns = {'new_activity':'variant'})

    case_variant = df.pivot_table(index = 'case_id', columns = 'new_activity', aggfunc = 'size', fill_value = 0).reset_index()

    return df_date, df_v, case_variant

def find_nearest_variant(case_variant, df_date):
    cases = np.array(case_variant.drop('case_id', axis=1))
    variants = np.array(
        df_date[[column for column in df_date.columns if '_pdf' in column]]
    ).astype(int)
    print(variants)
    variants_list = []
    for i in range(variants.shape[0]):
        temp = np.sum((cases - variants[i])**2, axis=1)
        variants_list.append(random.choice(np.where(temp == np.min(temp))[0]))

    return variants_list


def prepare_simulation_data(data_dir='./app/data'):
    df_date, df_v, case_variant = load_inputs(data_dir)
    variants_list = find_nearest_variant(case_variant, df_date)
    df_date_re = pd.concat([df_date, df_v.loc[case_variant['case_id'].loc[variants_list]].reset_index(drop=True)], axis=1)
    pickle.dump(df_date_re, open('app/data/df_date_re.pkl', 'wb'))

    df_new = df_date_re.copy()
    df_new['starttime'] = pd.to_datetime(df_new['starttime'])
    df_new = df_new.sort_values(by='starttime')
    start_date = df_new['starttime'].iloc[0]
    cases = list(df_new.case_id)
    arrival_times = [(i - start_date).total_seconds() for i in sorted(list(df_new['starttime']))]
    variants = list(df_new.variant)


    event_duration = pickle.load(open(os.path.join(data_dir, 'event_duration_generated.pickle'), 'rb'))
    for i in ['visit_start', 'visit_end', '퇴실(귀가)', '퇴실(사망)', '퇴실(입원)', '퇴실(전원)']:
        event_duration[i] = {"value": [1], "pdf": [1]}

    resource_pdf = pickle.load(open(os.path.join(data_dir, 'activity_resource_generated.pickle'), 'rb'))
    trans_time = pd.read_csv(os.path.join(data_dir, 'transition_time_generated.csv'))
    trans_time.set_index(['new_activity', 'next_act'], inplace=True)

    return cases, start_date, arrival_times, variants, event_duration, resource_pdf, trans_time

def load_and_prepare_data(remove_activity=None, resequence_pair=None):
    df = pd.read_csv('app/data/masked_log.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    if remove_activity:
        print(remove_activity)
        df = df[df['new_activity'] != remove_activity].copy()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['starttime'] = pd.to_datetime(df['starttime'])
    df['endtime'] = pd.to_datetime(df['endtime'])
    df['주상병_two'] = df['주상병'].str[:1]
    df['주상병_three'] = df['주상병'].str[:2]

    df_date = df[(df['new_activity'] == 'visit_start') & (~pd.isnull(df['주상병_truc']))][['case_id', 'starttime', '주상병_truc', '통합_진료구역']].reset_index()
    df_pdf = pd.read_csv('app/data/activity_fre-hussh.csv', index_col=['주상병_truc', '통합_진료구역'])
    if remove_activity:
        df_pdf = df_pdf.drop(remove_activity+'_pdf', axis=1)
    df_date = pd.concat([df_date, df_date.apply(lambda x: df_pdf.loc[x['주상병_truc'], x['통합_진료구역']], axis=1)], axis=1)
    df_date[[c for c in df_date.columns if '_pdf' in c]] = df_date[[c for c in df_date.columns if '_pdf' in c]].applymap(eval)

    for column in df_date.columns:
        if '_pdf' in column:
            df_date[column] = df_date[column].apply(lambda x: random.choices(list(x.keys()), list(x.values()), k=1)[0])

    df_v = df.groupby('case_id')['new_activity'].apply(list).reset_index()
    df_v.index = df_v['case_id']
    df_v = df_v.drop('case_id', axis=1).rename(columns={'new_activity': 'variant'})

    case_variant = df.pivot_table(index='case_id', columns='new_activity', aggfunc='size', fill_value=0).reset_index()

    variants_list = find_nearest_variant(case_variant, df_date)
    df_date_re = pd.concat([df_date, df_v.loc[case_variant['case_id'].loc[variants_list]].reset_index(drop=True)], axis=1)
    
    if resequence_pair:
        pass

    start_date = df_date_re['starttime'].min()

    event_duration = pickle.load(open('app/data/event_duration_generated.pickle', 'rb'))
    for i in ['visit_start', 'visit_end', '퇴실(귀가)', '퇴실(사망)', '퇴실(입원)', '퇴실(전원)']:
        event_duration[i] = {'value': [1], 'pdf': [1]}

    resource_pdf = pickle.load(open('app/data/activity_resource_generated.pickle', 'rb'))
    trans_time = pd.read_csv('app/data/transition_time_generated.csv')
    trans_time.set_index(['new_activity', 'next_act'], inplace=True)

    return df, df_date_re, start_date, event_duration, resource_pdf, trans_time