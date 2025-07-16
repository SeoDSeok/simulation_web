from flask import Blueprint, render_template, request, flash, redirect, url_for
from app.simulator.loader import prepare_simulation_data
from app.simulator.engine import Simulator, Timer
from app.simulator.formatter import format_result
import pandas as pd
import os

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return redirect(url_for('main.upload_inputs'))

@main.route('/upload_inputs', methods=['GET', 'POST'])
def upload_inputs():
    if request.method == 'POST':
        event_log = request.files.get('event_log')
        activity_freq = request.files.get('activity_freq')
        event_duration = request.files.get('event_duration')
        activity_resource = request.files.get('activity_resource')
        transition_time = request.files.get('transition_time')

        save_path = 'app/data'
        os.makedirs(save_path, exist_ok=True)

        event_log.save(os.path.join(save_path, 'Event_log_real.csv'))
        activity_freq.save(os.path.join(save_path, 'activity_fre-hussh.csv'))
        event_duration.save(os.path.join(save_path, 'event_duration_hussh_80.pickle'))
        activity_resource.save(os.path.join(save_path, 'activity_resource_detailed.pickle'))
        transition_time.save(os.path.join(save_path, 'transition_time_hussh.csv'))

        flash('파일이 성공적으로 업로드되었습니다.')
        return redirect(url_for('main.run_simulation'))

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
