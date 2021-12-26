from datetime import timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.models.baseoperator import chain
from dag_utils import BashTaskBuilder
from slack_utils import task_fail_slack_alert

with DAG(
    'FIFA',
    default_args = {
        'owner': 'adrian',
        'depends_on_past':  False,
        'email': ['adrianmarino@gmail.com'],
        'on_failure_callback': task_fail_slack_alert,
        'retries': 5,
        'retry_delay': timedelta(seconds=10)
    },
    description       = 'Fifa: Train model and generate test result',
    schedule_interval = '0 0 */1 * *',
    start_date        = days_ago(0),
    catchup           = False,
    tags              = ['fifa']
) as dag:
    def create_train_tasks(worker_id):
        return BashTaskBuilder('train_worker_{}'.format(worker_id)) \
            .var_fields({
                'Device'  : 'train_device',
                'Cuda mem': 'train_cuda_process_memory_fraction',
                'Folds'   : 'train_folds',
                'Study'   : 'train_optuna_study',
                'Trials'  : 'train_optuna_trials',
                'DB URL'  : 'train_optuna_db_url',
                'Timeout' : 'train_optuna_timeout'
            }) \
            .script("""
            python bin/train.py \
                --device {{ var.value.train_device }} \
                --cuda-process-memory-fraction {{ var.value.train_cuda_process_memory_fraction }} \
                --folds {{ var.value.train_folds }} \
                --study {{ var.value.train_optuna_study }} \
                --db-url {{ var.value.train_optuna_db_url }} \
                --trials {{ var.value.train_optuna_trials }} \
                --timeout {{ var.value.train_optuna_timeout }}
            """) \
            .build()

    def create_optimization_report_task():
        return BashTaskBuilder('optimization_report') \
            .var_fields({
                'Device'      : 'train_device',
                'Study'       : 'train_optuna_study',
                'DB URL'      : 'train_optuna_db_url',
                'Seeds Count' : 'report_seeds_count',
                'Folds'       : 'report_folds'
            }) \
            .script("""
            python bin/optmimization_report.py \
                --device {{ var.value.train_device }} \
                --folds {{ var.value.train_folds }} \
                --study {{ var.value.train_optuna_study }} \
                --db-url {{ var.value.train_optuna_db_url }} \
                --seeds-count {{ var.value.report_seeds_count }} \
                --folds {{ var.value.report_folds }}
            """) \
            .build()

    def create_test_task():
        return BashTaskBuilder('test_model') \
            .var_fields({
                'Device'       : 'train_device',
                'Study'        : 'train_optuna_study',
                'DB URL'       : 'train_optuna_db_url'
            }) \
            .script("""
            python bin/test_model.py \
                --study {{ var.value.train_optuna_study }} \
                --db-url {{ var.value.train_optuna_db_url }} \
                --device {{ var.value.train_device }}
            """) \
            .build()

    # Create all tasks...
    workers_count = int(Variable.get('train_workers_count'))

    train_workers       = [create_train_tasks(id) for id in range(1, workers_count+1)]
    optimization_report = create_optimization_report_task()
    test_model          = create_test_task()

    # Workflow...
    train_workers >> optimization_report >> test_model
