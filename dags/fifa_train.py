from datetime import timedelta
from textwrap import dedent
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import time


class BashTaskBuilder:
    def __init__(self, task_id, depends_on_past=False):
        self.__task_id         = task_id
        self.__depends_on_past = depends_on_past
        self.__script          = ''
        self.__content         = """
        eval "$(conda shell.bash hook)"
        conda activate datitos
        cd {{ var.value.project_path }}
        echo "---------------------------------------"
        echo "| Conda Env: $CONDA_DEFAULT_ENV"
        echo "| Task: {{ task.task_id }}"\n
        """

    def __append(self, value): self.__content += '{}\n'.format(value)
    def __echo(self, value): self.__append('echo "| {}"'.format(value))
    def __separator(self): self.__append('echo "---------------------------------------"')
    
    def var_fields(self, properties):
        for (name, value) in properties.items():
            self.var_field(name, value)
        return self

    def script(self, script):
        self.__script = script
        return self

    def message(self, message):
        self.__echo(message)
        return self

    def var_field(self, name, value):
        self.__echo(name + ': {{ var.value.' + value + '}}')
        return self

    def field(self, name, value):
        self.__echo('{}: {}'.format(name, value))
        return self

    def build(self):
        self.__separator()
        self.__echo('')
        self.__echo('')
        self.__separator()
        self.__echo('SCRIPT')
        self.__separator()

        if self.__script:
            self.__append(self.__script)
        else:
            self.__echo('Add script here...')

        self.__separator()
        self.__echo('')
        self.__echo('')
        return BashOperator(
            task_id         = self.__task_id,
            depends_on_past = self.__depends_on_past,
            bash_command    = dedent(self.__content)
        )

with DAG(
    'train_and_test_fifa',
    default_args = {
        'owner': 'adrian',
        'depends_on_past':  False,
        'email': ['adrianmarino@gmail.com'],
        'email_on_failure': False,
        'email_on_retry':   False,
        'retries': 5,
        'retry_delay': timedelta(seconds=10)
    },
    description       = 'Fifa: Train model and generate test result',
    schedule_interval = '0 0 */3 * * *',
    start_date        = days_ago(2),
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
                'Device'       : 'train_device',
                'Study'        : 'train_optuna_study',
                'DB URL'       : 'train_optuna_db_url',
                'Seeds Count:' : 'report_seeds_count',
                'Folds'        : 'report_folds'
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

    # Workflow...
    train_1 = create_train_tasks(1)
    train_2 = create_train_tasks(2)
    train_3 = create_train_tasks(3)
    train_4 = create_train_tasks(4)
    train_5 = create_train_tasks(5)
    optimization_report = create_optimization_report_task()
    test_model = create_test_task()

    [train_1, train_2, train_3, train_4, train_5] >> optimization_report >> test_model