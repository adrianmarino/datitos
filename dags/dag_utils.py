from textwrap import dedent

from airflow.operators.bash import BashOperator

from text_box_builder import TextBoxBuilder


class BashTaskBuilder:

    def __init__(self, task_id, depends_on_past=False):
        self.__task_id = task_id
        self.__depends_on_past = depends_on_past
        self.__script = ''
        self.__variables = TextBoxBuilder('Variables') \
            .line('Conda Env: $CONDA_DEFAULT_ENV') \
            .line('Task: {{ task.task_id }}')
        self.fields = []
        self.__content = \
"""
eval "$(conda shell.bash hook)"
conda activate {{ var.value.conda_env }}
cd {{ var.value.project_path }}
"""

    def __append(self, value):
        self.__content += '{}\n'.format(value)

    def var_fields(self, properties):
        for (name, value) in properties.items():
            self.var_field(name, value)
        return self

    def script(self, script):
        self.__script = script
        return self

    def var_field(self, name, value):
        self.__variables.line(name + ': {{ var.value.' + value + '}}')
        return self

    def field(self, name, value):
        self.__variables.line('{}: {}'.format(name, value))
        return self

    def build(self):
        self.__content += f'{self.__variables.build()}'

        if self.__script:
            self.__content += self.__script
        else:
            self.__content += 'echo "Missing script to execute!"'

        return BashOperator(
            task_id=self.__task_id,
            depends_on_past=self.__depends_on_past,
            bash_command=dedent(self.__content)
        )
