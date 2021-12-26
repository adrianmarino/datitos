from textwrap import dedent
from airflow.operators.bash import BashOperator

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
