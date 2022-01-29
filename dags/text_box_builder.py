from bunch import Bunch

DEFAULT_CHARACTERS = Bunch(
    separator=Bunch(first=['┏', '┓'], intermediate=['┣', '┫'], last=['┗', '┛'], line='┃'),
    char='━'
)


class OutputBuilder:
    def __init__(self, max_with, characters=DEFAULT_CHARACTERS):
        self.__output = ""
        self.__max_with = max_with
        self.__offset = 2
        self.__separator_count = 0
        self.__characters = characters

    def echo(self, value):
        line = f'{self.__characters.separator.line} {value}'
        line += f'{(self.__max_with - len(line) + self.__offset) * " "} {self.__characters.separator.line}'

        self.__output += f'echo "{line}"\n'
        return self

    def __sep_pp(self, last):
        if last:
            return self.__characters.separator.last
        if self.__separator_count == 0:
            return self.__characters.separator.first
        else:
            return self.__characters.separator.intermediate

    def separator(self, last=False):
        pp = self.__sep_pp(last)
        self.__output += f'echo "{pp[0]}{self.__characters["char"] * (self.__max_with + self.__offset)}{pp[1]}"\n'
        self.__separator_count += 1

        return self

    def build(self):
        self.separator(True)
        return self.__output


class TextBoxBuilder:
    def __init__(self, tittle):
        self.__lines = []
        self.__tittle = tittle

    def line(self, line):
        if line:
            self.__lines.append(line)
        return self

    def __max_width(self):
        return len(max(self.__lines + [self.__tittle], key=lambda it: len(it)))

    def build(self):
        output = OutputBuilder(self.__max_width()) \
            .separator() \
            .echo(self.__tittle) \
            .separator()

        [output.echo(line) for line in self.__lines]

        return output.build()
