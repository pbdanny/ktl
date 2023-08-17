
from . import etl


class Test2(etl.Test):
    def __init__(self):
        super().__init__()

    def show_again(self):
        etl.Test.show_hello(self)
        etl.Test.show_hello(self)


def test():
    print("hello")
