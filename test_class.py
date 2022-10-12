class Father:
    def __init__(self):
        pass

    def register_buffer(self, name, value):
        self.__dict__[name] = value


class Son(Father):
    Father.register_buffer(Son, 'nop', -1)
    def __init__(self):
        pass

    @staticmethod
    def get_value():
        return Son.nop

    @staticmethod
    def set_value(inte):
        Son.nop = inte

    def update(self):
        Son.nop += 1



if __name__ == '__main__':
    s = Son()
    s1 = Son()
    print(s.get_value())
    print(s1.get_value())
    print(Son.value)

    s.value = 2
    s.update()
    print(s.get_value())
    print(s1.get_value())
    print(Son.value)

