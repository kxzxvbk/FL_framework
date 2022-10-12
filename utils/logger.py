import os
import time


class Logger:
    def __init__(self, path):
        self.path = os.path.join(path, 'txt_file.txt')
        if os.path.exists(path):
            # raise OSError('Duplicated logging path!')
            pass
        if not os.path.exists(path):
            os.makedirs(path)
        with open(self.path, 'w') as f:
            f.write('Created at ' + time.asctime(time.localtime(time.time())) + '\n')

    def logging(self, s):
        print(s)
        with open(self.path, mode='a') as f:
            f.write('[' + time.asctime(time.localtime(time.time())) + ']    ' + s + '\n')
