import os
import random


def gen_train_txt(txt_path):
    f = open(txt_path, 'w')
    for i in range(2000):
        x = random.random()
        y = random.random()
        line = str(x) + '***' + str(y) + '\n'
        print(line)
        f.write(line)
    f.close()


if __name__ == '__main__':
    gen_train_txt('./data.txt')