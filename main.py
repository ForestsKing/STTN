from exp.exp import EXP
from utils.setseed import set_seed

if __name__ == '__main__':
    set_seed(42)

    exp = EXP()
    # exp.train()
    exp.test()
