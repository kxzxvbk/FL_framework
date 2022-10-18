from utils.epoch_simulator import Simulator
from args.moco_arg1 import args_parser

if __name__ == '__main__':
    args = args_parser()
    simulator = Simulator(args)
    simulator.run()
