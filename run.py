from utils.simulator import Simulator
from args.test_arg import args_parser

if __name__ == '__main__':
    args = args_parser()
    simulator = Simulator(args)
    simulator.run()
