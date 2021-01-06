import argparse

import model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='cora', type=str)
parser.add_argument("--num_per_class", default=2, type=int)
parser.add_argument("--batch_prop", default=1., type=float)
parser.add_argument("--temperature", default=100., type=float)
parser.add_argument("--alpha", default=10., type=float)
parser.add_argument("--beta", default=.0, type=float)
parser.add_argument("--W1", default=.1, type=float)
parser.add_argument("--W2", default=.1, type=float)
args = parser.parse_args()

DATASET = args.dataset
TEMPERATURE = float(args.temperature)
ALPHA = float(args.alpha)
BETA = float(args.beta)
W1 = float(args.W1)
W2 = float(args.W2)
NUM_PER_CLASS = int(args.num_per_class)
BATCH_PROP = float(args.batch_prop)

model
