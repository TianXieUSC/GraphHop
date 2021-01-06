import argparse

import model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='cora', type=str, help='select datasets.')
parser.add_argument("--num_per_class", default=20, type=int, help='select number of labeled examples per class.')
parser.add_argument("--batch_prop", default=512, type=int, help='select batch number')
parser.add_argument("--temperature", default=.1, type=float, help='select temperature')
parser.add_argument("--alpha", default=1., type=float, help='select alpha')
parser.add_argument("--beta", default=1., type=float, help='select beta')
parser.add_argument("--W1", default=.5, type=float)
parser.add_argument("--W2", default=.5, type=float)
args = parser.parse_args()

DATASET = args.dataset
TEMPERATURE = float(args.temperature)
ALPHA = float(args.alpha)
BETA = float(args.beta)
W1 = float(args.W1)
W2 = float(args.W2)
NUM_PER_CLASS = int(args.num_per_class)
BATCH_PROP = int(args.batch_prop)

model
