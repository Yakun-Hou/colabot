import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--model")
args = parser.parse_args()

checkpoint = torch.load(args.model, map_location='cpu')
checkpoint = {'net': checkpoint['net']}
torch.save(checkpoint, args.model)
