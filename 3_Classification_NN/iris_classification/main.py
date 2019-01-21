import argparse
from train import train
from test import test

parameter .....

if parameter['mode'] == train:
    train(parameter)
else:
    test(parameter,model_save_path)
