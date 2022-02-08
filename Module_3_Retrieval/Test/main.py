from testRecallBatchMultiple import test
import json
import torch


def main():
    optims = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    bools = {"True": True, "False": False}

    f = open('config.json', 'r')
    hyperparameters = json.load(f)

    test("", hyperparameters["test_parameters"]["iterations"], train_parameters=hyperparameters["trained_on"],
         hyper=hyperparameters["test_parameters"], start=hyperparameters["test_parameters"]["start"])


if __name__ == '__main__':
    main()
