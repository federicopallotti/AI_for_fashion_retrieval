**Code for testing the Triplet Loss using a pretrained Mobilenet V2.**

The main is the `main.py` file, parameters for the test are stored in the configuration file `config.json`.

The `neptuneLogger.py` file contains specific code for logging and test monitoring, using the online tool "neptuneAI".

Weights are cached in the `data` folder. The Triplet Loss code is in the `utils.py` file.

`requirements.txt` provides all the required packages.

`ComposeDataset.py` contains the dataloader for feeding the dataset to the network.

The backbone and the overall model are in `models/GarnmentsNet.py`.

In order to run the code is necessary to edit the `config.json` file with desired hyperparameters and run 'python3 main.py' in a cuda available environment.

The original dataset needs to be modified in order to substitute the mannequin images with their respective cropped item.

The Dataset for the testing procedure needs to have this relative path '../DatasetFolder/DatasetCV'

The cvs files for testing should have this relative path : '../DatasetFolder/DatasetCV/{category}/test_pairs_paired.txt'

THIS CODE SHOULD BE RUNNED ON A LINUX SYSTEM

