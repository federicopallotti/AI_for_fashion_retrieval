**Code for modifing the original yu-vton dataset, in order to train the network**

Each mannequins images of the dataset is replaced with the relative parsed item.

!WARNING : RUNNING THIS CODE WILL MODIFY DIRECTLY THE INPUT DATASET REPLACING THE FULL BODY IMAGES.

`config.json` file contains the parameters for computing this code:

{
  "verbose" : "True", --> if verbose execution or not

  "dataset" : "train", --> the dataset to modify (train/test)

  "category" : "lower_body", --> The category to modify (lower_body/upper_body)

  "root_dir" : "C:\\Users\\pales\\Desktop\\yu-data" --> the path of the dataset
 }

the dataset folder hierarchy needs to be in its original form as described below, provided by the Phd Marcella Cornia.

legend:
+ --> directory
- --> file

+yu-vton:
    -palette.txt --> parser palette
    -yu-vton.csv --> principal csv
    +lower_body:
        +images --> contains images for lower body
        +parsing --> contains already parsed mannequins
        -train_pairs.txt --> train csv
        -test_pairs_paired.txt --> test csv
    +upper_body:
        +images --> contains images for upper body
        +parsing --> contains already parsed mannequins
        -train_pairs.txt --> train csv
        -test_pairs_paired.txt --> test csv    
    

These are only the required files and folders.


The `parseExtractor.py` file is the code for computing this task.

The `extractorUtils.py` contains some useful utility methods.




