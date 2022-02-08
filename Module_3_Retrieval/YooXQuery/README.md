**Code for computing a retrieval clothing query on YOOX dataset.**


The `images` directory gets the full input image.

The `out` directory is where the output (YooX items) for the input query are stored.

In `models` is stored the pretrained model of MobilenetV2.

In `parsed` should be stored the parsed version of the input image.

The `ExtractParsed.py` file extracts the parsed clothes from the segmented image.

The `UtilsExtractParsed.py` file contains the utility code for generating the csv file for input and parsed.

The `palette.json` file is the palette list for extracting the segmentations.

In the `inputs` folder are automatically stored the inputs for the network (cropped item) after the extraction from the segmented file.

The only requirements for the user is to store in the `images` folder the input image to retrieve and in parsed the respective same segmented image.

The `ComposeDataset.py` file contains the dataloader for the YOOX dataset.

This code could compute a filtering step before computing the retrieval. This is useful if the computation on the entire dataset (almost 100.000 images) 
takes too time.

The dataset relative path should be '../DatasetFolder/yoox' from the current folder.

The csv file to manage the Yoox Dataset should have this relative path from the current folder '../DatasetFolder/yoox/final_yoox-{category}.csv', 
with 'lower_body' or 'upper_body' as category.

the weights path should be : 'data/model-weights-{category}.pth', with 'lower_body' or 'upper_body' as category.

To run the code on the inputs stored in the `images` folder, edit the `config.json` with the desired hyperparameters and run 'python3 main.py'.

parameters to set in `config.json`: 

part : is the instruction about the segmented clothes (Upper-clothes o Pants).
category : is the two categories of dataset (upper_body or lower_body).
fields : is an instruction for the preprocessing pipeline, specifies if should be used the pipeline for bermudas or an other generic clothes.
filtering : specifies if is necessary to filter the dataset.
filter_for : if filtering is True the 'filter_for' field is used, else is ignored.
recall : number of the first retrieved items.

YooX Dataset hyerarchy folder:

DatasetFolder:
    -yoox:
        -lower_body:
            -denim_denim-bermudas
            -denim_denim-capris
            -denim_denim-pants
             ecc...
        -upper_body:
            -shirts_blouses
            -tops--tees_tank-tops
            -shirts_shirts
             ecc...
        -final_yoox-lower_body.csv
        -final_yoox-upper_body.csv


ALL THE CODE SHOULD BE RUNNED ON A LINUX SYSTEM.