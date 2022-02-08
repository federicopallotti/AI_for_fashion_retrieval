from YOOXTester import YOOXTester
import platform
from ExtractParsed import make
import os
import json


def make_test(recall, category, filtering, filter_for):

    if platform.system() == 'Windows':
        root = "D:\\datasetCV_nomanichini_classificato"
        imgs_path = os.path.join(root, category)
        weights_path = "data\\model-weights-lower_body.pth"
    elif platform.system() == 'Linux' or platform.system() == 'Darwin':
        root = '../DatasetFolder/yoox'
        imgs_path = os.path.join(root, category)
        weights_path = "data/model-weights-lower_body.pth"

    csv_path = '../DatasetFolder/yoox' + os.sep + 'final_yoox-' + category + '.csv'
    tester = YOOXTester(root, imgs_path, category, recall, platform.system(), weights_path, filtering, filter_for, csv_path)
    tester(verbose=False)


def main():
    '''
    PARAMETERS TO SET :
    part : is the instruction about the segmented clothes (Upper-clothes o Pants).
    category : is the two categories of dataset (upper_body or lower_body).
    fields : is an istruction for the preprocessing pipeline, specifies if should be used the pipeline for bermudas or an other generic clothes.
    filtering : specifies if is necessary to filter the dataset.
    filter_for : if filtering is True the 'filter_for' field is used, otherwise is ignored.
    '''

    bools = {"True": True, "False": False}

    f = open('config.json', 'r')
    hyperparameters = json.load(f)

    part = hyperparameters['part']
    category = hyperparameters['category']
    fields = hyperparameters['fields']
    filtering = bools[hyperparameters['filtering']]
    filter_for = hyperparameters["filter_for"]

    make(category=category, Fields=fields, part=part)
    make_test(hyperparameters['recall'], category, filtering, filter_for)


if __name__ == '__main__':
    main()