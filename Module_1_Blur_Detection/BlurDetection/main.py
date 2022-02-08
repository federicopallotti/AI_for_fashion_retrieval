from BlurDetection import extractor
import json


def main():
    f = open('config.json', 'r')
    params = json.load(f)

    extractor(params['path'], params['mode'],params['kernel_n'], params['parameters_mode1'],
              params['parameters_mode2'], params['threshold'], params['format'])

if __name__ == '__main__':
    main()