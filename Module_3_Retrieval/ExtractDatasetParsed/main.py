from parseExtractor import Extractor
import json

def main():
    f = open('config.json', 'r')
    hyper = json.load(f)
    ex = Extractor(hyper)
    ex.perform()


if __name__ == "__main__":
    main()