import argparse

from model import *


def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True) 
    args = parser.parse_args() 

    return args.file

def main():
    features_file = Parser()
    model = train_model('dataset.xlsx')
    predictions = predict(features_file, model)

    print(predictions)


if __name__ == '__main__':
    main()