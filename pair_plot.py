import pandas as pd
import numpy as np
import argparse, glob, sys
import seaborn as sns
import matplotlib.pyplot as plt

def pair_plot(data, hue):
    sns.pairplot(data, hue=hue, diag_kind='kde')
    plt.show(block = True)


def main():
    "Entrypoint for plotting pair_plot."

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--file',
        help='Csv to plot. Must be Hogwarts data',
        default='dataset_train.csv')

    args = parser.parse_args()

    file = glob.glob(args.file)

    if len(file)==0:
        print("No such csv file to plot")

    else:
        try:
            data = pd.read_csv(file[0], encoding='latin1', index_col = 'Index')
            columns = data._get_numeric_data().columns
            houses = data['Hogwarts House'].dropna().unique()

        except:
            print("unable to read csv. Csv must be Hogwarts Data")
            sys.exit()

    if len(houses) != 4:
        print("Wrong number of houses in dataset, is this Hogwarts?")
        sys.exit()

    #try:
    select_col = list(columns.get_values())
    select_col.append('Hogwarts House')
    pair_plot(data[select_col],'Hogwarts House')

   # except:
    #    print("Data seems incorrect: go see your head teacher")

if __name__ == "__main__":
    main()
