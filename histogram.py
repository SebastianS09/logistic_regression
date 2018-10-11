import pandas as pd
import numpy as np
import argparse, glob, sys
from describe import calculate_metrics
import matplotlib.pyplot as plt

def hist_plot(data,houses):

    columns = list(data.values())[0].keys().values
    hist_data = {}
    for j in columns:
        out=pd.DataFrame()
        for i in data.keys():
            out = pd.concat([out,data[i][j]],axis=1)
        out.columns = data.keys()
        hist_data[j] = out

    N = len(data.keys())
    ind = np.arange(N)
    width = 0.5
    i=1

    for column in columns:
        f = plt.figure(i)
        p1 = plt.bar(ind,hist_data[column].loc['25%'], width)
        p2 = plt.bar(ind,hist_data[column].loc['50%'], width, bottom=hist_data[column].loc['25%'])
        p3 = plt.bar(ind,hist_data[column].loc['75%'], width, bottom=hist_data[column].loc['25%']+hist_data[column].loc['50%'])
        p4 = plt.bar(ind,hist_data[column].loc['max'], width, bottom=hist_data[column].loc['25%']+hist_data[column].loc['50%']+hist_data[column].loc['75%'])
        plt.title(column)
        plt.legend((p1[0],p2[0],p3[0],p4[0]), ('1st quart.', '2nd', '3rd', '4th'))
        plt.xticks(ind, houses)
        plt.show(block=True)
        i+=1

def main():
    "Entrypoint for plotting histogram."

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
            data = pd.read_csv(file[0], encoding='latin1', index_col = 'Index').dropna()
            columns = data._get_numeric_data().columns
            houses = data['Hogwarts House'].dropna().unique()

        except:
            print("unable to read csv. Csv must be Hogwarts Data")
            sys.exit()

    results = {}
    metrics_lines = ['min','25%', '50%', '75%', 'max']

    if len(houses) != 4:
        print("Wrong number of houses in dataset, is this Hogwarts?")
        sys.exit()
    try:
        for house in houses:
            house_data = calculate_metrics(data.loc[data['Hogwarts House'] == house],columns).loc[metrics_lines]

            for name in columns:
                house_data[name] = (house_data[name] - house_data[name].loc['min'])/(house_data[name].loc['max']-house_data[name].loc['min'])

            results[house] = house_data.diff().dropna()

        hist_plot(results,houses)

    except:
        print("Data seems incorrect: go see your head teacher")

if __name__ == "__main__":
    main()
