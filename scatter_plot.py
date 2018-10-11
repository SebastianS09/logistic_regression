import pandas as pd
import numpy as np
import argparse, glob, sys
import matplotlib.pyplot as plt
from describe import calculate_metrics

def get_correlation(data):
    columns = data._get_numeric_data().columns
    correlations = pd.DataFrame(columns=columns, index= columns)
    results = calculate_metrics(data, columns)

    for col in correlations.columns:
        for row in correlations.index:

            new_df1 = pd.DataFrame(data[col])
            new_df1.columns = ['Feature1']
            new_df2 = pd.DataFrame(data[row])
            new_df2.columns = ['Feature2']

            new_df1['Mean'] = results[col].loc['Mean']
            new_df2['Mean'] = results[row].loc['Mean']

            new_df1['difference1'] = new_df1['Feature1'] - new_df1['Mean']
            new_df2['difference2'] = new_df2['Feature2'] - new_df2['Mean']

            new_df = pd.concat([new_df1, new_df2], axis=1)
            new_df = new_df.dropna().reset_index(drop=True)

            correlations[col].loc[row] = sum(new_df['difference1']*new_df['difference2'])/np.sqrt(sum(new_df['difference1']**2)*sum(new_df['difference2']**2))
    return correlations

def scatter_plot(data):
    fig = plt.figure()
    correlations = get_correlation(data)

    for column in correlations.columns:
        plt.scatter(np.arange(0, len(correlations.index)), list(correlations[column]))
        plt.legend(correlations.index, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(np.arange(0, len(correlations.columns)),list(correlations.columns), rotation = 45, horizontalalignment='right')
        plt.title('Correlation of the different variables', size = 16)

    plt.show(block=True)

def main():
    "Entrypoint for plotting scatter_plot."

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
            houses = data['Hogwarts House'].dropna().unique()

        except:
            print("unable to read csv. Csv must be Hogwarts Data")
            sys.exit()

    if len(houses) != 4:
        print("Wrong number of houses in dataset, is this Hogwarts?")
        sys.exit()
    try:
        scatter_plot(data)
    except:
        print("Data seems incorrect: go see your head teacher")

if __name__ == "__main__":
    main()
