import pandas as pd
import numpy as np
import argparse, glob, sys, distutils.util


class describing:

    def __init__(self):
        self = self

    def __counter(self, X, col):
        Y = pd.DataFrame(X.filter([str(col)]))
        Y.columns = ['col']
        Y = Y.dropna().reset_index(drop=True)
        Y['One'] = 1
        return sum(Y['One'])

    def __meaner(self,X,col):
        counter = self.__counter(X, col)
        if counter == 0:
            return np.nan
        else:
            return sum(X[col].dropna())/counter
    def count(self, X, col):
        return self.__counter(X,col)

    def mean(self,X,col):
        return self.__meaner(X,col)

    def std(self, X, col):
        New = pd.DataFrame(X[col].dropna().reset_index(drop=True))
        std = pd.DataFrame((New[col] - self.__meaner(New,col))**2)
        counter = self.__counter(X, col)
        if counter == 0:
            return np.nan
        else:
            return np.sqrt(1/(self.__counter(std,col))*sum(std[col]))

    def minimum(self, X, col):
        Y = pd.DataFrame(X[col])
        m = Y[col].iloc[0]
        for i in np.arange(1, len(Y[col])):
            if (m > Y[col].iloc[i]): m = Y[col].iloc[i]
        return m

    def maximum(self, X, col):
        Y = pd.DataFrame(X[col])
        ma = Y[col].iloc[0]
        for i in np.arange(1, len(Y[col])):
            if (ma < Y[col].iloc[i]): ma = Y[col].iloc[i]
        return ma

    def quartile(self,X,col,percentage):
        quart = pd.DataFrame(X[col].dropna().reset_index(drop=True))
        quart = quart.sort_values(by=[col], ascending=True, axis=0).reset_index(drop=True)
        return quart[col].iloc[int((len(quart[col])*percentage/100))]

def calculate_metrics(data,columns):
    results = pd.DataFrame(index=['Count','Mean','std','min','25%','50%','75%','max'])
    for name in columns:
                metrics = []
                descr = describing()

                metrics.append(descr.count(X = data, col = name))
                metrics.append(descr.mean(X = data, col = name))
                metrics.append(descr.std(X = data, col = name))
                metrics.append(descr.minimum(X = data, col = name))
                metrics.append(descr.quartile(X = data, col = name, percentage=25))
                metrics.append(descr.quartile(X = data, col = name, percentage=50))
                metrics.append(descr.quartile(X = data, col = name, percentage=75))
                metrics.append(descr.maximum(X = data, col = name))

                metrics = pd.DataFrame(metrics)
                metrics.columns = [name]
                metrics.index = ['Count','Mean','std','min','25%','50%','75%','max']

                results = pd.concat([results, metrics], axis = 1)
    return results

def main():
    "Entrypoint for describing."

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--files',
        help='Csv files to describe. Default is all files in folder',
        default='*.csv')

    parser.add_argument(
        '--silent',
        help='Write output to new csv file in current directory. Values are True / False',
        default='False')

    args = parser.parse_args()

    files = glob.glob(args.files)

    try:
        silent = bool(distutils.util.strtobool(args.silent))
    except:
        print('--silent takes True / False as input. Setting to silent mode \n')
        silent = True

    if len(files)==0:
        print("No such csv file(s) to describe")

    else:

        for i in range(len(files)):
            try:
                data = pd.read_csv(files[i], encoding='latin1', index_col='Index').dropna(1,"all")
            except:
                print("unable to read csv")
                sys.exit()
            columns = data._get_numeric_data().columns

            results = calculate_metrics(data,columns)
            if silent:
                results.to_csv("description_"+files[i])
            print(results)

if __name__ == "__main__":
    main()