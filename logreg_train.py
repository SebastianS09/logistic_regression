import pandas as pd
import numpy as np
import argparse, glob, sys
from tqdm import tqdm

class preprocess:
    
    def __init__(self):
        self=self
    
    def __cleaning(self, path, column, dataset='all'):
        data = pd.read_csv(path)
        if dataset == 'test':
            data = data.drop(['Index',str(column)], axis=1)
            final_data = data.dropna().reset_index(drop=True)
        else:
            data = data.dropna().reset_index(drop=True)
            data = data.drop(['Index'], axis=1)
            spread = pd.get_dummies(data[str(column)])
            final_data = pd.concat([data, spread], axis=1)
        return final_data

    def __minimum(self, frame):
        m = frame.iloc[0]
        for i in np.arange(1, len(frame)):
            if m > frame.iloc[i]: m = frame.iloc[i]
        return m

    def __maximum(self, frame):
        m = frame.iloc[0]
        for i in np.arange(1, len(frame)):
            if m < frame.iloc[i]: m = frame.iloc[i]
        return m

    def __selection(self, data):
        inter = data._get_numeric_data()
        final_data = pd.DataFrame(index=data.index)
        for column in list(inter.columns):
            final_data[column] = (inter[column] - self.__minimum(inter[column]))/(self.__maximum(inter[column]) - self.__minimum(inter[column]))
        return final_data

    def __add_intercept(self, data):
        cols = data.columns
        data['Intercept'] = 1
        return data.filter(['Intercept'] + list(cols))

    def preprocess(self, path, column, dataset="all"):
        data = self.__cleaning(path, column, dataset)
        inter_data = self.__selection(data)
        final_data = self.__add_intercept(inter_data)
        return final_data

class LogReg:
    
    def __init__(self, lr, iterations):
        self.lr = lr
        self.iterations = iterations

    def __sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        
        for i in tqdm(np.arange(0,self.iterations)):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = 1/y.size * np.dot(X.T, (h-y))
            self.theta -= self.lr*gradient
    
    def probability(self, X):
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self,X):
        return np.random.binomial(1, self.__sigmoid(np.dot(X, self.theta)))


def weights(data, outputs, lr, iterations):
    inter = data.drop(outputs, axis=1)
    results = pd.DataFrame(index=np.arange(0, len(list(inter.columns))))
    models = {}
    for target in outputs:
        models[target] =  LogReg(lr = lr, iterations = iterations)
        print("fitting model for: "+str(target))
        models[target].fit(inter, data[target])
        partial = pd.DataFrame(models[target].theta)
        partial.columns = ['Weights '+str(target)]
        results = pd.concat([results, partial], axis=1)   
    return {'values': results, 'models': models}


def accuracy(data, models):
    outputs = list(models.keys())
    output = pd.DataFrame(data.filter(outputs, axis=1).idxmax(axis=1))
    inter = data.drop(outputs, axis=1)
    probabilities = pd.DataFrame(index=np.arange(0, len(list(inter.columns))))

    for model in models.items():
        proba = pd.DataFrame(model[1].probability(inter))
        proba.columns = [model[0]]
        probabilities = pd.concat([probabilities, proba], axis=1)

    probabilities['Results'] = probabilities.idxmax(axis=1)
    probabilities['Actual'] = output
    
    count = 0
    for i in np.arange(0,len(probabilities['Results'])):
        if probabilities['Results'].iloc[i]==probabilities['Actual'].iloc[i]:
            count += 1
    return count/len(probabilities['Results'])


def training(path,output_column, lr=0.01, iterations=10000):
    pre = preprocess()
    data = pre.preprocess(path, output_column)
    outputs = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    weight = weights(data, outputs, lr, iterations)

    weight['values'].to_csv('logreg_weights.csv')
    print('\n Accuracy on training set: '+str(round(accuracy(data, weight['models'])*100,1))+'% \n')

def is_number(s):
    try:
        complex(s) # for int, long, float and complex
    except ValueError:
        return False

    return True

def main():
    "Entrypoint for training."

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--file',
        help='Input for logistic regression. Must be Hogwarts Data',
        default='dataset_train.csv')

    parser.add_argument(
        '--learning-rate',
        help='Learning rate for logistic regression. Default is 0.01',
        default=0.01)

    parser.add_argument(
        '--iterations',
        help='Iterations of logistic regression. Default is 10 000',
        default=10000)

    args = parser.parse_args()

    file = glob.glob(args.file)

    assert is_number(args.learning_rate), "learning rate must be a number"
    assert is_number(args.iterations), "iterations must be a number"

    lr = float(args.learning_rate)
    iterations = round(float(args.iterations))

    if len(file)==0:
        print("No such csv file(s) to run")

    else:
        try:
            path = file[0]
        except:
            print("unable to read csv")
            sys.exit()

        training(path, 'Hogwarts House', lr, iterations)

if __name__ == "__main__":
    main()
