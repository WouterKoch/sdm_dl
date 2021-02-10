import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow import keras

if __name__ == '__main__':
    target = pd.read_csv('Bombus/59.5.csv', index_col=['decimalLatitude', 'decimalLongitude'])
    df = pd.read_csv('Plantae/59.5.csv', index_col=['decimalLatitude', 'decimalLongitude'])

    df = pd.concat([df, target], axis=1).fillna(0).astype('int32').reset_index()
    ntargets = len(target.columns)
    ninput = len(df.columns)
    target = None

    df['decimalLatitude'] = (df['decimalLatitude'] - 45) / 45
    df['decimalLongitude'] = (df['decimalLongitude']) / 33
    model = keras.models.load_model("trained_model_binary.h5")

    df = pd.concat([df, pd.DataFrame(model.predict(x=df.to_numpy().astype('float32')) * 100)], axis=1)

    y_true = df.iloc[:, -(2 * ntargets):-ntargets]
    print(y_true.values)
    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_true,
                                                                                      df.iloc[:, -ntargets:] > .5)
    labels = df.columns.values[-(2 * ntargets):-ntargets]

    for l, label in enumerate(labels):
        print(label, ": recall=", recall[l], " - precision=", precision[l], " - support=", support[l])

    for species in range(0, ntargets):
        plt.figure(figsize=(20, 16))
        plt.subplot(211)
        plt.imshow(
            df.pivot(index=['decimalLatitude'], columns='decimalLongitude', values=df.columns.values[-(species + 1)]),
            interpolation='nearest')
        plt.title(df.columns.values[-(species + 1 + ntargets)] + " predicted")

        plt.subplot(212)
        plt.imshow(df.pivot(index=['decimalLatitude'], columns='decimalLongitude',
                            values=df.columns.values[-(1 + species + ntargets)]), interpolation='nearest')
        plt.title(df.columns.values[-(species + 1 + ntargets)] + " actual")
        plt.show()

        #

        for threshold in np.arange(.1, 1, .1):
            precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_true,
                                                                                              df.iloc[:,
                                                                                              -ntargets:] > threshold)
            print(df.columns.values[-(species + 1 + ntargets)], threshold, precision[species], recall[species],
                  fbeta_score[species])
