import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten


def regression(pred = False):

    # On récupère le dataFrame
    df = pd.read_csv("train.csv")

    # Colonne cible
    target_col = 'LABEL'

    # On encode les variables en chaine de caractères : les 3 labels
    labelencoder = LabelEncoder()
    df[target_col] = labelencoder.fit_transform(df[target_col])


    # On isole la colonne cible et on la supprime
    y = df[target_col]
    df.drop([target_col], axis=1, inplace=True)


    # On crée le jeu de tests et d'entraînement
    print("## Création des données de train et de test -- DEBUT ##\n")

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


    # On normalise les données
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print("## Création des données de train et de test -- FIN ET NORMALISE ##\n")


    # Création des modèles
    

    def build_lstm_model(neurons1, neurons2, optimizer='adam'):
        model = Sequential()

        model.add(Dense(units=neurons1, activation='relu'))
        
        model.add(Dense(3, activation='softmax'))

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
        return model


    tf.random.set_seed(42)
    neurons1 = 128
    neurons2 = 64
    epochs = 20
    batch_size = 32
    optimizer = 'adam'


   
    bestAcc = 0
    bestModel = build_lstm_model(neurons1=128, neurons2=16,  optimizer=optimizer)

    # Choix des meilleurs paramètres
    for i in tqdm(np.linspace(8, 512, 30)):        

        model = build_lstm_model(neurons1=i, neurons2=0, optimizer=optimizer)

        history1 = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

        if test_acc > bestAcc:
            bestAcc = test_acc
            bestModel = model


    print('\nBest accuracy:', bestAcc)
    print("\nBest Model : ", bestModel.summary)

    # On save le best Model à un autre emplacement
    bestModel.save('saved_model/model3')
  

  
    print(bestModel.summary())
    
if __name__=="__main__":
    regression(pred = False)
