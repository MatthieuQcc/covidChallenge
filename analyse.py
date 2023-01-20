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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier



def regression(pred = False):


    # On récupère le dataFrame
    df = pd.read_csv("train.csv")

    # Colonne cible
    target_col = 'LABEL'

    # on récupère la colonne cible
    y = df[target_col]

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
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    print("## Création des données de train et de test -- FIN ET NORMALISE ##\n")


    # Test avec Random Forest

    '''
    minScore = 0 
    bestModel = RandomForestClassifier()


    for i in tqdm(range(2, 50)):
        for j in range(1, 10):
            clf = RandomForestClassifier(max_depth=i, min_samples_leaf=j, random_state=0).fit(X_train_transformed, y_train)
            currentScore = accuracy_score(y_test, clf.predict(X_test_transformed))
            # print("Acc score pour i = ", i, "  --->  ", currentScore)
            if currentScore > minScore:
                minScore = currentScore        
                bestModel = clf 

    print("\nRésultat trouvé : max depth = ", bestModel.max_depth)
    print("\nRésultat trouvé : max min_samples_leaf = ", bestModel.min_samples_leaf)
    print("\nAvec un score acc = ", minScore)
    '''


    # Meilleur score pour l'instant : RandomForestClassifier(max_depth=18, min_samples_leaf=5, random_state=0)


    model2 = RandomForestClassifier(max_depth=18, min_samples_leaf=5, random_state=0).fit(X_train_transformed, y_train)
    print("\nscore model 2 : ", accuracy_score(y_test, model2.predict(X_test_transformed)))


    # On génère le csv pour Kaggle
    if(pred == True):
        print("\n ## Génération du CSV ! ##")
        # On traite les données de test_set.csv
        test_data = pd.read_csv("prev.csv")
        test_data = test_data.drop(columns=["ID"])
        print(test_data.head(20))

        # On normalise les données en se basant sur le training set
        X_test_data_transformed = scaler.transform(test_data)

        # On génère le csv
        header = ["ID", "LABEL"]
        data = []
        for i in tqdm(range(len(X_test_data_transformed))):
            prediction = labelencoder.inverse_transform(model2.predict([X_test_data_transformed[i]]))
            data.append([i, prediction[0]])

        with open('predictionsKaggle.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # write data
            writer.writerows(data)


if __name__=="__main__":
    regression(pred = True)