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
import tensorflow as tf


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


# On traite les données de test_set.csv
test_data = pd.read_csv("prev.csv")
test_data = test_data.drop(columns=["ID"])

# On normalise les données en se basant sur le training set
X_test_data_transformed = scaler.transform(test_data)



# On load le model

model1 = tf.keras.models.load_model('saved_model/model1')
model2 = tf.keras.models.load_model('saved_model/model2')
model3 = tf.keras.models.load_model('saved_model/model3')

print(model1.evaluate(X_test, y_test, verbose=2))
print(model2.evaluate(X_test, y_test, verbose=2))
print(model3.evaluate(X_test, y_test, verbose=2))






print("\nPROBA")
pred = model3.predict(X_test_data_transformed)
print(pred)

print("\nCLASSE,")
print(np.argmax(pred, axis=1))

pred = np.argmax(pred, axis=1)

pred = labelencoder.inverse_transform(pred)

data = []
for i in range(len(pred)): data.append([i, pred[i]])

# On génère le csv pour Kaggle
print("\n ## Génération du CSV ! ##")

# On génère le csv
header = ["ID", "LABEL"]

with open('predictionsKaggle.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    # write data
    writer.writerows(data)
