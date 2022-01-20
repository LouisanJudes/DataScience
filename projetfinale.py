# -*- coding: utf-8 -*-
"""
PROJET Apprentissage dynamique

JUDES RAMESH Louisan
ABDOUS Erwann
TALEB BENDIAB Zahia Nada

"""
"""
Created on Tue Jan 11 22:46:56 2022

@author: louis
"""

import pandas as pd
from catboost import CatBoostRegressor
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor 
from sklearn.linear_model import LinearRegression

#chemin d'accès au document
zf = zipfile.ZipFile("C:/Users/louis/Desktop/projet aada/martdata.zip")
test = pd.read_csv(zf.open("Test.csv"))
train= pd.read_csv(zf.open("Train.csv"))

""" feature engineering"""

#datatype
train.dtypes
#on remarque qu'il y a énormément de valeurs littérale (car ce sont beaucoup de catégories)


#on crée une fonction qui permet de remplacer toutes les valeurs littérale en valeurs numérique pour simplifier les régression
def convertir_en_numérique(dataframe):
    for i in range(len(dataframe['Outlet_Size'])):
        if dataframe['Outlet_Size'][i]=='Small':
            dataframe['Outlet_Size'][i]=1 
        if dataframe['Outlet_Size'][i]=='Medium':
            dataframe['Outlet_Size'][i]=2
        if dataframe['Outlet_Size'][i]=='High':
           dataframe['Outlet_Size'][i]=3
    for i in range(len(dataframe['Item_Fat_Content'])):
        if dataframe['Item_Fat_Content'][i]=='Low Fat' or dataframe['Item_Fat_Content'][i]=='low fat' or dataframe['Item_Fat_Content'][i]=='LF':
            dataframe['Item_Fat_Content'][i]=1 
        if dataframe['Item_Fat_Content'][i]=='Regular' or dataframe['Item_Fat_Content'][i]=='reg':
            dataframe['Item_Fat_Content'][i]=2
    for i in range(len(dataframe['Item_Type'])):
        if dataframe['Item_Type'][i]=='Dairy':
            dataframe['Item_Type'][i]=1 
        if dataframe['Item_Type'][i]=='Soft Drinks':
            dataframe['Item_Type'][i]=2
        if dataframe['Item_Type'][i]=='Meat':
            dataframe['Item_Type'][i]=3
        if dataframe['Item_Type'][i]=='Fruits and Vegetables':
            dataframe['Item_Type'][i]=4
        if dataframe['Item_Type'][i]=='Household':
            dataframe['Item_Type'][i]=5
        if dataframe['Item_Type'][i]=='Baking Goods':
            dataframe['Item_Type'][i]=6
        if dataframe['Item_Type'][i]=='Snack Foods':
            dataframe['Item_Type'][i]=7
        if dataframe['Item_Type'][i]=='Frozen Foods':
            dataframe['Item_Type'][i]=8
        if dataframe['Item_Type'][i]=='Breakfast':
            dataframe['Item_Type'][i]=9
        if dataframe['Item_Type'][i]=='Health and Hygiene':
            dataframe['Item_Type'][i]=10
        if dataframe['Item_Type'][i]=='Hard Drinks':
            dataframe['Item_Type'][i]=11
        if dataframe['Item_Type'][i]=='Canned':
            dataframe['Item_Type'][i]=12
        if dataframe['Item_Type'][i]=='Breads':
            dataframe['Item_Type'][i]=13
        if dataframe['Item_Type'][i]=='Starchy Foods':
            dataframe['Item_Type'][i]=14
        if dataframe['Item_Type'][i]=='Others':
            dataframe['Item_Type'][i]=15
        if dataframe['Item_Type'][i]=='Seafood':
            dataframe['Item_Type'][i]=16
    for i in range(len(dataframe['Outlet_Location_Type'])):
        if dataframe['Outlet_Location_Type'][i]=='Tier 1':
            dataframe['Outlet_Location_Type'][i]=1 
        if dataframe['Outlet_Location_Type'][i]=='Tier 2':
            dataframe['Outlet_Location_Type'][i]=2
        if dataframe['Outlet_Location_Type'][i]=='Tier 3':
            dataframe['Outlet_Location_Type'][i]=3
    for i in range(len(dataframe['Outlet_Type'])):
        if dataframe['Outlet_Type'][i]=='Supermarket Type1':
            dataframe['Outlet_Type'][i]=1
        if dataframe['Outlet_Type'][i]=='Supermarket Type2':
            dataframe['Outlet_Type'][i]=2
        if dataframe['Outlet_Type'][i]=='Supermarket Type3':
            dataframe['Outlet_Type'][i]=3
        if dataframe['Outlet_Type'][i]=='Grocery Store':
            dataframe['Outlet_Type'][i]=4
    return(dataframe)



#on crée une nouvelle dataframe train en enlevant les deux colonnes identifiants qui ne sont pas réellement informatives
df_train = train.drop(columns=['Item_Identifier','Outlet_Identifier'])

#on utilise la fonction convertir_en_numérique pour n'avoir que des valeurs numériques dans la dataframe
data_train= convertir_en_numérique(df_train)

#on crée une nouvelle dataframe test en enlevant les deux colonnes identifiants qui ne sont pas réellement informatives
df_test =test.drop(columns=['Item_Identifier','Outlet_Identifier'])

#on utilise la fonction convertir_en_numérique pour n'avoir que des valeurs numériques dans la dataframe
data_test= convertir_en_numérique(df_test)

#permet de voir les valeurs manquantes
train.isnull().sum()

#on remarque qu'il n'y a que les deux colonnes "Item_Weight" et "Outlet_Size" qui ont des valeurs manquantes.
#donc on calcule les moyennes des  colonnes pour pouvoir les remplacer dans leurs dataframes respectives
moyenne_weight_train=data_train.Item_Weight.describe()[1]
moyenne_Size_train=data_train.Outlet_Size.describe()[1]

moyenne_weight_test=data_test.Item_Weight.describe()[1]
moyenne_Size_test=data_test.Outlet_Size.describe()[1]

#on remplace les valeurs manquantes par les moyennes des colonnes 
data_train['Item_Weight'].fillna(moyenne_weight_train, inplace=True)
data_train['Outlet_Size'].fillna(round(moyenne_Size_train), inplace=True)

data_test['Item_Weight'].fillna(moyenne_weight_test, inplace=True)
data_test['Outlet_Size'].fillna(round(moyenne_Size_test), inplace=True)

#On crée un set d'entraînement pour modéliser et un set de validation pour vérifier les performances 
X = data_train.drop(['Item_Outlet_Sales'], axis=1)
y = data_train.Item_Outlet_Sales

#on utilise la fonction train_test_split afin de crée c'est deux sets d'entraînement et de validation à partir de la même dataframe
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.75)

"""model building and model evaluation"""

#on applique les différents modèles de régression

#catboost
model=CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, eval_metric='RMSE')
model.fit(X_train, y_train,eval_set=(X_validation, y_validation),plot=True)
predicted_y_catboost= model.predict(X_validation)

#DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0) 
regressor.fit(X_train, y_train)
y_pred =regressor.predict(X_validation)

#regression lineaire
lmodellineaire = LinearRegression()
lmodellineaire.fit(X_train, y_train)
predicted_y_3 = lmodellineaire.predict(X_validation)


#calcul et affichage pour chaque modèle du rmse qui est l'erreur quadratique ainsi que de r2 qui est le coefficient de corrélation au carré 
expected_y = y_validation

print('model catboost \n')

rmse_catboost= mean_squared_error(expected_y, predicted_y_catboost, squared=False)
print('rmse:', rmse_catboost)

r2_catboost=r2_score(expected_y, predicted_y_catboost)
print('r2:',r2_catboost,'\n')

print('model DecisionTreeRegressor \n')
rmse_DecisionTreeRegressor= mean_squared_error(expected_y, y_pred, squared=False)
print('rmse:',rmse_DecisionTreeRegressor)

r2_DecisionTreeRegressor=r2_score(expected_y, y_pred)
print('r2:',r2_DecisionTreeRegressor,'\n')

print('model linearRegression \n')
rmse_3= mean_squared_error(expected_y, predicted_y_3, squared=False)
print('rmse:',rmse_3)

r2_3=r2_score(expected_y, predicted_y_3)
print('r2:',r2_3)


#on prédit Item_Outlet_Sales pour le fichier "Test.csv"
prediction_test= pd.DataFrame()
prediction_test['Item_Identifier'] = test['Item_Identifier']
prediction_test['Outlet_Identifier'] = test['Outlet_Identifier']
prediction_test['Item_Outlet_Sales'] = model.predict(data_test)

"""
#fonction bonus: on crée un fichier csv avec les prédictions pour le fichier "Test.csv"

prediction_test.to_csv("prediction_test.csv")
"""


# Conclusion


#Lors de notre travail, nous avons pu essayé trois modèles de régression: CatBoostRegressor, DecisionTreeRegressor et LinearRegression.
#Après exécution de notre code, et cela de multiples fois, nous avons remarqué que les trois modèles étaient plutôt stables dans leurs valeurs de retour 
#et avons décidé de garder un des jets pour vous donner un ordre de grandeur de ce que nous avons trouvé:

#model catboost 

# rmse: 1073.2640970620776
# r2: 0.5932120995126708 

# model DecisionTreeRegressor 

# rmse: 1571.3294365396514
# r2: 0.12805454836874075 

# model linearRegression 

# rmse: 1274.8387976487886
# r2: 0.4260614875266914

#On remarque de manière assez claire que le CatBoostRegressor est le meilleur des trois modèles puisqu'il possède le rmse le plus bas
#et que son r^2 est le plus proche de 1 des trois modèles essayés ici.
#La régression linéaire (LinearRegression ici) semble être la pire des trois au vu des valeurs que notre code nous a donné.
#On peut se demander pourquoi catboost regressor est le meilleur modèle. On peut supposer que la manière dont elle fonction est en cause.
#En effet, catboost permet de choisir les paramètres, et notamment le "learning rate" et le nombre d'itérations qui influence grandement sur le résultat final.
#De plus, on peut déduire qu'il vaudrait mieux utiliser des modèles non linéaires pour obtenir les meilleurs résultats possibles.

#On note que la prédiction pour le fichier Test.csv a été faite en toute fin
#Catboostregressor est le meilleur des trois modèles qu'on ait au vu de nos ressources, mais on peut supposer qu'au vu des critères d'évaluation de performances,
#la prediction restera assez approximative.