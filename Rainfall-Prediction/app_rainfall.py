#conda activate main
#streamlit run 2_rainfall.py


#st.error == red color
#st.success == green color
#st.warning == yellow
#st.info == blue

#IMPORTING NECESSARY DEPENDENCIES
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#READING AND PROCESSING DATA
# df = pd.read_csv("weatherAUS.csv")
# pd.set_option("display.max_columns", None)
#
# print(df.columns[df.isna().any()])
#
# df = df.dropna()
#
# print(df.columns[df.isna().any()])
# print(df.shape)
#
# windgustdir = {'NNW':0, 'NW':1, 'WNW':2, 'N':3, 'W':4, 'WSW':5, 'NNE':6, 'S':7, 'SSW':8, 'SW':9, 'SSE':10,
#        'NE':11, 'SE':12, 'ESE':13, 'ENE':14, 'E':15}
# winddir9am = {'NNW':0, 'N':1, 'NW':2, 'NNE':3, 'WNW':4, 'W':5, 'WSW':6, 'SW':7, 'SSW':8, 'NE':9, 'S':10,
#        'SSE':11, 'ENE':12, 'SE':13, 'ESE':14, 'E':15}
# winddir3pm = {'NW':0, 'NNW':1, 'N':2, 'WNW':3, 'W':4, 'NNE':5, 'WSW':6, 'SSW':7, 'S':8, 'SW':9, 'SE':10,
#        'NE':11, 'SSE':12, 'ENE':13, 'E':14, 'ESE':15}
# df["WindGustDir"] = df["WindGustDir"].map(windgustdir)
# df["WindDir9am"] = df["WindDir9am"].map(winddir9am)
# df["WindDir3pm"] = df["WindDir3pm"].map(winddir3pm)
#
# df["RainToday"] = pd.get_dummies(df["RainToday"], drop_first = True)
# df["RainTomorrow"] = pd.get_dummies(df["RainTomorrow"], drop_first = True)
#
# # df1 = df.groupby(["Location"])["RainTomorrow"].value_counts().sort_values().unstack()
#
# location = {'Portland':1, 'Cairns':2, 'Walpole':3, 'Dartmoor':4, 'MountGambier':5,
#        'NorfolkIsland':6, 'Albany':7, 'Witchcliffe':8, 'CoffsHarbour':9, 'Sydney':10,
#        'Darwin':11, 'MountGinini':12, 'NorahHead':13, 'Ballarat':14, 'GoldCoast':15,
#        'SydneyAirport':16, 'Hobart':17, 'Watsonia':18, 'Newcastle':19, 'Wollongong':20,
#        'Brisbane':21, 'Williamtown':22, 'Launceston':23, 'Adelaide':24, 'MelbourneAirport':25,
#        'Perth':26, 'Sale':27, 'Melbourne':28, 'Canberra':29, 'Albury':30, 'Penrith':31,
#        'Nuriootpa':32, 'BadgerysCreek':33, 'Tuggeranong':34, 'PerthAirport':35, 'Bendigo':36,
#        'Richmond':37, 'WaggaWagga':38, 'Townsville':39, 'PearceRAAF':40, 'SalmonGums':41,
#        'Moree':42, 'Cobar':43, 'Mildura':44, 'Katherine':45, 'AliceSprings':46, 'Nhil':47,
#        'Woomera':48, 'Uluru':49}
# df["Location"] = df["Location"].map(location)
#
# df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%dT", errors = "coerce")
#
# df["Date_month"] = df["Date"].dt.month
# df["Date_day"] = df["Date"].dt.day

# df.to_csv("preprocessed_1.csv", index=False)

df_loaded = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preprocessed_1.csv'))
x = df_loaded.drop(["RainTomorrow", "Date"], axis=1)
y = df_loaded["RainTomorrow"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

print("The number of classes before SMOTE {}".format(Counter(y_train)))
sm=SMOTE(random_state=0)
x_train, y_train = sm.fit_resample(x_train, y_train)
print("The number of classes after SMOTE {}".format(Counter(y_train)))

sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

SVM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'svm_model_saved.sav')


def load_or_train_svm():
       # The saved model file is not in the repository. SVM is slow on the full training set,
       # so the first run trains on a random sample of 20,000 rows and saves the model for next time.
       if os.path.exists(SVM_PATH):
              with open(SVM_PATH, 'rb') as f:
                     return pickle.load(f)
       rng = np.random.RandomState(0)
       idx = rng.choice(len(x_train_sc), size=min(20000, len(x_train_sc)), replace=False)
       model = SVC().fit(x_train_sc[idx], np.asarray(y_train)[idx])
       with open(SVM_PATH, 'wb') as f:
              pickle.dump(model, f)
       return model

       #CREATING A STREAMLIT PAGE & INTEGRATING THE ML MODELS
import streamlit as st #package used to create webpage
from streamlit_option_menu import option_menu #package used for option box

st.title('RAINFALL PREDICTOR')
with st.sidebar:
       option = option_menu('CHOOSE YOUR MODEL',options=['LOGISTIC REGRESSION','KNN CLASSIFIER','SVM CLASSIFIER','DECISION TREE'])

c1, c2, c3 = st.columns(3)
with c1:
       location2 = {'Portland': 1, 'Cairns': 2, 'Walpole': 3, 'Dartmoor': 4, 'MountGambier': 5,
                   'NorfolkIsland': 6, 'Albany': 7, 'Witchcliffe': 8, 'CoffsHarbour': 9, 'Sydney': 10,
                   'Darwin': 11, 'MountGinini': 12, 'NorahHead': 13, 'Ballarat': 14, 'GoldCoast': 15,
                   'SydneyAirport': 16, 'Hobart': 17, 'Watsonia': 18, 'Newcastle': 19, 'Wollongong': 20,
                   'Brisbane': 21, 'Williamtown': 22, 'Launceston': 23, 'Adelaide': 24, 'MelbourneAirport': 25,
                   'Perth': 26, 'Sale': 27, 'Melbourne': 28, 'Canberra': 29, 'Albury': 30, 'Penrith': 31,
                   'Nuriootpa': 32, 'BadgerysCreek': 33, 'Tuggeranong': 34, 'PerthAirport': 35, 'Bendigo': 36,
                   'Richmond': 37, 'WaggaWagga': 38, 'Townsville': 39, 'PearceRAAF': 40, 'SalmonGums': 41,
                   'Moree': 42, 'Cobar': 43, 'Mildura': 44, 'Katherine': 45, 'AliceSprings': 46, 'Nhil': 47,
                   'Woomera': 48, 'Uluru': 49}
       Location = st.selectbox('ENTER LOCATION',options=sorted(location2))
       Location = location2[Location]#cnverts text inout from user into number input

       MinTemp = st.number_input('ENTER MIN TEMP')
       MaxTemp = st.number_input('ENTER MAX TEMP')
       Rainfall = st.number_input('ENTER RAINFALL')
       Evaporation = st.number_input('ENTER EVAPORATION')
       Sunshine = st.number_input('ENTER SUNSHINE')
       WindGustDir = st.number_input('ENTER WIND GUST DIRECTION')
       Date_day = st.number_input('ENTER DAY',min_value=1,max_value=31)

with c2:
       WindGustSpeed = st.number_input('ENTER WIND GUST SPEED')
       WindDir9am = st.number_input('ENTER WIND DIR @ 9 AM')
       WindDir3pm = st.number_input('ENTER WIND DIR @ 3 PM')
       WindSpeed9am = st.number_input('ENTER WIND SPEED @ 9 AM')
       WindSpeed3pm = st.number_input('ENTER WIND SPEED @ 3 PM')
       Humidity9am = st.number_input('ENTER HUMIDITY @ 9 AM')
       Humidity3pm = st.number_input('ENTER HUMIDITY @ 3 PM')

with c3:
       Pressure9am = st.number_input('ENTER PRESSURE @ 9 AM')
       Pressure3pm = st.number_input('ENTER PRESSURE @ 3 PM')
       Cloud9am = st.number_input('ENTER CLOUDS @ 9 AM')
       Cloud3pm = st.number_input('ENTER CLOUDS @ 3 PM')
       Temp9am = st.number_input('ENTER TEMPERATURE @ 9 AM')
       Temp3pm = st.number_input('ENTER TEMPERATURE @ 3 PM')
       RainToday = st.number_input('ENTER RAIN TODAY')
       Date_month = st.number_input('ENTER MONTH',min_value=1,max_value=12)

new = [[Location,  MinTemp,  MaxTemp,  Rainfall,  Evaporation,  Sunshine,
        WindGustDir,  WindGustSpeed,  WindDir9am,  WindDir3pm,  WindSpeed9am,
        WindSpeed3pm,  Humidity9am,  Humidity3pm,  Pressure9am,  Pressure3pm,
        Cloud9am,  Cloud3pm,  Temp9am,  Temp3pm,  RainToday,  Date_month,  Date_day]]
col1, col2, col3 = st.columns([18,1,1])#
with col1:
       submit = st.button('SUBMIT')

if submit:#if the submit button is clicked
       if option == 'KNN CLASSIFIER':
              # KNN
              knn = KNeighborsClassifier(n_neighbors=3)#calling the algorithm
              knn.fit(x_train_sc, y_train)#training the algorithm
              knn_pred = knn.predict(x_test_sc)#using the trained model to predict for testing dataset

              knn_res = knn.predict(sc.transform(new))#using the trained model to predict for new user input

              st.header('RESULTS OF KNN MODEL')
              if knn_res == 0:
                     st.info('THERE IS NO POSSIBILITY OF RAIN TOMORROW')
              else:
                     st.info('THERE IS A POSSIBILITY OF RAIN TOMORROW')

              st.header('MODEL PARAMETERS')
              st.error("ACCURACY OF THE MODEL IS : {0}%".format(round(accuracy_score(y_test, knn_pred), 2) * 100))
              st.warning("PRECISION OF THE MODEL IS : {0}%".format(round(precision_score(y_test, knn_pred), 2) * 100))
              st.info("F1 SCORE OF THE MODEL IS : : {0}%".format(round(f1_score(y_test, knn_pred), 2) * 100))
              st.success("RECALL OF THE MODEL IS : : {0}%".format(round(recall_score(y_test, knn_pred), 2) * 100))

              st.header('REPORT')
              report1 = classification_report(y_test, knn_pred, target_names=['no rain', 'rain'],output_dict=True)#creating a classification report
              report1_df_loaded = pd.DataFrame(report1).transpose()#converting to dataframe
              st.dataframe(report1_df_loaded, height=212, width=1000)#displaying dataframe on webpage


       elif option == 'SVM CLASSIFIER':
              # SVC
              svc = SVC()
              # svc.fit(x_train_sc, y_train)
              #
              # svc_pred = svc.predict(x_test_sc)
              # save the model to disk, since svm is taking long computation time, we'll save the model and call it later.
              # filename = 'svm_model_saved.sav'
              # pickle.dump(svc, open(filename, 'wb'))

              # some time later...

              # load the model from disk; if it isn't there yet, train it once and save it
              svc_loaded = load_or_train_svm()
              svc_loaded_pred = svc_loaded.predict(x_test_sc)

              svc_res = svc_loaded.predict(sc.transform(new))
              st.header('RESULTS OF SVM MODEL')
              if svc_res == 0:
                     st.info('THERE IS NO POSSIBILITY OF RAIN TOMORROW')
              else:
                     st.info('THERE IS A POSSIBILITY OF RAIN TOMORROW')

              st.header('MODEL PARAMETERS')
              st.error("ACCURACY OF THE MODEL IS :: {0}%".format(
                     round(accuracy_score(y_test, svc_loaded_pred), 2) * 100))
              st.warning("PRECISION OF THE MODEL IS : : {0}%".format(
                     round(precision_score(y_test, svc_loaded_pred), 2) * 100))
              st.info("F1 SCORE OF THE MODEL IS : : {0}%".format(
                     round(f1_score(y_test, svc_loaded_pred), 2) * 100))
              st.success("RECALL OF THE MODEL IS : : {0}%".format(
                     round(recall_score(y_test, svc_loaded_pred), 2) * 100))

              st.header('REPORT')
              report2 = classification_report(y_test, svc_loaded_pred, target_names=['no rain', 'rain'],
                                              output_dict=True)
              report2_df_loaded = pd.DataFrame(report2).transpose()
              st.dataframe(report2_df_loaded, height=212, width=1000)

              

       elif option == 'LOGISTIC REGRESSION':
              # LOGISTIC REGRESSION
              lr = LogisticRegression()
              lr.fit(x_train_sc, y_train)
              lr_pred = lr.predict(x_test_sc)

              lr_res = lr.predict(sc.transform(new))
              st.header('RESULTS OF LOG REGRESSION')
              if lr_res == 0:
                     st.info('THERE IS NO POSSIBILITY OF RAIN TOMORROW')
              else:
                     st.info('THERE IS A POSSIBILITY OF RAIN TOMORROW')

              st.header('MODEL PARAMETERS')
              st.error(
                     "ACCURACY OF THE MODEL IS : : {0}%".format(round(accuracy_score(y_test, lr_pred), 2) * 100))
              st.warning("PRECISION OF THE MODEL IS : : {0}%".format(
                     round(precision_score(y_test, lr_pred), 2) * 100))
              st.info("F1 SCORE OF THE MODEL IS : : {0}%".format(round(f1_score(y_test, lr_pred), 2) * 100))
              st.success(
                     "RECALL OF THE MODEL IS : : {0}%".format(round(recall_score(y_test, lr_pred), 2) * 100))

              st.header('REPORT')
              report3 = classification_report(y_test, lr_pred, target_names=['no rain', 'rain'], output_dict=True)
              report3_df_loaded = pd.DataFrame(report3).transpose()
              st.dataframe(report3_df_loaded, height=212, width=1000)

              

       if option == 'DECISION TREE':
              dt = DecisionTreeClassifier()
              dt.fit(x_train_sc,y_train)
              dt_pred = dt.predict(x_test_sc)
              dt_res = dt.predict(sc.transform(new))

              st.header('RESULTS OF DECISION TREE')
              if dt_res == 0:
                     st.info('THERE IS NO POSSIBILITY OF RAIN TOMORROW')
              else:
                     st.info('THERE IS A POSSIBILITY OF RAIN TOMORROW')

              st.header('MODEL PARAMETERS')
              st.error(
                     "ACCURACY OF THE MODEL IS : : {0}%".format(round(accuracy_score(y_test, dt_pred), 2) * 100))
              st.warning("PRECISION OF THE MODEL IS : : {0}%".format(
                     round(precision_score(y_test, dt_pred), 2) * 100))
              st.info("F1 SCORE OF THE MODEL IS : : {0}%".format(round(f1_score(y_test, dt_pred), 2) * 100))
              st.success(
                     "RECALL OF THE MODEL IS : : {0}%".format(round(recall_score(y_test, dt_pred), 2) * 100))

              st.header('REPORT')
              report4 = classification_report(y_test, dt_pred, target_names=['no rain', 'rain'], output_dict=True)
              report4_df_loaded = pd.DataFrame(report4).transpose()
              st.dataframe(report4_df_loaded, height=212, width=1000)

              

else:#if the button is NOT clciked
       st.error('CLICK SUBMIT BUTTON')





























