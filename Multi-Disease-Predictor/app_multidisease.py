#IMPORTING NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

#LOADING THE DATASET & PRE - PROCESSING
df = pd.read_csv('multi_disease.csv')

    #mapping target values
a = np.arange(0,len(df['prognosis'].unique())+1,1).tolist()
b = df['prognosis'].unique().tolist()
c = {j:i for i,j in zip(a,b)}
disease = {i:j for i,j in zip(a,b)}
output = df['prognosis'].unique().tolist()

df['prognosis'] = df['prognosis'].map(c)

    #since there are 132 features, we'll use feature selection to eliminate unnecessary ones
df = df.drop(['fluid_overload'],axis=1)#fluid_overload has single value in all of it's columns, so we can skip that
    #splitting data
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1050,test_size=0.14)

    #defining a function to eliminate more features based on pearson coefficient
def correlation(dataset, threshold):
  col_corr = set()
  corr_matrix = dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i,j])>threshold:
        colname = corr_matrix.columns[i]
        col_corr.add(colname)
  return col_corr
corr_features = correlation(x_train,0.38)#0.38 is the threshold value
#all the feature variables which did not qualify threshold is dropped
x_train = x_train.drop(corr_features,axis=1)
x_test = x_test.drop(corr_features,axis=1)

#TRAINING THE MODELS
    #DECISION TREE
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)

    #NAIVE BAYES
gnb = GaussianNB()
gnb.fit(x_train,y_train)
gnb_pred = gnb.predict(x_test)

    #RANDOM FOREST
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)

    #PLOTTING CORRELATION HEATMAP TO DISPLAY
from sklearn.metrics import confusion_matrix

#CREATING A STREAMLIT WEBPAGE & INTEGRATING THE ML MODELS
st.title('MULTI-DISEASE PREDICTOR')
with st.sidebar:
    side = option_menu('ML ALGORITHMS',options=['DECISION TREE','NAIVE BAYES','RANDOM FOREST'])
    #creating columns to place input field
t1, t2, t3 = st.columns(3)
dict2 = {'YES':'1','NO':'0'}
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        itching = st.selectbox('ITCHING?', options=['NO', 'YES'])
        itching = dict2[itching]

        skin_rash = st.selectbox('SKIN RASH?', options=['NO', 'YES'])
        skin_rash = dict2[skin_rash]

        nodal_skin_eruptions = st.selectbox('NODAL SKIN ERUPTIONS?', options=['NO', 'YES'])
        nodal_skin_eruptions = dict2[nodal_skin_eruptions]

        continuous_sneezing = st.selectbox('FREQUENT SNEEZING?', options=['NO', 'YES'])
        continuous_sneezing = dict2[continuous_sneezing]

        joint_pain = st.selectbox('JOINT PAIN?', options=['NO', 'YES'])
        joint_pain = dict2[joint_pain]

        stomach_pain = st.selectbox('STOMACH PAIN?', options=['NO', 'YES'])
        stomach_pain = dict2[stomach_pain]

        muscle_wasting = st.selectbox('MUSCLE LOSS?', options=['NO', 'YES'])
        muscle_wasting = dict2[muscle_wasting]

        vomiting = st.selectbox('VOMITING?', options=['NO', 'YES'])
        vomiting = dict2[vomiting]
    with c2:
        fatigue = st.selectbox('FATIGUE?', options=['NO', 'YES'])
        fatigue = dict2[fatigue]

        weight_gain = st.selectbox('WEIGHT GAIN?', options=['NO', 'YES'])
        weight_gain = dict2[weight_gain]

        anxiety = st.selectbox('ANXIETY?', options=['NO', 'YES'])
        anxiety = dict2[anxiety]

        weight_loss = st.selectbox('WEIGHT LOSS?', options=['NO', 'YES'])
        weight_loss = dict2[weight_loss]

        sunken_eyes = st.selectbox('SUNKEN EYES?', options=['NO', 'YES'])
        sunken_eyes = dict2[sunken_eyes]

        headache = st.selectbox('HEADACHE?', options=['NO', 'YES'])
        headache = dict2[headache]

        yellowish_skin = st.selectbox('YELLOWISH SKIN?', options=['NO', 'YES'])
        yellowish_skin = dict2[yellowish_skin]

        pain_behind_the_eyes = st.selectbox('PAIN BEHIND EYES?', options=['NO', 'YES'])
        pain_behind_the_eyes = dict2[pain_behind_the_eyes]

        skin_peeling = st.selectbox('SKIN PEELING?', options=['NO', 'YES'])
        skin_peeling = dict2[skin_peeling]

    with c3:
        blister = st.selectbox('BLISTER?', options=['NO', 'YES'])
        blister = dict2[blister]

        constipation = st.selectbox('CONSTIPATION?', options=['NO', 'YES'])
        constipation = dict2[constipation]

        swelling_of_stomach = st.selectbox('SWELLING OF STOMACH?', options=['NO', 'YES'])
        swelling_of_stomach = dict2[swelling_of_stomach]

        cramps = st.selectbox('CRAMPS?', options=['NO', 'YES'])
        cramps = dict2[cramps]

        spinning_movements = st.selectbox('HEAD SPINNING?', options=['NO', 'YES'])
        spinning_movements = dict2[spinning_movements]

        weakness_of_one_body_side = st.selectbox('WEAKNESS ON ONE SIDE OF BODY?', options=['NO', 'YES'])
        weakness_of_one_body_side = dict2[weakness_of_one_body_side]

        family_history = st.selectbox('FAMILY HISTORY AVAILABLE?', options=['NO', 'YES'])
        family_history = dict2[family_history]

        pus_filled_pimples = st.selectbox('PUS FILLED PIMPLES?', options=['NO', 'YES'])
        pus_filled_pimples = dict2[pus_filled_pimples]

btn = st.button('SUBMIT')

answers = {'itching': itching, 'skin_rash': skin_rash, 'nodal_skin_eruptions': nodal_skin_eruptions,
           'continuous_sneezing': continuous_sneezing, 'joint_pain': joint_pain, 'stomach_pain': stomach_pain,
           'muscle_wasting': muscle_wasting, 'vomiting': vomiting, 'fatigue': fatigue, 'weight_gain': weight_gain,
           'anxiety': anxiety, 'weight_loss': weight_loss, 'sunken_eyes': sunken_eyes, 'headache': headache,
           'yellowish_skin': yellowish_skin, 'pain_behind_the_eyes': pain_behind_the_eyes,
           'constipation': constipation, 'swelling_of_stomach': swelling_of_stomach, 'cramps': cramps,
           'spinning_movements': spinning_movements, 'weakness_of_one_body_side': weakness_of_one_body_side,
           'family_history': family_history, 'pus_filled_pimples': pus_filled_pimples,
           'skin_peeling': skin_peeling, 'blister': blister}
#build the input row with exactly the columns (and order) the models were trained on;
#symptoms not asked on the page are set to 0 (not present)
new = pd.DataFrame([[int(answers.get(col, 0)) for col in x_train.columns]], columns=x_train.columns)

if btn:
    if side == 'DECISION TREE':
        #MODEL PREDICTION
        st.header('POSSIBLE DIAGNOSIS OF :')
        dt_res = dt.predict(new)
        st.success(disease[dt_res[0]])
        #DISPLAYING MODEL PARAMETERS
        st.header('MODEL PARAMETERS')
        st.error("ACCURACY : {0}%".format(round(accuracy_score(y_test, dt_pred), 4)*100))
        
        #DISPLAYING CLASSIFICATION REPORT
        st.header('CLASSIFICATION REPORT')
        report1 = classification_report(y_test, dt_pred,target_names=output, output_dict=True)
        report1_df = pd.DataFrame(report1).transpose()
        st.dataframe(report1_df, height=212, width=1000)

    elif side == 'RANDOM FOREST':
        # MODEL PREDICTION
        st.header('POSSIBLE DIAGNOSIS OF :')
        rf_res = rf.predict(new)
        st.success(disease[rf_res[0]])
        #DISPLAYING MODEL PARAMETERS
        st.header('MODEL PARAMETERS')
        st.error("ACCURACY : {0}%".format(round(accuracy_score(y_test, rf_pred), 4) * 100))

        #DISPLAYING CLASSIFICATION REPORT
        st.header('CLASSIFICATION REPORT')
        report2 = classification_report(y_test, rf_pred, target_names=output, output_dict=True)
        report2_df = pd.DataFrame(report2).transpose()
        st.dataframe(report2_df, height=212, width=1000)

    else:
        # MODEL PREDICTION
        st.header('POSSIBLE DIAGNOSIS OF :')
        gnb_res = gnb.predict(new)
        st.success(disease[gnb_res[0]])
        #DISPLAYING MDOEL PARAMETERS
        st.header('MODEL PARAMETERS')
        st.error("ACCURACY : {0}%".format(round(accuracy_score(y_test, gnb_pred), 4) * 100))

        #DISPLAYING CLASSIFICATION REPORT
        st.header('CLASSIFICATION REPORT')
        report3 = classification_report(y_test, gnb_pred, target_names=output, output_dict=True)
        report3_df = pd.DataFrame(report3).transpose()
        st.dataframe(report3_df, height=212, width=1000)
