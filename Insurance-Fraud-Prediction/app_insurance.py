#IMPORTING ALL THE NECESSARY LIBRARIES
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

DATA_PATH = Path(__file__).parent / 'insurance_claims_mod.csv'

CATEGORICAL = ['insured_sex', 'insured_occupation', 'insured_relationship', 'incident_severity',
               'property_damage', 'police_report_available', 'collision_type', 'policy_state',
               'insured_education_level', 'auto_make', 'auto_year']


#DATA PREPROCESSING & TRAINING (cached so it runs once, not on every click)
@st.cache_resource
def train_model():
    df_org = pd.read_csv(DATA_PATH)
    df_org = df_org.replace('?', 'UNKNOWN')  # missing values are stored as '?' in the dataset

    df = df_org.copy()
    #one encoder PER column, kept so the user's input is encoded exactly like the training data
    encoders = {}
    for col in CATEGORICAL:
        encoders[col] = LabelEncoder().fit(df[col])
        df[col] = encoders[col].transform(df[col])
    df['fraud_reported'] = df['fraud_reported'].map({'N': 0, 'Y': 1})

    x = df.drop(columns='fraud_reported')
    y = df['fraud_reported']
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=0, stratify=y)

    # Feature scaling: fit on the training data only, then apply the same scaling to the test data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # class_weight='balanced' because only about 25% of claims are fraud; without it the model
    # mostly predicts "no fraud"
    model = LogisticRegression(random_state=0, class_weight='balanced')
    model.fit(X_train, Y_train)
    return df_org, list(x.columns), encoders, sc, model, Y_test, model.predict(X_test)


df_org, feature_columns, encoders, sc, model, Y_test, y_pred = train_model()


#creating a streamlit page
st.title("INSURANCE FRAUD PREDICTOR")
st.sidebar.title("INPUT PARAMETERS")

LABELS = {'insured_sex': 'ENTER SEX', 'insured_occupation': 'ENTER OCCUPATION : ',
          'insured_relationship': 'ENTER INSURED RELATIONSHIP : ', 'incident_severity': 'ENTER INCIDENT SEVERITY : ',
          'property_damage': 'ENTER PROPERTY DAMAGE : ', 'police_report_available': 'IS POLICE REPORT AVAILABLE : ',
          'collision_type': 'ENTER COLLISION TYPE : ', 'policy_state': 'ENTER POLICY STATE : ',
          'insured_education_level': 'ENTER INSURED EDUCATION LEVEL : ', 'auto_make': 'ENTER AUTO MAKE : ',
          'auto_year': 'ENTER AUTO YEAR : '}

user_input = {'age': st.sidebar.number_input("ENTER AGE : ", step=1, min_value=18, max_value=100, value=40)}
for col in CATEGORICAL:
    choice = st.sidebar.selectbox(LABELS[col], options=list(encoders[col].classes_))
    user_input[col] = encoders[col].transform([choice])[0]

new = pd.DataFrame([user_input])[feature_columns]  # same column order as training
result = model.predict(sc.transform(new))[0]

submit = st.sidebar.button("SUBMIT")

if submit:
    if result == 0:
        st.success('NO FRAUD DETECTED')
    else:
        st.error('POSSIBLE FRAUD DETECTED')

    tab1, tab2, tab3 = st.tabs(['MODEL PARAMETERS', 'CLASSIFICATION REPORT', 'MODEL GRAPHS'])
    with tab1:
        st.header('MODEL PARAMETERS')
        st.info("ACCURACY : {0:.1f}%".format(accuracy_score(Y_test, y_pred) * 100))
        st.info("PRECISION : {0:.1f}%".format(precision_score(Y_test, y_pred, zero_division=0) * 100))
        st.info("RECALL : {0:.1f}%".format(recall_score(Y_test, y_pred) * 100))
        st.info("F1 SCORE : {0:.1f}%".format(f1_score(Y_test, y_pred) * 100))
    with tab2:
        st.header('CLASSIFICATION REPORT')
        report = classification_report(Y_test, y_pred, target_names=['no fraud', 'fraud'], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), height=212, width=1000)
    with tab3:
        st.header('MODEL GRAPHS')
        st.caption('Number of claims in each group, split by whether fraud was reported.')
        plt.style.use('fivethirtyeight')
        for col in ['insured_sex', 'insured_occupation', 'incident_severity', 'police_report_available', 'collision_type']:
            fig, ax = plt.subplots(figsize=(10, 6))
            pd.crosstab(df_org[col], df_org['fraud_reported']).plot.bar(ax=ax)
            ax.set_ylabel('Number of claims')
            ax.tick_params(axis='x', labelrotation=20)
            st.pyplot(fig)

else:
    st.error('COMPLETE ALL PARAMETERS')
