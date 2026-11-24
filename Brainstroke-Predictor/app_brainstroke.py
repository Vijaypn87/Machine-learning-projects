#IMPORT ALL THE NECESSARY LIBRARIES
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

DATA_PATH = Path(__file__).parent / 'brain_stroke.csv'

#ENCODINGS - the same dictionaries are used for the training data and for the user's input,
#so the model always sees the numbers it was trained on
GENDER = {'MALE': 1, 'FEMALE': 2}
YES_NO = {'YES': 1, 'NO': 0}  # hypertension & heart_disease are already 0/1 in the dataset
MARRIED = {'YES': 1, 'NO': 2}
WORK_TYPE = {'PRIVATE': 1, 'SELF-EMPLOYED': 2, 'GOVERNMENT': 3, 'CHILD': 4}
RESIDENCE = {'URBAN': 1, 'RURAL': 2}
SMOKING = {'FORMERLY SMOKED': 1, 'NEVER SMOKED': 2, 'SMOKES': 3, 'UNKNOWN': 4}


#READING THE DATASET, PRE-PROCESSING & TRAINING (cached so it runs once, not on every click)
@st.cache_resource
def train_models():
    data = pd.read_csv(DATA_PATH)
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 2})
    data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 2})
    data['work_type'] = data['work_type'].map({'Private': 1, 'Self-employed': 2, 'Govt_job': 3, 'children': 4})
    data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 2})
    data['smoking_status'] = data['smoking_status'].map({'formerly smoked': 1, 'never smoked': 2, 'smokes': 3, 'Unknown': 4})

    x = data.iloc[:, :-1].values  # input columns
    y = data.iloc[:, -1].values  # output column

    #SPLIT FIRST, then over-sample only the training data.
    #Over-sampling before the split leaks synthetic copies of test rows into training and inflates the scores.
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=2100, test_size=0.3, stratify=y)
    X_train, Y_train = SMOTE(random_state=0).fit_resample(X_train, Y_train)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    models = {
        'LOGISTIC REGRESSION': LogisticRegression(random_state=0, max_iter=1000),
        'RANDOM FOREST': RandomForestClassifier(random_state=0),
        'DECISION TREE': DecisionTreeClassifier(random_state=0),
        'SVM CLASSIFIER': SVC(random_state=0),
    }
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, Y_train)
        predictions[name] = model.predict(X_test)
    return sc, models, Y_test, predictions


sc, models, Y_test, predictions = train_models()


#CREATING A STREAMLIT WEBPAGE
st.title('BRAIN STROKE PREDICTOR')
st.sidebar.title('INPUT PARAMETERS')

gender = GENDER[st.sidebar.selectbox('ENTER GENDER', options=list(GENDER))]
age = st.sidebar.number_input('ENTER AGE', min_value=0, max_value=120, value=40, step=1)
hypertension = YES_NO[st.sidebar.selectbox('HYPERTENSION?', options=list(YES_NO))]
heart_disease = YES_NO[st.sidebar.selectbox('HEART DISEASE?', options=list(YES_NO))]
married = MARRIED[st.sidebar.selectbox('MARRIAGE HISTORY', options=list(MARRIED))]
work_type = WORK_TYPE[st.sidebar.selectbox('WORK TYPE', options=list(WORK_TYPE))]
residence_type = RESIDENCE[st.sidebar.selectbox('ENTER RESIDENCE TYPE', options=list(RESIDENCE))]
smoke = SMOKING[st.sidebar.selectbox('HISTORY OF SMOKING', options=list(SMOKING))]
avg_glucose_level = st.sidebar.number_input('ENTER AVERAGE GLUCOSE LEVEL', min_value=50.0, value=100.0, step=1.0)
bmi = st.sidebar.number_input('ENTER BMI', min_value=10.0, value=25.0, step=0.1)

#same column order as the dataset
new = [[gender, age, hypertension, heart_disease, married, work_type, residence_type, avg_glucose_level, bmi, smoke]]

submit = st.sidebar.button('SUBMIT')

CMAPS = {'LOGISTIC REGRESSION': 'Blues', 'RANDOM FOREST': 'Greens', 'DECISION TREE': 'Reds', 'SVM CLASSIFIER': 'Purples'}


def show_model(name):
    model = models[name]
    pred = predictions[name]

    # DISPLAYING RESULT
    if model.predict(sc.transform(new))[0] == 1:
        st.error('POSSIBILITY OF BRAIN STROKE DETECTED : PLEASE CONSULT A DOCTOR')
    else:
        st.success('NO POSSIBILITY OF BRAIN STROKE DETECTED')

    # DISPLAYING MODEL PARAMETERS (stroke = positive class)
    st.header('MODEL PARAMETERS')
    st.warning('ACCURACY : {0:.1f}%'.format(accuracy_score(Y_test, pred) * 100))
    st.success('PRECISION : {0:.1f}%'.format(precision_score(Y_test, pred, zero_division=0) * 100))
    st.error('F1 SCORE : {0:.1f}%'.format(f1_score(Y_test, pred) * 100))
    st.info('RECALL SCORE : {0:.1f}%'.format(recall_score(Y_test, pred) * 100))

    # DISPLAYING CLASSIFICATION REPORT
    st.header('CLASSIFICATION REPORT')
    report = classification_report(Y_test, pred, target_names=['no stroke', 'stroke'], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), height=212, width=750)

    # DISPLAYING CONFUSION MATRIX of THIS model
    st.header('CONFUSION MATRIX')
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(Y_test, pred), annot=True, fmt='d', cmap=CMAPS[name], ax=ax,
                xticklabels=['no stroke', 'stroke'], yticklabels=['no stroke', 'stroke'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)


#TRAINING ML MODELS & INTEGRATING IT WITH STREAMLIT PAGE
if submit:  # if the submit button is clicked
    tabs = st.tabs(list(models) + ['MODEL GRAPHS'])
    for tab, name in zip(tabs, models):
        with tab:
            show_model(name)

    with tabs[-1]:
        # MODEL GRAPHS
        data_graph = pd.read_csv(DATA_PATH)

        st.subheader("CATEGORICAL FEATURES COUNTPLOT")
        for col, title in [('gender', 'Gender'), ('hypertension', 'Hypertension'), ('heart_disease', 'Heart Disease'),
                           ('ever_married', 'Marriage History'), ('work_type', 'Work Type'),
                           ('Residence_type', 'Residence Type')]:
            fig, ax = plt.subplots()
            sns.countplot(x=col, hue='stroke', data=data_graph, ax=ax)
            ax.set_title(f'{title} count-plot')
            st.pyplot(fig)

        st.subheader("MODELS COMPARISON")
        st.caption('Computed on the held-out test set. Precision/recall are for the "stroke" class, '
                   'which is rare (about 5% of rows), so look at recall as well as accuracy.')
        names = list(models)
        scores = {
            'Accuracy': [accuracy_score(Y_test, predictions[n]) * 100 for n in names],
            'Precision': [precision_score(Y_test, predictions[n], zero_division=0) * 100 for n in names],
            'Recall': [recall_score(Y_test, predictions[n]) * 100 for n in names],
        }
        for (metric, values), color in zip(scores.items(), ['red', 'blue', 'green']):
            fig, ax = plt.subplots()
            ax.bar(names, values, color=color, width=0.4)
            ax.set_ylim(0, 100)
            ax.set_xlabel("Models")
            ax.set_ylabel(metric + " (%)")
            ax.set_title(f"{metric} of different models")
            ax.tick_params(axis='x', labelrotation=15)
            st.pyplot(fig)

else:
    st.error('CLICK SUBMIT BUTTON')
