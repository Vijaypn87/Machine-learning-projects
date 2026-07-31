 #IMPORT ALL THE NECESSARY LIBRARIES
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#READING THE DATASET & PRE-PROCESSING
st.set_option('deprecation.showPyplotGlobalUse', False)

data = pd.read_csv('brain_stroke.csv')#loading the dataset
# data = data.drop(['avg_glucose_level','bmi'],axis=1)#dropping unnecessary columns
print(data.columns[data.isna().any()])#checking for NaN values
print(data.min())


data['gender'] = data['gender'].map({'Male':'1','Female':'2'})
data['ever_married'] = data['ever_married'].map({'Yes':'1','No':'2'})
data['work_type'] = data['work_type'].map({'Private':'1', 'Self-employed':'2', 'Govt_job':'3', 'children':'4'})
data['Residence_type'] = data['Residence_type'].map({'Urban':'1', 'Rural':'2'})
data['smoking_status'] = data['smoking_status'].map({'formerly smoked':'1', 'never smoked':'2', 'smokes':'3', 'Unknown':'4'})

print(data['stroke'].value_counts())
#since the output data is imbalanced we have to over-sample and balance it
#SPLITTING THE DATA
from imblearn.over_sampling import SMOTE
x = data.iloc[:,:-1].values#taking input columns
y = data.iloc[:,-1].values#taking output column
oversample = SMOTE()
X, Y = oversample.fit_resample(x, y)
data=data.sample(frac=1).reset_index(drop=True)#shuffling dataset
print(pd.DataFrame(Y).value_counts())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=2100,test_size=0.3)#splitting dataset into training and testing
sc = StandardScaler()#scaling the data
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#CREATING A STREAMLIT WEBPAGE
st.title('BRAIN STROKE PREDICTOR')
st.sidebar.title('INPUT PARAMETERS')

gender = st.sidebar.selectbox('ENTER GENDER',options={'MALE','FEMALE'})
gender1 = {'MALE':'1','FEMALE':'2'}#creating a text to number dictionary
gender = gender1[gender]#converting text to number using dictionary

age = st.sidebar.number_input('ENTER AGE',min_value=9,step=1)

hypertension = st.sidebar.selectbox('HYPERTENSION?',options={'YES','NO'})
hypertension1 = {'YES':'1','NO':'2'}
hypertension = hypertension1[hypertension]

heart_disease = st.sidebar.selectbox('HEART DISEASE?',options={'YES','NO'})
heart_disease1 = {'YES':'1','NO':'2'}
heart_disease = heart_disease1[heart_disease]

married = st.sidebar.selectbox('MARRIAGE HISTORY',options={'YES','NO'})
married1 = {'YES':'1','NO':'2'}
married = married1[married]

work_type = st.sidebar.selectbox('WORK TYPE',options={'PRIVATE':'1', 'SELF-EMPLOYED':'2', 'GOVERNMENT':'3', 'CHILD':'4'})
work_type1 = {'PRIVATE':'1', 'SELF-EMPLOYED':'2', 'GOVERNMENT':'3', 'CHILD':'4'}
work_type = work_type1[work_type]

residence_type = st.sidebar.selectbox('ENTER RESIDENCE TYPE',options={'URBAN':'1', 'RURAL':'2'})
residence_type1 = {'URBAN':'1', 'RURAL':'2'}
residence_type = residence_type1[residence_type]

smoke = st.sidebar.selectbox('HISTORY OF SMOKING',options={'FORMERLY SMOKED':'1', 'NEVER SMOKED':'2', 'SMOKES':'3', 'UNKNOWN':'4'})
smoke1 = {'FORMERLY SMOKED':'1', 'NEVER SMOKED':'2', 'SMOKES':'3', 'UNKNOWN':'4'}
smoke = smoke1[smoke]

avg_glucose_level = st.sidebar.number_input("ENTER AVERAGE GLUCOSE LEVEL",step=1, min_value=55)

bmi = st.sidebar.number_input("ENTER BMI", min_value=14)

new = [[gender,age,hypertension,heart_disease,married,work_type,residence_type,avg_glucose_level,bmi,smoke]]#storing all user inputs in an array

submit = st.sidebar.button('SUBMIT')#creating a submit button

labels = ["yes","no"]
#TRAINING ML MODELS & INTEGRATING IT WITH STREAMLIT PAGE
if submit:#if the submit button is clicked
    t1, t2, t3, t4, t5 = st.tabs(['LOGISTIC REGRESSION', 'RANDOM FOREST', 'DECISION TREE', 'SVM CLASSIFIER', 'MODEL GRAPHS'])
    with t1:
        # LOGISTIC REGRESSION
        # TRAINING THE MODEL
        lr = LogisticRegression(random_state=0)  # calling the function
        lr.fit(X_train, Y_train)  # training the model
        lr_pred = lr.predict(X_test)  # predicting values for testing data
        lr_res = lr.predict(sc.transform(new))  # predicting values for new user inputs

        # DISPLAYING RESULT
        if lr_res == 1:
            st.error('POSSIBILITY OF BRAIN STROKE DETECTED : PLEASE CONSULT A DOCTOR')
        else:
            st.success('NO POSSIBILITY OF BRAIN STROKE DETECTED')
        # DISPLAYING MODEL PARAMETERS
        st.header('MODEL PARAMETERS')
        st.warning('ACCURACY : {0}%'.format(round(accuracy_score(Y_test, lr_pred), 2) * 100))
        st.success('PRECISION : {0}%'.format(round(precision_score(Y_test, lr_pred), 2) * 100))
        st.error('F1 SCORE : {0}%'.format(round(f1_score(Y_test, lr_pred), 2) * 100))
        st.info('RECALL SCORE : {0}%'.format(round(recall_score(Y_test, lr_pred), 2) * 100))

        # DISPLAYING CLASSIFICATION REPORT
        st.header('CLASSIFICATION REPORT')
        report1 = classification_report(Y_test, lr_pred, output_dict=True)  # creating a classification report
        report1_df = pd.DataFrame(report1).transpose()  # converting report into a dataframe
        st.dataframe(report1_df, height=212, width=750)  # displaying dataframe on webpage

        #DISPLAYING CONFUSION MATRIX
        # Compute confusion matrix
        cm = confusion_matrix(Y_test, lr_pred)
        cm_df = pd.DataFrame(cm)

        # Plot heatmap
        st.header('CONFUSION MATRIX')
        sns.heatmap(cm_df, annot=True, cmap='Blues')
        st.pyplot()

    with t2:  # inside tab2
        # RANDOM FOREST
        # TRAINING THE MODEL
        rand = RandomForestClassifier()
        rand.fit(X_train, Y_train)
        rand_pred = rand.predict(X_test)
        rand_res = rand.predict(sc.transform(new))

        # DISPLAYING RESULT
        if rand_res == 1:
            st.error('POSSIBILITY OF BRAIN STROKE DETECTED : PLEASE CONSULT A DOCTOR')
        else:
            st.success('NO POSSIBILITY OF BRAIN STROKE DETECTED')
        # DISPLAYING MODEL PARAMETERS
        st.header('MODEL PARAMETERS')
        st.warning('ACCURACY : {0}%'.format(round(accuracy_score(Y_test, rand_pred), 2) * 100))
        st.success('PRECISION : {0}%'.format(round(precision_score(Y_test, rand_pred), 2) * 100))
        st.error('F1 SCORE : {0}%'.format(round(f1_score(Y_test, rand_pred), 2) * 100))
        st.info('RECALL SCORE : {0}%'.format(round(recall_score(Y_test, rand_pred), 2) * 100))

        # DISPLAYING CLASSIFICATION REPORT
        st.header('CLASSIFICATION REPORT')
        report2 = classification_report(Y_test, rand_pred, output_dict=True)
        report2_df = pd.DataFrame(report2).transpose()
        st.dataframe(report2_df, height=212, width=750)

        # DISPLAYING CONFUSION MATRIX
        # Compute confusion matrix
        cm = confusion_matrix(Y_test, rand_pred)
        cm_df = pd.DataFrame(cm)#converting confusion matrix to dataframe

        # Plot heatmap
        st.header('CONFUSION MATRIX')
        sns.heatmap(cm_df, annot=True, cmap='Greens')
        st.pyplot()

    with t3:
        # DECISION TREE
        # TRAINING THE MODEL
        tree = DecisionTreeClassifier()
        tree.fit(X_train, Y_train)
        tree_pred = tree.predict(X_test)
        tree_res = tree.predict(sc.transform((new)))

        # DISPLAYING RESULT
        if tree_res == 1:
            st.error('POSSIBILITY OF BRAIN STROKE DETECTED : PLEASE CONSULT A DOCTOR')
        else:
            st.success('NO POSSIBILITY OF BRAIN STROKE DETECTED')
        # DISPLAYING MODEL PARAMETERS
        st.header('MODEL PARAMETERS')
        st.warning('ACCURACY : {0}%'.format(round(accuracy_score(Y_test, tree_pred), 2) * 100))
        st.success('PRECISION : {0}%'.format(round(precision_score(Y_test, tree_pred), 2) * 100))
        st.error('F1 SCORE : {0}%'.format(round(f1_score(Y_test, tree_pred), 2) * 100))
        st.info('RECALL SCORE : {0}%'.format(round(recall_score(Y_test, tree_pred), 2) * 100))

        # DISPLAYING CLASSIFICATION REPORT
        st.header('CLASSIFICATION REPORT')
        report3 = classification_report(Y_test, tree_pred, output_dict=True)
        report3_df = pd.DataFrame(report3).transpose()
        st.dataframe(report3_df, height=212, width=750)

        # DISPLAYING CONFUSION MATRIX
        # Compute confusion matrix
        cm = confusion_matrix(Y_test, rand_pred)
        cm_df = pd.DataFrame(cm)

        # Plot heatmap
        st.header('CONFUSION MATRIX')
        sns.heatmap(cm_df, annot=True, cmap='Reds')
        st.pyplot()

    with t4:
        # SVM CLASSIFIER
        # TRAINING THE MODEL
        svc = SVC()
        svc.fit(X_train, Y_train)
        svc_pred = svc.predict(X_test)
        svc_res = svc.predict(sc.transform((new)))

        # DISPLAYING RESULT
        if svc_res == 1:
            st.error('POSSIBILITY OF BRAIN STROKE DETECTED : PLEASE CONSULT A DOCTOR')
        else:
            st.success('NO POSSIBILITY OF BRAIN STROKE DETECTED')
        # DISPLAYING MODEL PARAMETERS
        st.header('MODEL PARAMETERS')
        st.warning('ACCURACY : {0}%'.format(round(accuracy_score(Y_test, svc_pred), 2) * 100))
        st.success('PRECISION : {0}%'.format(round(precision_score(Y_test, svc_pred), 2) * 100))
        st.error('F1 SCORE : {0}%'.format(round(f1_score(Y_test, svc_pred), 2) * 100))
        st.info('RECALL SCORE : {0}%'.format(round(recall_score(Y_test, svc_pred), 2) * 100))

        # DISPLAYING CLASSIFICATION REPORT
        st.header('CLASSIFICATION REPORT')
        report4 = classification_report(Y_test, svc_pred, output_dict=True)
        report4_df = pd.DataFrame(report4).transpose()
        st.dataframe(report4_df, height=212, width=750)

        # DISPLAYING CONFUSION MATRIX
        # Compute confusion matrix
        cm = confusion_matrix(Y_test, rand_pred)
        cm_df = pd.DataFrame(cm)

        # Plot heatmap
        st.header('CONFUSION MATRIX')
        sns.heatmap(cm_df, annot=True, cmap='Purples')
        st.pyplot()

    with t5:
        # MODEL GRAPHS
        data_graph = pd.read_csv('brain_stroke.csv')  # loading the dataset
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.subheader("CATEGORICAL FEATURES COUNTPLOT")
        plt.figure()
        ax = sns.countplot(x='gender', hue='stroke', data=data_graph)
        ax.set_title('Gender count-plot')
        st.pyplot()

        plt.figure()
        ax = sns.countplot(x='hypertension', hue='stroke', data=data_graph)
        ax.set_title('Hypertension count-plot')
        st.pyplot()

        plt.figure()
        ax = sns.countplot(x='heart_disease', hue='stroke', data=data_graph)
        ax.set_title('Heart Disease count-plot')
        st.pyplot()

        plt.figure()
        ax = sns.countplot(x='ever_married', hue='stroke', data=data_graph)
        ax.set_title('Marriage History count-plot')
        st.pyplot()

        plt.figure()
        ax = sns.countplot(x='work_type', hue='stroke', data=data_graph)
        ax.set_title('Work Type count-plot')
        st.pyplot()

        plt.figure()
        ax = sns.countplot(x='Residence_type', hue='stroke', data=data_graph)
        ax.set_title('Residence Type count-plot')
        st.pyplot()

        st.subheader("MODELS COMPARSION")
        models = ["LOGISTIC", "RANDOM FOREST", "DECISION TREE", "SVM CLASSIFIER"]
        accuracies = [78, 97, 94, 89]
        precisions = [77, 99, 93, 90]

        # Accuracy plot
        fig1, ax1 = plt.subplots()
        ax1.bar(models, accuracies, color='red', width=0.4)
        ax1.set_ylim(0, 100)
        ax1.set_xlabel("Models")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy of different models")

        # Precision plot
        fig2, ax2 = plt.subplots()
        ax2.bar(models, precisions, color='blue', width=0.4)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("Models")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision of different models")

        # Display the plots on Streamlit
        st.pyplot(fig1)
        st.pyplot(fig2)

else:
    st.error('CLICK SUBMIT BUTTON')