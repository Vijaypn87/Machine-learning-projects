#processing data
import pandas as pd

data = pd.read_csv("G6_survey_lung_cancer.csv")  # loading the dataset

data['LUNG_CANCER'] = data['LUNG_CANCER'].map({"YES": 1, "NO": 0})

data['GENDER'] = data['GENDER'].map({"M":1,"F":2})

print(data.columns[data.isna().any()])  # checking for NAN values

# seperate the dataset into input and output values
x = data.iloc[:, :-1].values  # choosing input columns
y = data.iloc[:, -1].values  # choosing output columns

# spliting dataset into training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=45, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(x.shape)

import streamlit as st

st.title("LUNG CANCER PREDICTION")

gender = st.selectbox("GENDER",options=["M","F"])
if gender == "M":
    gender=1
else:
    gender=0
age = st.number_input("ENTER YOUR AGE",step=1)
smoking = st.selectbox("SMOKING",options=[1,2])
yellowfinger = st.selectbox("YELLOW_FINGERS",options=[1,2])
anxiety = st.selectbox("ANXIETY",options=[1,2])
peerpressure = st.selectbox("PEER_PRESSURE",options=[1,2])
chronicdisease = st.selectbox("CHRONIC DISEASE",options=[1,2])
fatigue = st.selectbox("FATIGUE",options=[1,2])
allergy = st.selectbox("ALLERGY",options=[1,2])
wheezing = st.selectbox("WHEEZING",options=[1,2])
alcoholconsuming = st.selectbox("ALCOHOL CONSUMING",options=[1,2])
coughing = st.selectbox("COUGHING",options=[1,2])
shortnessofbreath = st.selectbox("SHORTNESS OF BREATH",options=[1,2])
swallowingdifficult = st.selectbox("SWALLOWING DIFFICULT",options=[1,2])
chestpain = st.selectbox("CHEST PAIN",options=[1,2])

new_user_input = [[age,gender,smoking,yellowfinger,
                   anxiety,peerpressure,chronicdisease,
                   fatigue,allergy,wheezing,alcoholconsuming,
                   coughing,shortnessofbreath,swallowingdifficult,
                   chestpain]]

from streamlit_option_menu import option_menu
with st.sidebar:
    option = option_menu("SELECT ONE OF THE FOLLOWING",options=["SVM","KNN","DECISION TREE","XGBOOST","MODEL GRAPHS"])
    submit = st.button("submit")
#
if submit:
# SVM
    if option == "SVM":
        from sklearn.svm import SVC

        svc = SVC(probability=True)
        svc.fit(x_train, y_train)

        svc_test_prediction = svc.predict(x_test)
        svc_new_user_input_prediction = svc.predict(new_user_input)  # predicting result for new user input
        svc_probabilistic = svc.predict_proba(new_user_input)  # getting probabilities for each class
        #
        # st.info(svc_probabilistic)

        # Extracting predicted probabilities for the positive class (class 1)
        positive_probs = svc_probabilistic[0]  # Assuming there's only one prediction for new_user_input

        # Generating labels for the pie chart
        # labels = [f'Class {i + 1}' for i in range(len(positive_probs))]

        # Plotting the pie chart
        import matplotlib.pyplot as plt

        st.title("DIAGONISIS")

        fig, ax = plt.subplots()
        ax.pie(svc_probabilistic[0], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        # getting accuracy ,precision of the model
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, svc_test_prediction)}")
        st.error(f"PRECISION OF THE MODEL : {precision_score(y_test, svc_test_prediction)}")
        st.warning(f"RECALL SCORE OF THE MODEL IS : {recall_score(y_test, svc_test_prediction)}")
        st.info(f"F1 SCORE OF THE MODEL IS : {f1_score(y_test, svc_test_prediction)}")
# KNN
    elif option == "KNN":
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)

        knn_test_prediction = knn.predict(x_test)
        knn_new_user_input_prediction = knn.predict(new_user_input)  # predicting result for new user input
        knn_probabilistic = knn.predict_proba(new_user_input)  # getting probabilities for each class

        positive_probs = knn_probabilistic[0]  # Assuming there's only one prediction for new_user_input

        import matplotlib.pyplot as plt

        st.title("DIAGONISIS")

        fig, ax = plt.subplots()
        ax.pie(knn_probabilistic[0], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, knn_test_prediction)}")
        st.error(f"PRECISION OF THE MODEL : {precision_score(y_test, knn_test_prediction)}")
        st.warning(f"RECALL SCORE OF THE MODEL IS : {recall_score(y_test, knn_test_prediction)}")
        st.info(f"F1 SCORE OF THE MODEL IS : {f1_score(y_test, knn_test_prediction)}")
# DECISION TREE
    elif option == "DECISION TREE":
        from sklearn.tree import DecisionTreeClassifier

        dt = DecisionTreeClassifier()
        dt.fit(x_train, y_train)

        dt_test_prediction = dt.predict(x_test)
        dt_new_user_input_prediction = dt.predict(new_user_input)  # predicting result for new user input
        dt_probabilistic = dt.predict_proba(new_user_input)

        positive_probs = dt_probabilistic[0]  # Assuming there's only one prediction for new_user_input

        import matplotlib.pyplot as plt

        st.title("DIAGONISIS")
        st.info(dt_new_user_input_prediction)

        fig, ax = plt.subplots()
        ax.pie(dt_probabilistic[0], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, dt_test_prediction)}")
        st.error(f"PRECISION OF THE MODEL : {precision_score(y_test, dt_test_prediction)}")
        st.warning(f"RECALL SCORE OF THE MODEL IS : {recall_score(y_test, dt_test_prediction)}")
        st.info(f"F1 SCORE OF THE MODEL IS : {f1_score(y_test, dt_test_prediction)}")
# XGBOOST
    elif option == "XGBOOST":
        from xgboost import XGBClassifier
        xgb = XGBClassifier()
        xgb.fit(x_train, y_train)

        xgb_test_prediction = xgb.predict(x_test)
        xgb_new_user_input_prediction = xgb.predict(new_user_input)  # predicting result for new user input
        xgb_probabilistic = xgb.predict_proba(new_user_input)

        positive_probs = xgb_probabilistic[0]  # Assuming there's only one prediction for new_user_input

        import matplotlib.pyplot as plt
        st.info(xgb_new_user_input_prediction)


        st.title("DIAGONISIS")

        fig, ax = plt.subplots()
        ax.pie(xgb_probabilistic[0], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, xgb_test_prediction)}")
        st.error(f"PRECISION OF THE MODEL : {precision_score(y_test,xgb_test_prediction)}")
        st.warning(f"RECALL SCORE OF THE MODEL IS : {recall_score(y_test, xgb_test_prediction)}")
        st.info(f"F1 SCORE OF THE MODEL IS : {f1_score(y_test, xgb_test_prediction)}")

    elif option == "MODEL GRAPHS":
        import matplotlib.pyplot as plt

        models = ["SVM", "KNN", "Decision Tree", "XGBoost"]
        accuracies = [87.0,87.0,92.4,91.9]
        precisions = [87.0,88.3,85.4,96.2]

        # Create a bar plot for accuracies
        plt.figure(figsize=(8, 6))
        plt.bar(models, accuracies,color='red')
        plt.xlabel('Models')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Comparison')
        st.pyplot(plt)

        # Create a bar plot for precisions
        plt.figure(figsize=(8, 6))
        plt.bar(models, precisions)
        plt.xlabel('Models')
        plt.ylabel('Precision (%)')
        plt.title('Precision Comparison')
        st.pyplot(plt)

        st.subheader("By Visual Comparison XGBOOST is th best")
else:
    st.error("CLICK SUBMIT BUTTON")