#processing data
import pandas as pd

data = pd.read_csv("G7_crop.csv")  # loading the dataset

CROPS = {"rice": 1, "maize": 0,"chickpea":2, "kidneybeans":3, "pigeonpeas":4, "mothbeans":5}
data['label'] = data['label'].map(CROPS)
CROP_NAMES = [name for name, code in sorted(CROPS.items(), key=lambda item: item[1])]  # name for code 0, 1, 2...

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

#taking inputs from user on webpage
#
import streamlit as st
st.title("Crop Recommendation")
n = st.number_input("N")
p = st.number_input("P")
k = st.number_input("K")
temp = st.number_input("temperature",step=10.35)
humi = st.number_input("humidity",step=10.78)
ph = st.number_input("ph",step=1.98)
rainfall = st.number_input("rainfall",step=150.978)
new_user_input = [[n,p,k,temp,humi,ph,rainfall]]

from streamlit_option_menu import option_menu
with st.sidebar:
    option = option_menu("SELECT ONE OF THE FOLLOWING",options=["SVM","KNN","DECISION TREE","XGBOOST","MODEL GRAPHS"])
    submit = st.button("submit")

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






        # Extracting predicted probabilities for the positive class (class 1)
        positive_probs = svc_probabilistic[0]  # Assuming there's only one prediction for new_user_input

        # Generating labels for the pie chart
        # labels = [f'Class {i + 1}' for i in range(len(positive_probs))]

        # Plotting the pie chart
        import matplotlib.pyplot as plt

        st.title("RECOMMENDATION")
        if svc_new_user_input_prediction[0]==1:
            st.info("predicted crop is rice")
        elif svc_new_user_input_prediction[0]==0:
            st.info("predicted crop is maize")
        elif svc_new_user_input_prediction[0] == 2:
            st.info("predicted crop is chickpea")
        elif svc_new_user_input_prediction[0] == 3:
            st.info("predicted crop is kidneybeans")
        elif svc_new_user_input_prediction[0] == 4:
            st.info("predicted crop is pigeonpeas")
        elif svc_new_user_input_prediction[0] == 5:
            st.info("predicted crop is mothbeans ")
        fig, ax = plt.subplots()
        ax.pie(svc_probabilistic[0], labels=CROP_NAMES, autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        # getting accuracy ,precision of the model
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, svc_test_prediction)}")

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

        st.title("RECOMMENDATION")
        if knn_new_user_input_prediction[0]==1:
            st.info("predicted crop is rice")
        elif knn_new_user_input_prediction[0]==0:
            st.info("predicted crop is maize")
        elif knn_new_user_input_prediction[0] == 2:
            st.info("predicted crop is chickpea")
        elif knn_new_user_input_prediction[0] == 3:
            st.info("predicted crop is kidneybeans")
        elif knn_new_user_input_prediction[0] == 4:
            st.info("predicted crop is pigeonpeas")
        elif knn_new_user_input_prediction[0] == 5:
            st.info("predicted crop is mothbeans ")
        fig, ax = plt.subplots()
        ax.pie(knn_probabilistic[0], labels=CROP_NAMES, autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, knn_test_prediction)}")

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

        st.title("RECOMMENDATION")
        if  dt_new_user_input_prediction[0] == 1:
            st.info("predicted crop is rice")
        elif  dt_new_user_input_prediction[0] == 0:
            st.info("predicted crop is maize")
        elif  dt_new_user_input_prediction[0] == 2:
            st.info("predicted crop is chickpea")
        elif  dt_new_user_input_prediction[0] == 3:
            st.info("predicted crop is kidneybeans")
        elif  dt_new_user_input_prediction[0] == 4:
            st.info("predicted crop is pigeonpeas")
        elif  dt_new_user_input_prediction[0] == 5:
            st.info("predicted crop is mothbeans ")

        fig, ax = plt.subplots()
        ax.pie(dt_probabilistic[0], labels=CROP_NAMES, autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")


        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, dt_test_prediction)}")

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

        st.title("RECOMMENDATION")
        if xgb_new_user_input_prediction[0] == 1:
            st.info("predicted crop is rice")
        elif  xgb_new_user_input_prediction[0] == 0:
            st.info("predicted crop is maize")
        elif  xgb_new_user_input_prediction[0] == 2:
            st.info("predicted crop is chickpea")
        elif  xgb_new_user_input_prediction[0] == 3:
            st.info("predicted crop is kidneybeans")
        elif  xgb_new_user_input_prediction[0] == 4:
            st.info("predicted crop is pigeonpeas")
        elif  xgb_new_user_input_prediction[0] == 5:
            st.info("predicted crop is mothbeans ")
        fig, ax = plt.subplots()
        ax.pie(xgb_probabilistic[0], labels=CROP_NAMES, autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, xgb_test_prediction)}")


    elif option == "MODEL GRAPHS":
        import matplotlib.pyplot as plt

        from sklearn.metrics import accuracy_score, precision_score
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from xgboost import XGBClassifier
        # scores are computed from the models on the test set, not typed in
        candidates = {"SVM": SVC(), "KNN": KNeighborsClassifier(), "Decision Tree": DecisionTreeClassifier(random_state=0), "XGBoost": XGBClassifier()}
        models, accuracies, precisions = [], [], []
        for name, model in candidates.items():
            pred = model.fit(x_train, y_train).predict(x_test)
            models.append(name)
            accuracies.append(accuracy_score(y_test, pred) * 100)
            precisions.append(precision_score(y_test, pred, average='macro', zero_division=0) * 100)

        for values, metric, color in [(accuracies, 'Accuracy', 'red'), (precisions, 'Precision', None)]:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(models, values, color=color)
            ax.set_ylim(0, 100)
            ax.set_xlabel('Models')
            ax.set_ylabel(metric + ' (%)')
            ax.set_title(metric + ' Comparison')
            st.pyplot(fig)

        st.subheader(f"Best accuracy on the test set: {models[accuracies.index(max(accuracies))]}")
else:
    st.error("CLICK SUBMIT BUTTON")
