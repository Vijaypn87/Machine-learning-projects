#processing data
import pandas as pd
import chardet

data = pd.read_csv("G2_landslide.csv")

print(data.columns[data.isna().any()])  # checking for NAN values

# seperate the dataset into input and output values
x = data.iloc[:, 1:].values  # choosing input columns
y = data['Landslide'].values  # choosing output columns

# spliting dataset into training and testing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=45, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(x.shape)

#taking inputs from user on webpage
#
import streamlit as st
st.title("LANDSLIDE PREDICTION")


Aspect = st.number_input("Aspect")
#
Curvature = st.number_input("Curvature")

Earthquake = st.number_input("Earthquake")
#
Elevation = st.number_input("Elevation")

#
Flow = st.number_input("Flow")
#
Lithology = st.number_input("Lithology")
#
NDVI = st.number_input("NDVI")
#
NDWI = st.number_input("NDWI")
#
Plan = st.number_input("Plan")
#
Precipitation = st.number_input("Precipitation")
#
Profile = st.number_input("Profile")
#
Slope = st.number_input("Slope")
#
temperature = st.number_input("temperature")
#
humidity = st.number_input("humidity")
#
rain = st.number_input("rain")
#
moisture = st.number_input("moisture")
#
pressure = st.number_input("pressure")
#
new_user_input = [[Aspect,Curvature,Earthquake,Elevation,Flow,Lithology,NDVI,NDWI,Plan,Precipitation,Profile,Slope,temperature,humidity,rain,moisture,pressure]]
#

from streamlit_option_menu import option_menu
with st.sidebar:
    option = option_menu("SELECT ONE OF THE FOLLOWING",options=["RANDOM FOREST","KNN","DECISION TREE","XGBOOST","MODEL GRAPHS"])
    submit = st.button("submit")
#
if submit:
# RANDOM FOREST
    if option == "RANDOM FOREST":
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        rf_test_data_predictor = rf.predict(x_test)


        rf_test_prediction = rf.predict(x_test)
        rf_new_user_input_prediction = rf.predict(new_user_input)  # predicting result for new user input
        rf_probabilistic = rf.predict_proba(new_user_input)  # getting probabilities for each class
        #

        # Extracting predicted probabilities for the positive class (class 1)
        positive_probs = rf_probabilistic[0]  # Assuming there's only one prediction for new_user_input

        # Generating labels for the pie chart
        # labels = [f'Class {i + 1}' for i in range(len(positive_probs))]

        # Plotting the pie chart
        import matplotlib.pyplot as plt

        st.title("LANDSLIDE")
        if rf_new_user_input_prediction == 0:
            st.success("NO LANDSLIDE")
        else:
            st.error("POSSIBILITY OF LANDSLIDE")

        fig, ax = plt.subplots()
        ax.pie(rf_probabilistic[0], autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axis('equal')

        # Displaying the pie chart in Streamlit
        st.pyplot(fig)

        st.title("MODEL PARAMETERS")

        # getting accuracy ,precision of the model
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        st.success(f"ACCURACY OF THE MODEL : {accuracy_score(y_test, rf_test_prediction)}")
        st.error(f"PRECISION OF THE MODEL : {precision_score(y_test, rf_test_prediction)}")
        st.warning(f"RECALL SCORE OF THE MODEL IS : {recall_score(y_test, rf_test_prediction)}")
        st.info(f"F1 SCORE OF THE MODEL IS : {f1_score(y_test, rf_test_prediction)}")
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

        st.title("LANDSLIDE")
        if knn_new_user_input_prediction == 0:
            st.success("NO LANDSLIDE")
        else:
            st.error("POSSIBILITY OF LANDSLIDE")

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

        st.title("LANDSLIDE")

        if dt_new_user_input_prediction == 0:
            st.success("NO LANDSLIDE")
        else:
            st.error("POSSIBILITY OF LANDSLIDE")

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

        st.title("LANDSLIDE")

        if xgb_new_user_input_prediction == 0:
            st.success("NO LANDSLIDE")
        else:
            st.error("POSSIBILITY OF LANDSLIDE")

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
        accuracies = [57.4, 66.6, 66.6, 81.4]
        precisions = [61.5, 68.1, 65.3, 86.3]

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
