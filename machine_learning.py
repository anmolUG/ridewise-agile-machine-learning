import streamlit as st
import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.graph_objects as go
# Import additional classifiers
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Custom CSS Styling - removed glass effect
st.set_page_config(page_title="RideWise ML Analysis", layout="centered")

# Apply styling without glass-box
st.markdown("""
    <style>
    /* Animated gradient for the title */
    .gradient-text {
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-flow 3s infinite alternate;
    }

    @keyframes gradient-flow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    /* Standard section styling instead of glass */
    .section-container {
        padding: 1.5rem 0;
        margin: 1rem 0;
        border-bottom: 1px solid #f0f0f0;
    }

    /* Override tab underline to blue */
    div[data-baseweb="tab"] button[aria-selected="true"] {
        border-bottom: 3px solid #2575fc;
        color: #2575fc;
    }

    /* Button styling */
    div.stButton > button {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75em 1em;
        width: 100%;
    }

    /* Remove unwanted top empty box */
    section[data-testid="stTabs"] > div:first-child {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def loadData():
    df = pd.read_csv("2010-capitalbikeshare-tripdata.csv")
    return df

# Basic preprocessing required for all the models.  
def preprocessing(df):
    X = df.iloc[:, [0, 3, 5]].values
    y = df.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y.flatten())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test, le


@st.cache_resource
def decisionTree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, tree


# @st.cache_resource
# def neuralNet(X_train, X_test, y_train, y_test):
#     scaler = StandardScaler()  
#     scaler.fit(X_train)  
#     X_train = scaler.transform(X_train)  
#     X_test = scaler.transform(X_test)

#     clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     score1 = metrics.accuracy_score(y_test, y_pred) * 100
#     report = classification_report(y_test, y_pred)
#     return score1, report, clf


@st.cache_resource
def Knn_Classifier(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf


@st.cache_resource
def svm_classifier(X_train, X_test, y_train, y_test):
    # Scale data for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = SVC(kernel='rbf', random_state=0)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf, scaler


@st.cache_resource
def naive_bayes_classifier(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf


@st.cache_resource
def random_forest_classifier(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf


@st.cache_resource
def logistic_regression_classifier(X_train, X_test, y_train, y_test):
    # Scale data for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, clf, scaler


def accept_user_data():
    duration = st.text_input("Enter the Duration: ")
    start_station = st.text_input("Enter the start station number: ")
    end_station = st.text_input("Enter the end station number: ")
    user_prediction_data = np.array([duration, start_station, end_station]).reshape(1, -1)
    return user_prediction_data


@st.cache_resource
def showMap():
    plotData = pd.read_csv("Trip history with locations.csv")
    Data = pd.DataFrame()
    Data['lat'] = plotData['lat']
    Data['lon'] = plotData['lon']
    return Data


def compare_all_models(X_train, X_test, y_train, y_test):
    st.markdown('<div class="gradient-text">Machine Learning Model Comparison Dashboard</div>', unsafe_allow_html=True)
    st.write("Comparing the performance of all machine learning models")
    
    # Create a progress bar to show the model training progress
    progress_bar = st.progress(0)
    
    # Create a dictionary to store all model accuracies
    all_models = {
        "Decision Tree": None,
        "K-Nearest Neighbors": None,
        "SVM": None,
        "Naive Bayes": None,
        "Random Forest": None,
        "Logistic Regression": None
    }
    
    # Train all models and get accuracies
    with st.spinner("Training Decision Tree..."):
        score, _, _ = decisionTree(X_train, X_test, y_train, y_test)
        all_models["Decision Tree"] = score
        progress_bar.progress(14)
    
    # with st.spinner("Training Neural Network..."):
    #     score, _, _ = neuralNet(X_train, X_test, y_train, y_test)
    #     all_models["Neural Network"] = score
    #     progress_bar.progress(28)
    
    with st.spinner("Training K-Nearest Neighbors..."):
        score, _, _ = Knn_Classifier(X_train, X_test, y_train, y_test)
        all_models["K-Nearest Neighbors"] = score
        progress_bar.progress(42)
    
    with st.spinner("Training SVM..."):
        score, _, _, _ = svm_classifier(X_train, X_test, y_train, y_test)
        all_models["SVM"] = score
        progress_bar.progress(56)
    
    with st.spinner("Training Naive Bayes..."):
        score, _, _ = naive_bayes_classifier(X_train, X_test, y_train, y_test)
        all_models["Naive Bayes"] = score
        progress_bar.progress(70)
    
    with st.spinner("Training Random Forest..."):
        score, _, _ = random_forest_classifier(X_train, X_test, y_train, y_test)
        all_models["Random Forest"] = score
        progress_bar.progress(84)
    
    with st.spinner("Training Logistic Regression..."):
        score, _, _, _ = logistic_regression_classifier(X_train, X_test, y_train, y_test)
        all_models["Logistic Regression"] = score
        progress_bar.progress(100)
    
    # Convert dictionary to dataframe for display
    df_models = pd.DataFrame(list(all_models.items()), columns=['Model', 'Accuracy (%)'])
    
    # Sort by accuracy
    df_models = df_models.sort_values(by='Accuracy (%)', ascending=False)
    
    # Display the model comparison table
    st.subheader("Model Accuracy Comparison")
    st.dataframe(df_models, use_container_width=True)
    
    # Create a bar chart comparing model accuracies
    fig = px.bar(
        df_models, 
        x='Model', 
        y='Accuracy (%)',
        color='Accuracy (%)',
        color_continuous_scale='viridis',
        title='Model Accuracy Comparison',
        text='Accuracy (%)'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify the best performing model
    best_model = df_models.iloc[0]['Model']
    best_accuracy = df_models.iloc[0]['Accuracy (%)']
    
    st.success(f"The best performing model is **{best_model}** with an accuracy of **{best_accuracy:.2f}%**")
    
    # Add recommendations based on model performance
    st.subheader("Recommendations")
    st.write("""
    - **High Accuracy Models:** Consider using these for your production environment
    - **Low Accuracy Models:** May need parameter tuning or more features
    - **Model Selection:** Choose based on both accuracy and interpretability needs
    """)
    
    # Add a button to return to the main page
    if st.button("Return to Main Page"):
        st.session_state.page = "main"
        st.rerun()


def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "main"
        
    # Load data and preprocessing (common for all pages)
    data = loadData()
    X_train, X_test, y_train, y_test, le = preprocessing(data)
    
    # Navigation logic
    if st.session_state.page == "compare_models":
        compare_all_models(X_train, X_test, y_train, y_test)
    else:  # Main page
        st.markdown('<div class="gradient-text">RideWise: Predicting Bike Trip Membership Types</div>', unsafe_allow_html=True)
        
        # Show Raw Data section
        if st.checkbox('Show Raw Data'):
            st.subheader("Raw Data Sample:")
            st.write(data.head())
            st.markdown('<hr>', unsafe_allow_html=True)
        
        # Model selection
        choose_model = st.sidebar.selectbox("Choose the ML Model",
            ["NONE", "Decision Tree", "K-Nearest Neighbours", 
             "SVM", "Naive Bayes", "Random Forest", "Logistic Regression"])
        
        # Add a button for model comparison
        if st.sidebar.button("Compare All Models"):
            st.session_state.page = "compare_models"
            st.rerun()
        
        # Individual model handling - no glass boxes
        if choose_model != "NONE":
            st.subheader(f"{choose_model} Model Analysis")
            
            if choose_model == "Decision Tree":
                score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
                st.text("Accuracy of Decision Tree model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()         
                        if st.button("Predict"):
                            pred = tree.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
        
            # elif choose_model == "Neural Network":
            #     score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
            #     st.text("Accuracy of Neural Network model:")
            #     st.write(score, "%")
            #     st.text("Classification Report:")
            #     st.text(report)
        
            #     try:
            #         if(st.checkbox("Want to predict on your own input?")):
            #             st.info("It is recommended to look at the dataset before entering values")
            #             user_prediction_data = accept_user_data()
            #             if st.button("Predict"):
            #                 scaler = StandardScaler()  
            #                 scaler.fit(X_train)  
            #                 user_prediction_data = scaler.transform(user_prediction_data)    
            #                 pred = clf.predict(user_prediction_data)
            #                 st.write("The Predicted Class is: ", le.inverse_transform(pred))
            #     except Exception as e:
            #         st.error(f"Error: {e}. Please enter valid values.")
        
            elif choose_model == "K-Nearest Neighbours":
                score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of KNN model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
            
            elif choose_model == "SVM":
                score, report, clf, scaler = svm_classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of SVM model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            user_prediction_data = scaler.transform(user_prediction_data)
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
            
            elif choose_model == "Naive Bayes":
                score, report, clf = naive_bayes_classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of Naive Bayes model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
            
            elif choose_model == "Random Forest":
                score, report, clf = random_forest_classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of Random Forest model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
            
            elif choose_model == "Logistic Regression":
                score, report, clf, scaler = logistic_regression_classifier(X_train, X_test, y_train, y_test)
                st.text("Accuracy of Logistic Regression model:")
                st.write(score, "%")
                st.text("Classification Report:")
                st.text(report)
        
                try:
                    if(st.checkbox("Want to predict on your own input?")):
                        st.info("It is recommended to look at the dataset before entering values")
                        user_prediction_data = accept_user_data()
                        if st.button("Predict"):
                            user_prediction_data = scaler.transform(user_prediction_data)
                            pred = clf.predict(user_prediction_data)
                            st.write("The Predicted Class is: ", le.inverse_transform(pred))
                except Exception as e:
                    st.error(f"Error: {e}. Please enter valid values.")
            
            # Add a separator
            st.markdown('<hr>', unsafe_allow_html=True)
        
        # Map and Visualization - no glass boxes
        st.subheader("Map of Bike Trip Start Locations")
        try:
            plotData = showMap()
            st.map(plotData, zoom=14)
        except Exception as e:
            st.warning(f"Could not load map data. Make sure the CSV file is in the correct location.")
            st.error(f"Error details: {e}")
    
        choose_viz = st.sidebar.selectbox("Choose Visualization",
            ["NONE", "Total number of vehicles from various Starting Points",
             "Total number of vehicles from various End Points",
             "Count of each Member Type"])
        
        if choose_viz != "NONE":
            st.subheader(choose_viz)
            if choose_viz == "Total number of vehicles from various Starting Points":
                fig = px.histogram(data['Start station'], x='Start station')
                st.plotly_chart(fig)
            elif choose_viz == "Total number of vehicles from various End Points":
                fig = px.histogram(data['End station'], x='End station')
                st.plotly_chart(fig)
            elif choose_viz == "Count of each Member Type":
                fig = px.histogram(data['Member type'], x='Member type')
                st.plotly_chart(fig)
            
    # Add back button
    st.sidebar.markdown("---")
    if st.sidebar.button("Back to Main App"):
        st.sidebar.info("Redirecting back to main application...")
        other_app_url = "http://localhost:8501/"
        st.markdown(f'<meta http-equiv="refresh" content="0;url={other_app_url}">', unsafe_allow_html=True)

if __name__ == "__main__":
    main()