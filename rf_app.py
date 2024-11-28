import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Streamlit App
st.title("Random Forest Classifier with Feature Importance Visualization")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load the data
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(data.head())

        # Step 2: Select Target Column
        target_column = st.selectbox("Select the Target Column", options=data.columns)
        
        if target_column:
            # Split into features (X) and target (y)
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Step 3: Train-Test Split
            test_size = st.slider("Select Test Size (as a percentage)", 10, 50, 20) / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            st.write("### Dataset Split")
            st.write(f"Training Data: {X_train.shape[0]} rows")
            st.write(f"Test Data: {X_test.shape[0]} rows")

            # Step 4: Train Random Forest Model
            n_estimators = st.slider("Number of Trees in Random Forest", 10, 500, 100)
            max_depth = st.slider("Maximum Depth of Trees (set None for no limit)", 1, 50, 10)

            if st.button("Train Model"):
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)

                # Step 5: Evaluate Model
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)
                st.write("### Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                # Step 6: Display Feature Importance
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": feature_importances
                }).sort_values(by="Importance", ascending=False)

                st.write("### Feature Importances")
                st.dataframe(importance_df)

                # Plot Feature Importances
                st.write("### Feature Importance Visualization")
                plt.figure(figsize=(10, 6))
                sns.barplot(x="Importance", y="Feature", data=importance_df)
                plt.title("Feature Importance")
                plt.tight_layout()
                st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"An error occurred: {e}")
