import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page configuration
st.set_page_config(page_title="Pumpkin Seeds ML Dashboard", layout="wide")

st.title("🎃 Pumpkin Seeds Regression Dashboard")

# File upload
uploaded_file = st.file_uploader("📁 Upload Pumpkin Seeds Dataset (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe())

    # Select numerical features and target
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("🎯 Choose Target Column")
    target_column = st.selectbox("Select the target column for regression", options=numeric_cols)

    features = [col for col in numeric_cols if col != target_column]
    X = df[features]
    y = df[target_column]

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model selection
    st.subheader("🔍 Choose a Regression Model")
    model_option = st.selectbox("Select a model", ["Linear Regression", "Decision Tree", "Random Forest", "SVR", "KNN Regression"])

    if model_option == "Linear Regression":
        model = LinearRegression()
        color = 'green'
    elif model_option == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=6, random_state=0)
        color = 'orange'
    elif model_option == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        color = 'teal'
    elif model_option == "SVR":
        model = SVR(kernel='rbf')
        color = 'purple'
    elif model_option == "KNN Regression":
        model = KNeighborsRegressor(n_neighbors=5)
        color = 'blue'

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader(f"📌 {model_option}")
    st.write("R2 Score:", round(r2_score(y_test, y_pred), 4))
    st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

    if model_option in ["Decision Tree", "Random Forest"]:
        st.subheader("📊 Feature Importance")
        importance = pd.Series(model.feature_importances_, index=features)
        fig, ax = plt.subplots()
        importance.sort_values().plot(kind='barh', color=color, ax=ax)
        st.pyplot(fig)

    # User prediction input
    st.markdown("---")
    st.subheader("🎯 Predict Using Custom Input")
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(feature, value=float(df[feature].mean()), format="%.2f")
    
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)[0]

    st.success(f"Predicted {target_column}: {round(prediction, 2)}")

    # Prediction graph
    st.markdown("### 📊 Prediction Graph")
    fig_pred, ax_pred = plt.subplots(figsize=(4, 2))
    ax_pred.barh([target_column], [prediction], color=color)
    ax_pred.set_xlim(0, df[target_column].max() + 10)
    ax_pred.set_title("Predicted Output")
    st.pyplot(fig_pred)

    # Model comparison
    st.markdown("---")
    st.subheader("📊 Compare All Regression Models")
    if st.button("Compare Models"):
        model_results = []
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=0),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
            "SVR": SVR(kernel='rbf'),
            "KNN Regression": KNeighborsRegressor(n_neighbors=5)
        }

        for name, m in models.items():
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            model_results.append([name,
                                  r2_score(y_test, preds),
                                  mean_absolute_error(y_test, preds),
                                  np.sqrt(mean_squared_error(y_test, preds))])

        results_df = pd.DataFrame(model_results, columns=["Model", "R2 Score", "MAE", "RMSE"])
        best_model = results_df.sort_values(by="R2 Score", ascending=False).iloc[0]["Model"]

        st.dataframe(results_df)

        st.markdown("### 📈 R2 Score Comparison")
        fig_cmp, ax_cmp = plt.subplots()
        sns.barplot(data=results_df, x="R2 Score", y="Model", palette="coolwarm", ax=ax_cmp)
        st.pyplot(fig_cmp)

        st.success(f"🏆 Best Performing Model: {best_model}")

else:
    st.info("📅 Please upload your Pumpkin_Seeds_Dataset.xlsx to begin.")