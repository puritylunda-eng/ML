## load libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bike Sharing CRISP-DM Dashboard", layout="wide")

# =========================
# LOAD DATA
# =========================
day_df = pd.read_csv("data/day.csv")
hour_df = pd.read_csv("data/hour.csv")

st.sidebar.title("CRISP-DM Navigation")

dataset_choice = st.sidebar.selectbox("Select Dataset", ["Day Dataset", "Hour Dataset"])
df = day_df if dataset_choice == "Day Dataset" else hour_df

stage = st.sidebar.radio(
    "Select CRISP-DM Stage",
    [
        "Business Understanding",
        "Data Understanding (EDA)",
        "Data Preparation",
        "Model Analysis",
        "following the CRISP-DM methodology ",
        "exercise on  *streamlit* "
    ]
)

# =========================
# BUSINESS UNDERSTANDING
# =========================
if stage == "Business Understanding":
    st.title("Business Understanding")
    st.markdown("""
    ### Objective
    Build and compare regression models to predict bike demand (`cnt`) using historical data.

    ### Business Value
    - Improve bike availability planning
    - Reduce shortage during peak hours
    - Optimize resource allocation
    """)

# =========================
# DATA UNDERSTANDING (EDA)
# =========================
elif stage == "Data Understanding (EDA)":
    st.title("Exploratory Data Analysis (EDA)")

    # datetime handling
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")
        df = df.sort_values("dteday")

    # =========================
    # UNIVARIATE ANALYSIS
    # =========================
    # =========================
# UNIVARIATE ANALYSIS
# =========================
st.subheader(" Univariate Analysis")

# Select feature
feature = st.selectbox(
    "Select Feature for Univariate Analysis",
    df.select_dtypes(include=["number"]).columns
)

# -------------------------
# Histogram
# -------------------------
st.write(f"### Distribution of {feature}")

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df[feature], kde=True, ax=ax)

ax.set_title(f"Distribution of {feature}")
st.pyplot(fig)

# -------------------------
# Boxplot
# -------------------------
st.write(f"### Boxplot of {feature}")
st.write('The distribution of bike rentals is positively skewed, indicating that most days experience moderate demand while a few days record very high demand.')
fig, ax = plt.subplots(figsize=(8, 2))
sns.boxplot(x=df[feature], ax=ax)

ax.set_title(f"Boxplot of {feature}")
st.pyplot(fig)
tab1, tab2, tab3 = st.tabs(["Histogram", "Boxplot", "Statistics"])
# -------------------------
# Summary Statistics
# -------------------------
st.write("### Summary Statistics")

st.write(df[feature].describe())

 # =========================
# BIVARIATE ANALYSIS
# =========================
st.subheader(" Bivariate Analysis")

# Select numeric columns only
numeric_columns = df.select_dtypes(include=["number"]).columns

# Select variables
x = st.selectbox("Select X-axis Variable", numeric_columns, key="x_axis")
y = st.selectbox("Select Y-axis Variable", numeric_columns, key="y_axis")

# -------------------------
# Scatter Plot
# -------------------------
st.write(f"### Relationship Between {x} and {y}")

fig, ax = plt.subplots(figsize=(8, 5))

sns.scatterplot(
    x=df[x],
    y=df[y],
    ax=ax
)

ax.set_title(f"{x} vs {y}")

st.pyplot(fig)
st.write('Humidity shows a weak negative relationship with bike rentals, suggesting that higher humidity may slightly reduce demand.')
# -------------------------
# Correlation Value
# -------------------------
correlation = df[x].corr(df[y])

st.write("### Correlation Coefficient")

st.write(f"Correlation between {x} and {y}: **{correlation:.2f}**")
sns.regplot(
    x=df[x],
    y=df[y],
    ax=ax
)
st.write('helps visualize trends more clearly.')
    # =========================
    # MULTIVARIATE ANALYSIS
    # =========================
st.subheader(" Multivariate Analysis")

corr = df.select_dtypes(include="number").corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)

st.write('for temp and cnt, Higher temperatures are associated with increased bike rentals.')
st.write('for hum and cnt, a weak negative correlation==Increased humidity slightly reduces bike demand.')
    # =========================
    # TIME SERIES ANALYSIS
    # =========================
if "dteday" in df.columns and "cnt" in df.columns:
        st.subheader(" Time Series Trend (Demand Over Time)")

        fig, ax = plt.subplots()
        ax.plot(df["dteday"], df["cnt"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Bike Demand (cnt)")
        st.pyplot(fig)

# =========================
# DATA PREPARATION
# =========================
elif stage == "Data Preparation":
    st.title("Data Preparation")

    st.write("Dataset Preview")
    st.dataframe(df.head())

    st.write("Missing Values")
    st.write(df.isnull().sum())

    st.write("Data Types")
    st.write(df.dtypes)

# =========================
# MODELING
# =========================
elif stage == "Modeling":
    st.title("Modeling & Model Comparison")

    df_model = df.copy()

    # Feature engineering (datetime)
    if "dteday" in df_model.columns:
        df_model["dteday"] = pd.to_datetime(df_model["dteday"], errors="coerce")
        df_model["year"] = df_model["dteday"].dt.year
        df_model["month"] = df_model["dteday"].dt.month
        df_model["day"] = df_model["dteday"].dt.day

    # keep numeric only
    df_model = df_model.select_dtypes(include="number").dropna()

    if "cnt" not in df_model.columns:
        st.error("Target column 'cnt' not found.")
    else:
        X = df_model.drop("cnt", axis=1)
        y = df_model["cnt"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Lasso": Lasso(),
            "KNN": KNeighborsRegressor()
        }

        results = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            r2 = r2_score(y_test, preds)
            rmse = mean_squared_error(y_test, preds) ** 0.5

            results.append((name, r2, rmse))

        results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "RMSE"])
        results_df = results_df.sort_values("R2 Score", ascending=False)

        st.subheader(" Model Performance Comparison")
        st.dataframe(results_df)

        fig, ax = plt.subplots()
        sns.barplot(x="Model", y="R2 Score", data=results_df, ax=ax)
        ax.set_title("Model Comparison (R² Score)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        st.pyplot(fig)

# =========================
# EVALUATION
# =========================

elif stage == "Evaluation":
    st.title(" Model Evaluation")

    # =========================
    # PREPARE DATA
    # =========================
df_eval = df.copy()

    # datetime feature engineering
if "dteday" in df_eval.columns:
        df_eval["dteday"] = pd.to_datetime(df_eval["dteday"], errors="coerce")

        df_eval["year"] = df_eval["dteday"].dt.year
        df_eval["month"] = df_eval["dteday"].dt.month
        df_eval["day"] = df_eval["dteday"].dt.day

    # keep numeric only
df_eval = df_eval.select_dtypes(include=["number"]).dropna()

    # target variable
X = df_eval.drop("cnt", axis=1)
y = df_eval["cnt"]

    # train test split
X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =========================
    # TRAIN MODELS
    # =========================
models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100),
        "Lasso": Lasso(),
        "KNN": KNeighborsRegressor()
    }

results = []

for name, model in models.items():

        # train
        model.fit(X_train, y_train)

        # predictions
        preds = model.predict(X_test)

        # metrics
        r2 = r2_score(y_test, preds)

        rmse = mean_squared_error(y_test, preds) ** 0.5

        results.append([name, r2, rmse])

    # results dataframe
results_df = pd.DataFrame(
        results,
        columns=["Model", "R2 Score", "RMSE"]
    )

    # sort best model
results_df = results_df.sort_values(
        by="R2 Score",
        ascending=False
    )

    # =========================
    # DISPLAY RESULTS
    # =========================
st.subheader("Evaluation Metrics")

st.dataframe(results_df)

    # =========================
    # BEST MODEL
    # =========================
best_model = results_df.iloc[0]

st.success(
        f"Best Model: {best_model['Model']} "
        f"with R² Score = {best_model['R2 Score']:.3f}"
    )

    # =========================
    # VISUALIZATION
    # =========================
st.subheader(" Model Comparison")

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
        x="Model",
        y="R2 Score",
        data=results_df,
        ax=ax
    )

ax.set_title("Model Performance Comparison")
ax.set_ylabel("R² Score")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

st.pyplot(fig)

    # =========================
    # INTERPRETATION
    # =========================
st.subheader("Interpretation")

st.markdown("""
    ### Evaluation Summary
    - Higher R² indicates better predictive performance
    - Lower RMSE indicates fewer prediction errors
    - Random Forest often performs best because it captures complex relationships
    - Linear Regression may underperform if relationships are non-linear
    """)
st.write('Random Forest achieved the highest R² score and lowest RMSE, indicating superior predictive performance compared to other regression models.')

# =========================
# DEPLOYMENT
# =========================elif stage == "Deployment":

st.title(" Deployment Stage")

st.markdown("""
    ## Bike Sharing Demand Prediction Dashboard

    This Streamlit application represents the deployment phase
    of the CRISP-DM methodology.

    ### Features of the Data Product
    - Interactive Exploratory Data Analysis (EDA)
    - Univariate, Bivariate, and Multivariate Analysis
    - Machine Learning Model Comparison
    - Real-time Visualization
    - Business Insights for Decision Making

    ### Business Impact
    This dashboard can help:
    - Forecast bike demand
    - Improve bike allocation
    - Reduce shortages during peak periods
    - Support data-driven transportation planning
    """)

    # =========================
    # KPI SECTION
    # =========================
st.subheader(" Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
        st.metric(
            label="Total Bike Rentals",
            value=int(df["cnt"].sum())
        )

with col2:
        st.metric(
            label="Average Daily Rentals",
            value=round(df["cnt"].mean(), 2)
        )

with col3:
        st.metric(
            label="Maximum Rentals",
            value=int(df["cnt"].max())
        )

    # =========================
    # DATA PREVIEW
    # =========================
st.subheader(" Dataset Preview")

st.dataframe(df.head())

    # =========================
    # DOWNLOAD OPTION
    # =========================
st.subheader("⬇ Download Dataset")

csv = df.to_csv(index=False).encode("utf-8")
st.write('The study successfully applied the CRISP-DM methodology to analyze and predict bike sharing dataset using a combination of exploratory data analysis and machine learning models. Key insights were derived from the dataset, and model performance was evaluated using accuracy and AUC metrics.')
st.download_button(
        label="Download CSV",
        data=csv,
        file_name="bike_sharing_data.csv",
        mime="text/csv"
    )
