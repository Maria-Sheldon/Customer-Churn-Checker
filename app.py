import json, joblib
import pandas as pd, plotly.express as px, streamlit as st

st.set_page_config(page_title="Churn Checker", page_icon="📉", layout="wide")
st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}

/* All text in white */
.stApp p,
.stApp span,
.stApp label,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp li,
.stApp div {
    color: #FFFFFF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f2238;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #171717;
    border-left: 4px solid #0d9488;
    padding: 10px;
    border-radius: 6px;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Dataframes and tables */
.stDataFrame {
    color: #FFFFFF;
}

/* Input fields */
.stSelectbox div,
.stNumberInput input {
    color: #FFFFFF;
}

</style>
""", unsafe_allow_html=True)
TEAL = ["#0d9488", "#0f2238"]

@st.cache_data
def load():
    d = pd.read_csv("telco_customer_churn.csv")
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    return d
@st.cache_resource
def model(): return joblib.load("churn_pipeline.pkl")
@st.cache_data
def metrics(): return json.load(open("metrics.json"))

df, M = load(), metrics()
st.sidebar.title("Churn Checker")
st.sidebar.caption("Predict. Understand. Retain.")
page = st.sidebar.radio("Navigate", ["Business Overview", "Customer Insights", "Churn Prediction", "Model Performance"])
st.sidebar.markdown("---")
st.sidebar.metric("Customers in dataset", f"{len(df):,}")
st.sidebar.metric("Observed churn rate", f"{(df.Churn=='Yes').mean():.1%}")

def bar_rate(col, title):
    g = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="Churn rate (%)")
    f = px.bar(g, x=col, y="Churn rate (%)", title=title, color_discrete_sequence=TEAL, text_auto=".1f")
    st.plotly_chart(f, width="stretch")

if page == "Business Overview":
    st.title("ChurnIQ – Customer Churn Prediction for Telecom Retention")
    c = st.columns(4)
    c[0].metric("Customers", f"{len(df):,}"); c[1].metric("Churned", f"{(df.Churn=='Yes').sum():,}")
    c[2].metric("Churn rate", f"{(df.Churn=='Yes').mean():.1%}"); c[3].metric("Avg monthly charge", f"${df.MonthlyCharges.mean():.2f}")
    st.subheader("Business problem")
    st.write("Acquiring a new telecom customer typically costs more than retaining one. Churn erodes revenue and customer lifetime value, "
             "so management needs to identify at-risk customers early and target retention spend where it matters.")
    st.subheader("Objective")
    st.write("Predict whether a customer will churn (Yes/No) with a supervised classification model, estimate churn probability, and recommend a retention action.")
    st.subheader("Dataset")
    st.write(f"IBM Telco Customer Churn dataset (Kaggle): {df.shape[0]:,} rows × {df.shape[1]} columns; target = `Churn`. "
             "`customerID` is dropped; 11 blank `TotalCharges` values (tenure = 0) are imputed in the pipeline.")
    st.subheader("Methodology")
    st.write(f"Data cleaning → one-hot encoding & scaling inside a scikit-learn Pipeline (fitted on training data only) → "
             f"80/20 stratified split ({M['n_train']:,}/{M['n_test']:,}) → compare Logistic Regression, Decision Tree, Random Forest, KNN with 5-fold CV → "
             f"final model: **{M['final']}** (best F1; class weights used to handle imbalance).")

elif page == "Customer Insights":
    st.title("Customer Insights (historical data)")
    seg = st.multiselect("Filter by contract", df.Contract.unique(), default=list(df.Contract.unique()))
    df = df[df.Contract.isin(seg)]
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.pie(df, names="Churn", title="Churn distribution", hole=.5, color_discrete_sequence=TEAL), width="stretch")
        bar_rate("Contract", "Churn rate by contract type")
        bar_rate("PaymentMethod", "Churn rate by payment method")
    with b:
        st.plotly_chart(px.histogram(df, x="tenure", color="Churn", barmode="overlay", title="Tenure vs churn", color_discrete_sequence=TEAL), width="stretch")
        st.plotly_chart(px.box(df, x="Churn", y="MonthlyCharges", title="Monthly charges vs churn", color="Churn", color_discrete_sequence=TEAL), width="stretch")
        bar_rate("InternetService", "Churn rate by internet service")
    bar_rate("TechSupport", "Churn rate by tech support")
    st.subheader("Key observations (computed from the data)")
    r = lambda col, v: (df[df[col] == v].Churn == "Yes").mean() * 100
    if len(df):
        st.info(f"Month-to-month churn: {r('Contract','Month-to-month'):.1f}% | Two-year: {r('Contract','Two year'):.1f}%  \n"
                f"Fiber optic churn: {r('InternetService','Fiber optic'):.1f}% | DSL: {r('InternetService','DSL'):.1f}%  \n"
                f"Electronic check churn: {r('PaymentMethod','Electronic check'):.1f}%  \n"
                f"Customers with no tech support (internet users): {r('TechSupport','No'):.1f}% vs with tech support: {r('TechSupport','Yes'):.1f}%")
    st.caption("These are historical associations, not proof of causation.")

elif page == "Churn Prediction":
    st.title("Predict Customer Churn")
    with st.form("f"):
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (months)", 0, 72, 12)
        monthly = c1.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5)
        contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = c2.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        pay = c2.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
        gender = c3.selectbox("Gender", ["Female", "Male"]); senior = c3.selectbox("Senior citizen", ["No", "Yes"])
        partner = c3.selectbox("Partner", ["No", "Yes"]); dep = c3.selectbox("Dependents", ["No", "Yes"])
        phone = c1.selectbox("Phone service", ["Yes", "No"])
        ml = c1.selectbox("Multiple lines", ["No", "Yes"])
        addon = {}
        if internet != "No":
            for k, lbl in [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"), ("DeviceProtection", "Device protection"),
                           ("TechSupport", "Tech support"), ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")]:
                addon[k] = (c2 if k in ("OnlineSecurity", "OnlineBackup", "DeviceProtection") else c3).selectbox(lbl, ["No", "Yes"], key=k)
        go = st.form_submit_button("Predict churn")
    if go:
        try:
            for k in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                addon.setdefault(k, "No internet service")
            row = dict(gender=gender, SeniorCitizen=int(senior == "Yes"), Partner=partner, Dependents=dep, tenure=tenure,
                       PhoneService=phone, MultipleLines=ml if phone == "Yes" else "No phone service", InternetService=internet, **addon,
                       Contract=contract, PaperlessBilling=paperless, PaymentMethod=pay, MonthlyCharges=monthly,
                       TotalCharges=round(tenure * monthly, 2))
            X = pd.DataFrame([row]); p = model().predict_proba(X)[0, 1]
            risk = "High" if p >= .6 else "Medium" if p >= .35 else "Low"
            c = st.columns(3)
            c[0].metric("Prediction", "Likely to churn" if p >= .5 else "Likely to stay")
            c[1].metric("Churn probability", f"{p:.1%}"); c[2].metric("Risk category", risk)
            st.progress(float(p))
            act = {"High": "Prioritise personal retention outreach; offer a contract-renewal incentive and investigate service issues (requires manager approval).",
                   "Medium": "Offer relevant add-ons (tech support / online security), encourage longer contracts and monitor satisfaction.",
                   "Low": "Maintain the relationship, promote loyalty programmes and explore cross-selling."}[risk]
            (st.error if risk == "High" else st.warning if risk == "Medium" else st.success)("**Recommended action:** " + act)
            st.caption("Risk bands: High ≥ 60%, Medium 35–60%, Low < 35%. Total charges estimated as tenure × monthly charges. Decision support only.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.title("Model Performance")
    rs = M["results"]; t = pd.DataFrame(rs).T.drop(columns="confusion_matrix").astype(float).round(3)
    st.success(f"Selected model: {M['final']}")
    st.dataframe(t, width="stretch")
    st.plotly_chart(px.bar(t.reset_index().melt("index"), x="index", y="value", color="variable", barmode="group", title="Model comparison (test set)"), width="stretch")
    cm = rs[M["final"]]["confusion_matrix"]
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred stay", "Pred churn"], y=["Actual stay", "Actual churn"], color_continuous_scale="Teal", title="Confusion matrix – selected model"), width="stretch")
    fi = pd.DataFrame(M["importance"], columns=["Feature", "Importance"])
    st.plotly_chart(px.bar(fi[::-1], x="Importance", y="Feature", orientation="h", title="Top feature importances (association, not causation)", color_discrete_sequence=TEAL), width="stretch")
    st.write(f"Train/test: {M['n_train']:,}/{M['n_test']:,}; churn share ≈ {M['test_churn']:.1%} (imbalanced → recall and F1 matter more than accuracy). "
             "A missed churner (false negative) costs the customer's future revenue; a false positive costs only an unnecessary retention offer.")
