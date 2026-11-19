import streamlit as st
import streamlit.components.v1 as components
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import time

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

st.set_page_config(page_title="AI FraudGuard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).parent

CSS = """\n<style>body, [data-testid="stAppViewContainer"] { background: #0b1020; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#111a2d,#0c1221); border-right: 1px solid #25344c; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; }
.fg-title { font-size: 2.15rem; font-weight: 800; letter-spacing: -.04em; color: #f4f7ff; margin-bottom: 0; }
.fg-sub { color: #9eabc4; margin-top: .25rem; }
.metric-card { background: linear-gradient(145deg,#18243b,#111a2c); border: 1px solid #263852; border-radius: 17px; padding: 18px 20px; min-height: 116px; box-shadow: 0 12px 30px rgba(0,0,0,.15); }
.metric-label { color:#9eb0cb; font-size:.86rem; margin-bottom:9px; }
.metric-value { color:#f2f6ff; font-size:1.8rem; font-weight:800; line-height:1.1; }
.metric-foot { color:#7186a5; font-size:.75rem; margin-top:8px; }
.panel { background:#121c2f; border:1px solid #243650; border-radius:18px; padding:18px; }
.tag { display:inline-block; border-radius:999px; padding:5px 10px; background:#153e38; color:#75e3b3; font-size:.75rem; font-weight:700; }
div[data-testid="stButton"] button { border-radius: 12px; border:1px solid #345078; transition:all .2s ease; }
div[data-testid="stButton"] button:hover { border-color:#62a8ff; box-shadow:0 0 18px rgba(64,145,255,.15); transform:translateY(-1px); }
div[data-testid="stFileUploader"] { border-radius:14px; }
hr { border-color:#23344e !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def make_demo_data(n=2600, seed=27):
    rng = np.random.default_rng(seed)
    y = rng.choice([0, 1], size=n, p=[0.965, 0.035])
    features = {}
    for i in range(1, 29):
        features[f"V{i}"] = rng.normal(0, 1, n)
    # Add modest patterns to make the illustrative demo learnable
    features["V4"] += y * 1.4
    features["V11"] += y * 1.8
    features["V14"] -= y * 2.2
    features["V17"] -= y * 1.7
    features["Time"] = rng.uniform(0, 172800, n)
    features["Amount"] = np.clip(rng.lognormal(3.1, 1.1, n), 0.01, 4500)
    df = pd.DataFrame(features)
    df["Class"] = y
    return df


@st.cache_data(show_spinner=False)
def train_models(df):
    clean = df.dropna().drop_duplicates().copy()
    if "Class" not in clean or clean["Class"].nunique() < 2:
        raise ValueError("Dataset needs a Class column containing both 0 and 1.")
    X = clean.drop(columns="Class")
    y = clean["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    logit = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))])
    logit.fit(X_train, y_train)
    fitted = {"Logistic Regression": logit}
    if XGB_AVAILABLE:
        xgb = XGBClassifier(n_estimators=160, max_depth=4, learning_rate=.08, subsample=.8,
                          colsample_bytree=.8, reg_lambda=2, eval_metric="logloss", random_state=42,
                          n_jobs=2, verbosity=0)
        xgb.fit(X_train, y_train)
        fitted["XGBoost"] = xgb
    metrics = {}
    for name, model in fitted.items():
        pred = model.predict(X_test)
        metrics[name] = {"Accuracy": accuracy_score(y_test, pred),
                         "Precision": precision_score(y_test, pred, zero_division=0),
                         "Recall": recall_score(y_test, pred, zero_division=0),
                         "F1-score": f1_score(y_test, pred, zero_division=0),
                         "Confusion matrix": confusion_matrix(y_test, pred),
                         "y_test": y_test, "pred": pred}
    return clean, X, fitted, metrics


def metric_card(label, value, foot="Dataset overview"):
    st.markdown(f'''<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>''', unsafe_allow_html=True)


def header(title, subtitle):
    # Render an actual HTML template inside Streamlit for each page header.
    template_map = {
        "Intelligence Fraud Detection Dashboard": "dashboard.html",
        "AI Transaction Scanner": "fraud_detection.html",
        "Transaction Explorer": "transactions.html",
        "Model Performance Lab": "analytics.html",
        "Fraud Analytics": "analytics.html",
        "Dataset Center": "dataset_center.html",
    }
    template_name = template_map.get(title, "dashboard.html")
    env = Environment(loader=FileSystemLoader(str(ROOT / "templates")), autoescape=True)
    template = env.get_template(template_name)
    js_path = ROOT / "static" / "js" / "app.js"
    js_script = js_path.read_text(encoding="utf-8") if js_path.exists() else ""
    html = template.render(title=title, subtitle=subtitle, js_script=js_script)
    components.html(html, height=150, scrolling=False)



def risk_score(model, row, features):
    try:
        p = model.predict_proba(row[features])[0][1]
        return float(p)
    except Exception:
        return float(model.predict(row[features])[0])



# Data source selector/upload
with st.sidebar:
    st.markdown("## 🛡️ AI FraudGuard")
    st.caption("Transaction Intelligence Platform")
    st.divider()
    uploaded = st.file_uploader("Upload transaction CSV", type=["csv"], help="Required columns: Time, Amount, V1–V28, Class (for model training).")
    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(raw_df):,} rows")
            st.session_state["uploaded_data"] = raw_df
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    data_mode = "Uploaded dataset" if "uploaded_data" in st.session_state else "Demo dataset"
    st.markdown(f"**Data source:** {data_mode}")
    st.divider()
    page = st.radio("Navigation", ["Dashboard", "AI Fraud Detector", "Transaction Explorer", "Model Performance", "Fraud Analytics", "Dataset Center"], label_visibility="collapsed")
    st.caption("Demo transactions are synthetic and for educational use.")


base = st.session_state.get("uploaded_data", make_demo_data())
if "Class" not in base.columns:
    st.error("Uploaded dataset is missing the required target column: Class.")
    st.stop()

try:
    data, feature_df, models, results = train_models(base)
except Exception as err:
    st.error(f"Model training could not start: {err}")
    st.info("Check that your CSV includes the 0/1 Class target column and numeric feature columns.")
    st.stop()

fraud_n = int((data["Class"] == 1).sum())
legit_n = int((data["Class"] == 0).sum())
fraud_pct = fraud_n / max(len(data), 1) * 100
selected_model = "XGBoost" if "XGBoost" in models else "Logistic Regression"

# Sidebar model selection
with st.sidebar:
    selected_model = st.selectbox("Prediction model", list(models.keys()), index=list(models.keys()).index(selected_model))
    st.markdown("<span class='tag'>● Models trained</span>", unsafe_allow_html=True)

if page == "Dashboard":
    header("Intelligence Fraud Detection Dashboard", "A quick view of transaction patterns, risk signals and model readiness.")
    st.write("")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Total Transactions", f"{len(data):,}")
    with c2: metric_card("Fraud Cases", f"{fraud_n:,}", "Flagged in loaded data")
    with c3: metric_card("Legitimate", f"{legit_n:,}", "Class 0 transactions")
    with c4: metric_card("Fraud Ratio", f"{fraud_pct:.3f}%", "Class 1 share")
    st.write("")
    left,right = st.columns([1.1,1])
    with left:
        st.markdown("### Transaction class distribution")
        donut = pd.DataFrame({"Class":["Legitimate","Fraud"], "Count":[legit_n,fraud_n]})
        fig = px.pie(donut, names="Class", values="Count", hole=.65, color="Class", color_discrete_map={"Legitimate":"#4aa3ff","Fraud":"#ff6685"})
        fig.update_layout(template="plotly_dark", height=330, margin=dict(t=20,b=20,l=20,r=20), legend_title="", paper_bgcolor="rgba(0,0,0,0)", annotations=[dict(text=f"{fraud_pct:.2f}%<br>fraud", x=.5,y=.5,showarrow=False,font_size=16)])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Transaction amount distribution")
        fig2 = px.histogram(data, x="Amount", color="Class", nbins=40, opacity=.78,
                            color_discrete_map={0:"#4aa3ff",1:"#ff6685"}, labels={"Class":"Class"})
        fig2.update_layout(template="plotly_dark",height=330,margin=dict(t=20,b=20,l=20,r=20),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### Model quick view")
    score_rows = [{"Model":name, "Accuracy":m["Accuracy"], "Precision":m["Precision"], "Recall":m["Recall"], "F1-score":m["F1-score"]} for name,m in results.items()]
    st.dataframe(pd.DataFrame(score_rows).set_index("Model").style.format("{:.3%}"), use_container_width=True)
    st.caption("Metrics are evaluated on a held-out stratified split. Demo mode uses synthetic data.")

elif page == "AI Fraud Detector":
    header("AI Transaction Scanner", "Run a model prediction on a transaction row from your dataset or edit feature values.")
    st.warning("This model is an educational prototype, not a banking authorization or production fraud decision system.")
    features = list(feature_df.columns)
    if not features:
        st.error("No numeric features are available for prediction.")
        st.stop()
    with st.form("scan_form"):
        st.markdown("#### Transaction input")
        mode = st.radio("Input method", ["Use transaction from dataset", "Manual feature values"], horizontal=True)
        if mode == "Use transaction from dataset":
            idx = st.number_input("Transaction row index", min_value=0, max_value=max(0,len(data)-1), value=0, step=1)
            row = feature_df.iloc[[int(idx)]].copy()
            st.write("Selected transaction preview")
            st.dataframe(row, use_container_width=True)
        else:
            row_dict = {}
            defaults = feature_df.median(numeric_only=True)
            cols = st.columns(3)
            for i, feat in enumerate(features):
                with cols[i%3]:
                    row_dict[feat] = st.number_input(feat, value=float(defaults.get(feat,0)), format="%.5f", key=f"manual_{feat}")
            row = pd.DataFrame([row_dict], columns=features)
        submit = st.form_submit_button("🔎 Analyze transaction", use_container_width=True, type="primary")
    if submit:
        with st.spinner("AI model is scanning transaction features…"):
            time.sleep(.6)
            model = models[selected_model]
            prediction = int(model.predict(row[features])[0])
            probability = risk_score(model,row,features)
        st.divider()
        if prediction == 1:
            st.error("🚨 FRAUD SUSPECTED — Flag for review")
            st.markdown("This transaction was classified as suspicious by the selected model. Verify the details before any real-world action.")
        else:
            st.success("✅ LEGITIMATE CLASSIFICATION")
            st.markdown("This transaction was classified as legitimate by the selected model. This is not a guarantee of safety.")
        pcol1,pcol2 = st.columns([1,2])
        with pcol1:
            st.metric("Model fraud score", f"{probability:.1%}")
            st.caption("Score is not necessarily a calibrated probability.")
        with pcol2:
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=probability*100, number={"suffix":"%"}, title={"text":"Fraud score"}, gauge={"axis":{"range":[0,100]},"bar":{"color":"#ff6384"},"steps":[{"range":[0,30],"color":"#153c34"},{"range":[30,70],"color":"#433923"},{"range":[70,100],"color":"#482638"}]}))
            gauge.update_layout(template="plotly_dark",height=250,margin=dict(t=40,b=10,l=15,r=15),paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(gauge,use_container_width=True)

elif page == "Transaction Explorer":
    header("Transaction Explorer", "Filter the dataset, inspect transactions and export the current selection.")
    filter_choice = st.radio("Show", ["All", "Fraud only", "Legitimate only"], horizontal=True)
    amount_max = float(data["Amount"].max()) if "Amount" in data else 0.0
    amount_min = float(data["Amount"].min()) if "Amount" in data else 0.0
    if "Amount" in data and amount_max > amount_min:
        amount_range = st.slider("Amount range", min_value=amount_min, max_value=amount_max, value=(amount_min,amount_max))
    else:
        amount_range=(amount_min,amount_max)
    view = data.copy()
    if filter_choice == "Fraud only": view = view[view.Class == 1]
    if filter_choice == "Legitimate only": view = view[view.Class == 0]
    if "Amount" in view: view=view[view.Amount.between(amount_range[0],amount_range[1])]
    st.caption(f"Showing {len(view):,} transactions")
    st.dataframe(view, use_container_width=True, height=430)
    st.download_button("⬇️ Download filtered CSV", view.to_csv(index=False).encode("utf-8"), "fraudguard_filtered.csv", "text/csv")

elif page == "Model Performance":
    header("Model Performance Lab", "Compare the trained models using metrics that matter for imbalanced fraud data.")
    st.info("Recall helps show how many fraud cases were found; precision describes how many flagged cases were fraud. Accuracy alone can be misleading for this imbalanced dataset.")
    perf = pd.DataFrame([{k:m[k] for k in ["Accuracy","Precision","Recall","F1-score"]} for name,m in results.items()], index=list(results.keys()))
    st.markdown("### Metrics comparison")
    st.dataframe(perf.style.format("{:.3%}"), use_container_width=True)
    metric_to_plot = st.selectbox("Chart metric", ["Accuracy","Precision","Recall","F1-score"])
    fig = px.bar(perf.reset_index(names="Model"), x="Model", y=metric_to_plot, color="Model", text_auto=".2%",
                 color_discrete_sequence=["#4aa3ff","#bd83ff"])
    fig.update_layout(template="plotly_dark",yaxis_tickformat=".0%",yaxis_range=[0,1],height=380,paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    for name,m in results.items():
        with st.expander(f"{name} — Confusion matrix"):
            cm=m["Confusion matrix"]
            cmfig=px.imshow(cm,text_auto=True,labels=dict(x="Predicted",y="Actual",color="Count"),x=["Legitimate","Fraud"],y=["Legitimate","Fraud"],color_continuous_scale="Blues")
            cmfig.update_layout(template="plotly_dark",height=350,paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(cmfig,use_container_width=True)

elif page == "Fraud Analytics":
    header("Fraud Analytics", "Explore how fraudulent and legitimate transactions differ across features.")
    if "Amount" in data:
        st.markdown("### Amount summary")
        st.dataframe(data.groupby("Class")["Amount"].agg(["count","mean","median","max"]).rename(index={0:"Legitimate",1:"Fraud"}),use_container_width=True)
        fig=px.box(data.sample(min(len(data),5000),random_state=10),x="Class",y="Amount",color="Class",points=False,color_discrete_map={0:"#4aa3ff",1:"#ff6685"})
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=400)
        st.plotly_chart(fig,use_container_width=True)
    if "Time" in data:
        st.markdown("### Transactions across dataset time")
        bins=np.linspace(data.Time.min(),data.Time.max(),25)
        time_df=data.assign(TimeWindow=pd.cut(data.Time,bins=bins,include_lowest=True)).groupby(["TimeWindow","Class"],observed=True).size().reset_index(name="Transactions")
        time_df["TimeWindow"]=time_df.TimeWindow.astype(str)
        fig=px.line(time_df,x="TimeWindow",y="Transactions",color="Class",markers=True,color_discrete_map={0:"#4aa3ff",1:"#ff6685"})
        fig.update_layout(template="plotly_dark",height=390,paper_bgcolor="rgba(0,0,0,0)",xaxis_tickangle=-45)
        st.plotly_chart(fig,use_container_width=True)

elif page == "Dataset Center":
    header("Dataset Center", "Dataset checks, class balance, preview and export.")
    c1,c2,c3=st.columns(3)
    with c1: metric_card("Rows",f"{len(data):,}")
    with c2: metric_card("Columns",f"{data.shape[1]}")
    with c3: metric_card("Missing values",f"{int(data.isna().sum().sum()):,}")
    st.markdown("### Class balance")
    st.dataframe(data["Class"].value_counts().rename(index={0:"Legitimate (0)",1:"Fraud (1)"}).to_frame("Rows"),use_container_width=True)
    st.markdown("### Dataset preview")
    st.dataframe(data.head(50),use_container_width=True)
    st.download_button("⬇️ Download loaded dataset", data.to_csv(index=False).encode("utf-8"), "fraudguard_dataset.csv", "text/csv")
    st.caption("For model training, the CSV must include a numeric Class column with 0 = legitimate and 1 = fraud, plus numeric transaction features.")

st.sidebar.markdown("---")
st.sidebar.caption("AI FraudGuard · Streamlit prototype")
