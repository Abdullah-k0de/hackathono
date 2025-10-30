# app.py — InjuryShield: ESPN Analytics UI (Black-Yellow-White)

import os, re, base64, numpy as np, pandas as pd, streamlit as st
from datetime import datetime
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE + THEME (Black-Yellow-White)
# =========================
st.set_page_config(page_title="InjuryShield ESPN Analytics", page_icon="⚽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
:root {
  --bg:#0b0b0b; --panel:#151515; --accent:#FFD700; --text:#FFFFFF; --muted:#BDBDBD;
}
html, body, .stApp { background-color: var(--bg); color: var(--text); font-family: Inter, sans-serif; }
h1,h2,h3,h4 { color: var(--accent); letter-spacing: 0.5px; }
.header {
  padding: 14px 20px;
  border-radius: 10px;
  background: #111111;
  border: 1px solid #FFD70055;
  box-shadow: 0 0 20px rgba(255,215,0,0.15);
}
.divider { height:1px; background:linear-gradient(90deg, transparent, var(--accent), transparent); margin: 16px 0; }
.kpi {
  background: var(--panel);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 10px;
  padding: 12px;
  text-align: center;
}
.kpi .label { color: var(--muted); font-size: 13px; }
.kpi .value { font-size: 22px; font-weight: 700; color: var(--accent); }
.coachcall {
  background: linear-gradient(90deg, #FFD70033, #FFD70011);
  border: 1px solid #FFD70088;
  border-radius: 10px;
  padding: 16px;
  text-align: center;
  color: var(--text);
  font-weight: 700;
  font-size: 18px;
  margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'><h1> InjuryShield — ESPN Analytics Dashboard</h1></div>", unsafe_allow_html=True)

# =========================
# DATA LOADING + CLEANING
# =========================
@st.cache_data
def load_raw_csv():
    if os.path.exists("player_injuries_impact.csv"):
        return pd.read_csv("player_injuries_impact.csv")
    st.error("CSV not found. Place player_injuries_impact.csv next to app.py.")
    st.stop()

@st.cache_data
def prepare_data(df_in: pd.DataFrame):
    df = df_in.copy()

    # normalize columns
    df.columns = [re.sub(r'[^0-9a-zA-Z_]+', '', c.strip().replace(' ', '_')) for c in df.columns]

    # common renames
    colmap = {
        "Name": "name",
        "Team_Name": "team",
        "Team": "team",
        "Position": "position",
        "Age": "age",
        "Season": "season",
        "FIFA_rating": "fifa_rating",
        "Injury": "injury",
        "Date_of_Injury": "Date_of_Injury",
        "Date_of_return": "Date_of_return",
    }
    for k, v in colmap.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # dates
    for c in [c for c in df.columns if "Date" in c or "date" in c]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # detect rating columns dynamically
    rating_cols = [c for c in df.columns if "rating" in c.lower() or "score" in c.lower()]

    before_cols = [c for c in rating_cols if "before" in c.lower()]
    after_cols  = [c for c in rating_cols if "after" in c.lower()]

    # force to numeric to avoid "Could not convert ..."
    for c in before_cols + after_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if before_cols:
        df["avg_rating_before"] = df[before_cols].mean(axis=1, skipna=True)
    else:
        df["avg_rating_before"] = np.nan

    if after_cols:
        df["avg_rating_after"] = df[after_cols].mean(axis=1, skipna=True)
    else:
        df["avg_rating_after"] = np.nan

    # recovery days (guard missing cols)
    if "Date_of_return" in df.columns and "Date_of_Injury" in df.columns:
        df["recovery_days"] = (df["Date_of_return"] - df["Date_of_Injury"]).dt.days
    else:
        df["recovery_days"] = np.nan

    df["rating_delta_after_minus_before"] = df["avg_rating_after"] - df["avg_rating_before"]

    # safe severe flag
    def _sev(x):
        if pd.isna(x):
            return 0
        return 1 if x > 28 else 0
    df["severe_recovery"] = df["recovery_days"].apply(_sev)

    if "name" not in df.columns:
        df["name"] = "unknown_player"

    return df

raw = load_raw_csv()
df = prepare_data(raw)

# =========================
# MODEL TRAINING
# =========================
@st.cache_data
def train_and_predict(df: pd.DataFrame):
    cat = [c for c in ["team", "position", "season", "injury"] if c in df.columns]
    num = [c for c in ["age", "fifa_rating", "avg_rating_before"] if c in df.columns]
    feats = cat + num

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num),
    ])

    results = []

    # helper to pick n_splits
    def _gkf_splits(groups_ser, max_splits=5):
        n_unique = groups_ser.nunique()
        return max(2, min(max_splits, n_unique))

    # 1) Severe risk (classification)
    cls_df = df.dropna(subset=feats + ["severe_recovery"]).copy()
    if not cls_df.empty and cls_df["severe_recovery"].nunique() > 1:
        Xc = cls_df[feats]
        yc = cls_df["severe_recovery"]
        groups = cls_df["name"]
        n_splits = _gkf_splits(groups)
        model = Pipeline([
            ("pre", pre),
            ("gbc", GradientBoostingClassifier(random_state=42)),
        ])
        yprob = cross_val_predict(
            model, Xc, yc,
            groups=groups,
            cv=GroupKFold(n_splits=n_splits),
            method="predict_proba"
        )[:, 1]
        cls_df["y_pred_proba_severe"] = yprob
        results.append({
            "task": "Severe",
            "AUC": roc_auc_score(yc, yprob),
            "AP": average_precision_score(yc, yprob),
        })
    else:
        cls_df["y_pred_proba_severe"] = np.nan

    # 2) Recovery days (regression)
    rec_df = df.dropna(subset=feats + ["recovery_days"]).copy()
    if not rec_df.empty:
        Xr = rec_df[feats]
        yr = rec_df["recovery_days"]
        groups = rec_df["name"]
        n_splits = _gkf_splits(groups)
        modelr = Pipeline([
            ("pre", pre),
            ("gbr", GradientBoostingRegressor(random_state=42)),
        ])
        yhat = cross_val_predict(
            modelr, Xr, yr,
            groups=groups,
            cv=GroupKFold(n_splits=n_splits),
        )
        rec_df["y_pred_recovery_days"] = yhat
        results.append({
            "task": "Recovery",
            "MAE": mean_absolute_error(yr, yhat),
            "R2": r2_score(yr, yhat),
        })
    else:
        rec_df["y_pred_recovery_days"] = np.nan

    # 3) Post-injury rating (regression)
    perf_df = df.dropna(subset=feats + ["avg_rating_after"]).copy()
    if not perf_df.empty:
        Xp = perf_df[feats]
        yp = perf_df["avg_rating_after"]
        groups = perf_df["name"]
        n_splits = _gkf_splits(groups)
        modelp = Pipeline([
            ("pre", pre),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ])
        yhatp = cross_val_predict(
            modelp, Xp, yp,
            groups=groups,
            cv=GroupKFold(n_splits=n_splits),
        )
        perf_df["y_pred_avg_rating_after"] = yhatp
    else:
        perf_df["y_pred_avg_rating_after"] = np.nan

    # --- Merge everything safely on name ---
    dash = df[["name"]].drop_duplicates().copy()

    if "y_pred_proba_severe" in cls_df.columns:
        dash = dash.merge(cls_df[["name", "y_pred_proba_severe"]], on="name", how="left")

    if "y_pred_recovery_days" in rec_df.columns:
        dash = dash.merge(rec_df[["name", "y_pred_recovery_days"]], on="name", how="left")

    if "y_pred_avg_rating_after" in perf_df.columns:
        dash = dash.merge(perf_df[["name", "y_pred_avg_rating_after"]], on="name", how="left")

    if "avg_rating_before" in df.columns:
        dash = dash.merge(df[["name", "avg_rating_before"]].drop_duplicates(), on="name", how="left")
    else:
        dash["avg_rating_before"] = np.nan

    # final derived
    dash["pred_rating_drop"] = dash["avg_rating_before"].fillna(0) - dash["y_pred_avg_rating_after"].fillna(0)

    def _rec(r):
        risk = r.get("y_pred_proba_severe", 0) or 0
        rec_days = r.get("y_pred_recovery_days", 0) or 0
        drop = r.get("pred_rating_drop", 0) or 0
        if risk >= 0.66:
            return "🚨 DO NOT START"
        elif rec_days >= 21:
            return "⚠️ Minutes Cap ≤30"
        elif drop >= 0.5:
            return "♻️ Late-Game Sub"
        else:
            return "✅ Start Regular"

    dash["Recommendation"] = dash.apply(_rec, axis=1)

    # ✅ bring injury info + start date forward
    cols_to_add = []
    if "injury" in df.columns:
        cols_to_add.append("injury")
    if "Date_of_Injury" in df.columns:
        cols_to_add.append("Date_of_Injury")

    if cols_to_add:
        extra_info = df[["name"] + cols_to_add].drop_duplicates(subset="name")
        dash = dash.merge(extra_info, on="name", how="left")

    return pd.DataFrame(results), dash


results_df, dash_df = train_and_predict(df)

# =========================
# 🧠 Analyst Console — Search + Visuals
# =========================
st.subheader("Analyst Console — Player Insights")

search_query = st.text_input("Search Player", "", key="search").strip().lower()
filtered = [p for p in dash_df["name"].unique() if search_query in p.lower()]
if not filtered:
    st.warning("No players found.")
    st.stop()

player = st.selectbox("Select Player", filtered, index=0)
row = dash_df[dash_df["name"] == player].iloc[0]

col1, col2, col3 = st.columns(3)

# Risk radar
with col1:
    st.markdown("##### Risk Radar")
    risk = row.get("y_pred_proba_severe", 0) or 0
    rec = row.get("y_pred_recovery_days", 0) or 0
    drop = row.get("pred_rating_drop", 0) or 0
    vals = [risk, min(rec / 45, 1), min((drop + 1) / 3, 1)]
    axes = ["Injury Risk", "Recovery Duration", "Performance Drop"]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=axes + [axes[0]],
        fill='toself',
        line=dict(width=2, color="#FFD700")
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color="#534600"),
                gridcolor="rgba(255,255,255,0.15)",
                linecolor="rgba(255,255,255,0.3)"
            ),
            angularaxis=dict(tickfont=dict(color="#FFD700"))
        ),
        font=dict(color="#FFD700"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# Performance chart (neon style but yellow theme)
with col2:
    st.markdown(f"##### Performance Trajectory — {player}")

    before = row.get("avg_rating_before")
    after  = row.get("y_pred_avg_rating_after")

    if pd.notna(before) and pd.notna(after):
        fig_curve = go.Figure()

        # Smooth curve with yellow glow
        fig_curve.add_trace(go.Scatter(
            x=[0, 1],
            y=[before, after],
            mode='lines+markers+text',
            line=dict(color='#FFD700', width=5, shape='spline'),
            marker=dict(size=16, color=['#FFFFFF', '#FFD700'],
                        line=dict(width=3, color='#000000')),
            text=['Before', 'After'],
            textposition=['top center', 'bottom center'],
            name=f"{player} Performance"
        ))

        fig_curve.add_trace(go.Bar(
            x=['Before', 'After'],
            y=[before, after],
            marker=dict(
                color=['rgba(255,255,255,0.2)', 'rgba(255,215,0,0.3)'],
                line=dict(color=['#FFFFFF', '#FFD700'], width=2)
            ),
            opacity=0.6,
            name='Performance Levels'
        ))

        delta = after - before
        delta_text = f"{delta:+.2f}" if pd.notna(delta) else "N/A"

        fig_curve.update_layout(
            yaxis_title="Rating",
            xaxis=dict(tickvals=[0, 1], ticktext=["Before", "After"], showgrid=False),
            title=dict(
                text=f"<b>Performance Change: {delta_text}</b>",
                x=0.5, xanchor='center', font=dict(size=14, color='#FFD700')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            margin=dict(l=10, r=10, t=60, b=10),
            showlegend=False
        )

        # underline line for accent
        fig_curve.add_shape(
            type="line", x0=0, x1=1,
            y0=min(before, after) - 1, y1=min(before, after) - 1,
            line=dict(color="#FFD700", width=3, dash="dot")
        )

        st.plotly_chart(fig_curve, use_container_width=True)
    else:
        st.info("No performance data available for this player.")


# Recovery chart
with col3:
    st.markdown("##### Recovery Spectrum")
    rec_series = dash_df["y_pred_recovery_days"].dropna()
    if not rec_series.empty:
        fig = px.histogram(
            rec_series,
            nbins=25,
            color_discrete_sequence=["#FFD700"],
            labels={"value": "Recovery Days (predicted)", "count": "Frequency"}
        )

        player_days = row.get("y_pred_recovery_days", 0)
        if pd.notna(player_days):
            fig.add_vline(
                x=player_days,
                line=dict(color="white", width=3),
                annotation_text=f"{player}: {player_days:.0f} days",
                annotation_position="top right",
                annotation_font=dict(color="#FFFFFF", size=12)
            )

        # Legend + layout
        fig.update_traces(name="League Distribution", showlegend=True)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                line=dict(color="white", width=3),
                                name=f"{player}'s predicted recovery"))
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                        font=dict(color="#FFD700")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#FFD700")
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No recovery predictions available yet.")

# =========================
# Coach Call
# =========================
st.markdown(f"<div class='coachcall'>Coach Call: {row['Recommendation']}</div>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =========================
# ESPN Broadcast Strip — with search + injury info
# =========================
st.subheader("ESPN Broadcast Strip — Top Risk Highlights")

# User controls
top_n = st.slider("How many players to show:", 6, 30, 10, 2, key="strip_slider")
search_query_strip = st.text_input("Search Player (ESPN strip)", "", key="strip_search").strip().lower()

# Filter
filtered_strip = dash_df[dash_df["name"].str.lower().str.contains(search_query_strip)] if search_query_strip else dash_df
strip = filtered_strip.head(top_n)

cols = st.columns(5)

for i, (_, r) in enumerate(strip.iterrows()):
    c = cols[i % 5]

    # handle injury and start date safely
    injury_text = (
        str(r.get("injury", "")).strip() if pd.notna(r.get("injury")) and str(r.get("injury")).strip() != "" else "Unknown"
    )
    injury_date = (
        pd.to_datetime(r.get("Date_of_Injury")).strftime("%b %d, %Y")
        if pd.notna(r.get("Date_of_Injury")) else "N/A"
    )

    # build card
    c.markdown(
        f"""
        <div style='background:#151515;border:1px solid #FFD70044;
                    border-radius:8px;padding:10px;margin:4px;'>
            <b>{r['name']}</b><br>
            <span style='color:#FFD700;'>Injury:</span> {injury_text}<br>
            <span style='color:#FFD700;'>Start Date:</span> {injury_date}<br>
            <span style='color:#FFD700;'>Risk:</span> {r.get('y_pred_proba_severe', np.nan):.2f}<br>
            <span style='color:#FFD700;'>Recovery:</span> {r.get('y_pred_recovery_days', np.nan) if pd.notna(r.get('y_pred_recovery_days', np.nan)) else 0:.0f} days<br>
            <span style='color:#FFD700;'>Δ Rating:</span> {r.get('pred_rating_drop', np.nan) if pd.notna(r.get('pred_rating_drop', np.nan)) else 0:+.2f}<br>
            <div style='margin-top:6px;color:#FFD700;font-weight:700;'>{r['Recommendation']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =========================
# KPI Summary
# =========================
st.subheader("Key Metrics Summary")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"<div class='kpi'><div class='label'>Players (unique)</div><div class='value'>{df['name'].nunique()}</div></div>",
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f"<div class='kpi'><div class='label'>Rows</div><div class='value'>{df.shape[0]}</div></div>",
        unsafe_allow_html=True
    )
with c3:
    if not results_df.empty and "MAE" in results_df.columns:
        mae = results_df.query("task=='Recovery'")["MAE"].iloc[0]
        mae = f"{mae:.2f}"
    else:
        mae = "--"
    st.markdown(
        f"<div class='kpi'><div class='label'>Recovery Average (days)</div><div class='value'>{mae}</div></div>",
        unsafe_allow_html=True
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# =========================
# 📋 Full Dashboard Table + ⬇️ Exports
# =========================
st.subheader("📋 Full Dashboard Table")
st.dataframe(
    dash_df.rename(columns={
        "y_pred_proba_severe": "Injury Risk",
        "y_pred_recovery_days": "Recovery Days (pred)",
        "y_pred_avg_rating_after": "Post-Injury Rating (pred)",
        "pred_rating_drop": "Δ Rating (pred)",
    }),
    use_container_width=True,
)

def dl_link(df, filename):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'

st.markdown("### Exports")
colA, colB = st.columns(2)
with colA:
    st.markdown(dl_link(dash_df, "dashboard.csv"), unsafe_allow_html=True)
with colB:
    st.markdown(dl_link(results_df, "model_metrics.csv"), unsafe_allow_html=True)

st.caption("© InjuryShield — ESPN Analytics · {}".format(datetime.now().strftime("%Y-%m-%d")))
