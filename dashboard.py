import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Monitoring - Credit Scoring", layout="wide")
st.title("Monitoring - Credit Scoring API")

# Charger les logs
log_path = Path("logs/predictions.jsonl")

if not log_path.exists() or log_path.stat().st_size == 0:
    st.warning("Aucun log de production trouvé.")
    st.stop()

logs = []
with open(log_path) as f:
    for line in f:
        logs.append(json.loads(line))

df = pd.DataFrame(logs)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --- Métriques clés ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Requêtes totales", len(df))
col2.metric("Taux de refus", f"{(df['decision'] == 'REFUSE').mean() * 100:.1f}%")
col3.metric("Temps inférence moyen", f"{df['inference_time_ms'].mean():.1f} ms")
col4.metric("Temps inférence P95", f"{df['inference_time_ms'].quantile(0.95):.1f} ms")

st.divider()

# --- Graphiques ---
left, right = st.columns(2)

with left:
    st.subheader("Distribution des scores prédits")
    hist_data = df["probability"].value_counts(bins=20).sort_index()
    hist_data.index = [f"{i.mid:.2f}" for i in hist_data.index]
    st.bar_chart(hist_data)

with right:
    st.subheader("Répartition des décisions")
    decision_counts = df["decision"].value_counts()
    st.bar_chart(decision_counts)

left2, right2 = st.columns(2)

with left2:
    st.subheader("Temps d'inférence (ms)")
    st.line_chart(df.set_index("timestamp")["inference_time_ms"])

with right2:
    st.subheader("Score moyen par heure")
    df["hour"] = df["timestamp"].dt.hour
    hourly = df.groupby("hour")["probability"].mean()
    st.line_chart(hourly)

# --- Dernières prédictions ---
st.divider()
st.subheader("Dernières prédictions")
st.dataframe(
    df.sort_values("timestamp", ascending=False).head(20),
    use_container_width=True,
)
