"""
dashboard_prix_dayahead.py
──────────────────────────
Dashboard Streamlit — Prix Day-Ahead ENTSO-E
  • Sélection de pays via cases à cocher
  • Plage de dates libre
  • Courbes interactives Plotly (zoom, hover, cacher une courbe au clic légende)
  • Cache automatique des données téléchargées

Lancer :
    pip install streamlit entsoe-py plotly pandas pyarrow
    streamlit run dashboard_prix_dayahead.py
"""

import os
import hashlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

# ── Config page ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prix Day-Ahead — ENTSO-E",
    page_icon="⚡",
    layout="wide",
)

# ── Palette (une couleur fixe par pays, cohérente entre rechargements) ─────────
PALETTE = [
    "#4C9BE8", "#E24444", "#2DB87A", "#F5A623", "#A855F7",
    "#EC4899", "#14B8A6", "#F97316", "#84CC16", "#06B6D4",
    "#8B5CF6", "#EF4444", "#10B981", "#F59E0B", "#3B82F6",
]

# ── Pays disponibles ───────────────────────────────────────────────────────────
PAYS = {
    "France":                   "FR",
    "Allemagne-Luxembourg":     "DE_LU",
    "Belgique":                 "BE",
    "Pays-Bas":                 "NL",
    "Espagne":                  "ES",
    "Italie (nord)":            "IT_NORD",
    "Portugal":                 "PT",
    "Suisse":                   "CH",
    "Autriche":                 "AT",
    "Pologne":                  "PL",
    "Suède (SE1)":              "SE_1",
    "Norvège (NO1)":            "NO_1",
    "Danemark (DK1)":           "DK_1",
    "Finlande":                 "FI",
    "Grande-Bretagne":          "GB",
}

CACHE_DIR = ".cache_prix"
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Fetch avec cache fichier ───────────────────────────────────────────────────
def cache_path(code: str, start: str, end: str) -> str:
    key = hashlib.md5(f"{code}{start}{end}".encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{code}_{key}.parquet")


def fetch_prices(code: str, start: date, end: date) -> pd.Series | None:
    """Retourne une Series horaire €/MWh. Met en cache en parquet."""
    path = cache_path(code, str(start), str(end))
    if os.path.exists(path):
        return pd.read_parquet(path).squeeze()

    api_key = st.secrets.get("ENTSOE_TOKEN", os.environ.get("ENTSOE_TOKEN", ""))
    if not api_key:
        st.error("🔑 Token ENTSO-E manquant. "
                 "Ajoute `ENTSOE_TOKEN` dans `.streamlit/secrets.toml` "
                 "ou en variable d'environnement.")
        st.stop()

    try:
        from entsoe import EntsoePandasClient
        from entsoe.exceptions import NoMatchingDataError
        client = EntsoePandasClient(api_key=api_key)
        ts_start = pd.Timestamp(start.isoformat(), tz="Europe/Paris")
        ts_end   = pd.Timestamp(end.isoformat(),   tz="Europe/Paris") + pd.Timedelta(days=1)
        sr = client.query_day_ahead_prices(code, start=ts_start, end=ts_end)
        if sr is not None and len(sr) > 0:
            sr.to_frame().to_parquet(path)
            return sr
        return None
    except NoMatchingDataError:
        return None
    except Exception as e:
        st.warning(f"⚠️ Erreur pour {code} : {e}")
        return None


# ── Agrégation (optionnelle) ───────────────────────────────────────────────────
def resample(sr: pd.Series, freq: str) -> pd.Series:
    if freq == "Horaire":
        return sr
    elif freq == "Journalier":
        return sr.resample("1D").mean()
    elif freq == "Hebdomadaire":
        return sr.resample("1W").mean()
    elif freq == "Mensuel":
        return sr.resample("1ME").mean()
    return sr


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Prix Day-Ahead")
    st.markdown("---")

    # Dates
    st.markdown("### 📅 Période")
    col1, col2 = st.columns(2)
    with col1:
        date_debut = st.date_input(
            "Début",
            value=date.today() - timedelta(days=365),
            min_value=date(2015, 1, 1),
            max_value=date.today(),
        )
    with col2:
        date_fin = st.date_input(
            "Fin",
            value=date.today() - timedelta(days=1),
            min_value=date(2015, 1, 1),
            max_value=date.today(),
        )

    if date_debut >= date_fin:
        st.error("La date de début doit être avant la date de fin.")
        st.stop()

    # Résolution
    st.markdown("### 📊 Résolution")
    resolution = st.selectbox(
        "Agréger par",
        ["Horaire", "Journalier", "Hebdomadaire", "Mensuel"],
        index=1,
    )

    st.markdown("---")
    st.markdown("### 🌍 Pays")

    # Boutons sélection rapide
    col_a, col_b = st.columns(2)
    with col_a:
        tout_cocher = st.button("Tout cocher", use_container_width=True)
    with col_b:
        tout_decocher = st.button("Tout décocher", use_container_width=True)

    # État des checkboxes dans session_state
    if "selection" not in st.session_state:
        st.session_state.selection = {p: (p in ["France", "Allemagne-Luxembourg"]) for p in PAYS}
    if tout_cocher:
        st.session_state.selection = {p: True for p in PAYS}
    if tout_decocher:
        st.session_state.selection = {p: False for p in PAYS}

    # Cases à cocher
    for i, pays in enumerate(PAYS):
        couleur = PALETTE[i % len(PALETTE)]
        checked = st.checkbox(
            pays,
            value=st.session_state.selection.get(pays, False),
            key=f"cb_{pays}",
        )
        st.session_state.selection[pays] = checked

    pays_selectionnes = [p for p, v in st.session_state.selection.items() if v]

    st.markdown("---")
    st.caption("Source : ENTSO-E Transparency Platform\n`query_day_ahead_prices`")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:1.8rem;margin-bottom:0'>⚡ Prix Day-Ahead — ENTSO-E</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:#888;margin-top:4px'>"
    f"{date_debut.strftime('%d/%m/%Y')} → {date_fin.strftime('%d/%m/%Y')} "
    f"· {resolution} · {len(pays_selectionnes)} pays sélectionné(s)</p>",
    unsafe_allow_html=True,
)

if not pays_selectionnes:
    st.info("Sélectionne au moins un pays dans le panneau gauche.")
    st.stop()

# ── Chargement des données ─────────────────────────────────────────────────────
donnees = {}
erreurs = []

progress = st.progress(0, text="Chargement des données…")
for k, pays in enumerate(pays_selectionnes):
    code = PAYS[pays]
    progress.progress((k + 1) / len(pays_selectionnes), text=f"Téléchargement : {pays}…")
    sr = fetch_prices(code, date_debut, date_fin)
    if sr is not None and len(sr) > 0:
        donnees[pays] = resample(sr, resolution)
    else:
        erreurs.append(pays)
progress.empty()

if erreurs:
    st.warning(f"Données indisponibles pour : {', '.join(erreurs)}")

if not donnees:
    st.error("Aucune donnée disponible pour la sélection.")
    st.stop()

# ── Graphique principal ────────────────────────────────────────────────────────
fig = go.Figure()

for pays, sr in donnees.items():
    idx = list(PAYS.keys()).index(pays)
    couleur = PALETTE[idx % len(PALETTE)]
    fig.add_trace(go.Scatter(
        x=sr.index,
        y=sr.values,
        name=pays,
        line=dict(color=couleur, width=1.8),
        mode="lines",
    ))

# Layout minimal — compatible toutes versions Plotly / Python 3.14
fig.update_layout(
    height=520,
    hovermode="x unified",
    margin=dict(l=10, r=10, t=30, b=10),
)
fig.update_xaxes(rangeslider_visible=True)
fig.update_yaxes(title_text="€/MWh")

st.plotly_chart(fig, use_container_width=True)

# ── Statistiques ───────────────────────────────────────────────────────────────
st.markdown("### 📊 Statistiques sur la période")

stats_rows = []
for pays, sr in donnees.items():
    stats_rows.append({
        "Pays": pays,
        "Moyenne (€/MWh)":  round(sr.mean(), 1),
        "Médiane":          round(sr.median(), 1),
        "Min":              round(sr.min(), 1),
        "Max":              round(sr.max(), 1),
        "Heures ≤ 0 €/MWh": int((sr <= 0).sum()),
        "% heures ≤ 0":    f"{(sr <= 0).mean()*100:.1f} %",
    })

df_stats = pd.DataFrame(stats_rows).set_index("Pays")

# Colorier les cellules moyenne
def color_mean(v):
    try:
        f = float(str(v).replace(",", "."))
        if f < 0:    return "background-color:#7f1d1d;color:white"
        elif f < 30: return "background-color:#1a3a2a;color:white"
        elif f > 150:return "background-color:#451a03;color:white"
    except Exception:
        pass
    return ""

st.dataframe(
    df_stats.style.applymap(color_mean, subset=["Moyenne (€/MWh)"]),
    use_container_width=True,
)

# ── Note prix négatifs ─────────────────────────────────────────────────────────
total_neg = sum((sr <= 0).sum() for sr in donnees.values())
if total_neg > 0:
    pays_neg = {p: int((sr <= 0).sum()) for p, sr in donnees.items() if (sr <= 0).any()}
    detail = " · ".join(f"{p} : {n}h" for p, n in sorted(pays_neg.items(), key=lambda x: -x[1]))
    st.info(f"⚠️ **{total_neg} heures à prix nul ou négatif** sur la période — {detail}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Données ENTSO-E Transparency Platform · "
    "Prix Day-Ahead couplage unique (SDAC, ex-PCR) · "
    "Cache local dans `.cache_prix/` — supprimer pour re-télécharger."
)
