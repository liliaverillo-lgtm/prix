"""
dashboard_prix_dayahead.py
───────────────────────────
Dashboard Streamlit — Prix Day-Ahead ENTSO-E

Fonctionnement :
  • Une base de données locale (master_db.parquet) accumule toutes les données.
  • Deux façons d'alimenter la base :
      1. Upload d'un fichier Parquet (ex. produit par le script de téléchargement)
      2. Téléchargement via l'API ENTSO-E pour les pays/périodes manquants
  • Le dashboard lit toujours depuis la base locale → rapide après le premier chargement.

Lancer :
    pip install streamlit entsoe-py plotly pandas pyarrow requests
    streamlit run dashboard_prix_dayahead.py
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config page ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prix Day-Ahead — ENTSO-E",
    page_icon="⚡",
    layout="wide",
)

# ── Constantes ─────────────────────────────────────────────────────────────────
CACHE_DIR  = ".cache_prix"
MASTER_DB  = os.path.join(CACHE_DIR, "master_db.parquet")

os.makedirs(CACHE_DIR, exist_ok=True)

# Pays disponibles : code ENTSO-E → nom affiché
PAYS = {
    "FR":      "France",
    "DE_LU":   "Allemagne",
    "BE":      "Belgique",
    "NL":      "Pays-Bas",
    "ES":      "Espagne",
    "IT_NORD": "Italie (nord)",
    "PT":      "Portugal",
    "CH":      "Suisse",
    "AT":      "Autriche",
    "PL":      "Pologne",
    "GB":      "Grande-Bretagne",
    "SE_1":    "Suède (SE1)",
    "NO_1":    "Norvège (NO1)",
    "DK_1":    "Danemark (DK1)",
    "FI":      "Finlande",
}
CODE_TO_NOM = PAYS                        # code  → nom
NOM_TO_CODE = {v: k for k, v in PAYS.items()}  # nom → code

PALETTE = [
    "#4C9BE8", "#E24444", "#2DB87A", "#F5A623", "#A855F7",
    "#EC4899", "#14B8A6", "#F97316", "#84CC16", "#06B6D4",
    "#8B5CF6", "#EF4444", "#10B981", "#F59E0B", "#3B82F6",
]


# ══════════════════════════════════════════════════════════════════════════════
#  BASE DE DONNÉES LOCALE
# ══════════════════════════════════════════════════════════════════════════════

def load_db() -> pd.DataFrame:
    """Charge le master_db.parquet. Retourne un DataFrame vide si absent."""
    if os.path.exists(MASTER_DB):
        df = pd.read_parquet(MASTER_DB)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    return pd.DataFrame()


def save_db(df: pd.DataFrame):
    """Sauvegarde le DataFrame dans master_db.parquet."""
    df.to_parquet(MASTER_DB)


def merge_into_db(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne new_df dans la base locale.
    - Les nouvelles dates/pays sont ajoutés.
    - Les données existantes sont mises à jour si new_df a des valeurs.
    Retourne le DataFrame fusionné.
    """
    # S'assurer que l'index est bien en UTC
    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")
    else:
        new_df.index = new_df.index.tz_convert("UTC")

    existing = load_db()
    if existing.empty:
        merged = new_df.copy()
    else:
        # combine_first : new_df prioritaire, existing comble les trous
        merged = new_df.combine_first(existing)

    save_db(merged)
    return merged


def db_info(df: pd.DataFrame) -> str:
    """Résumé lisible de la base."""
    if df.empty:
        return "Base vide"
    start = df.index.min().strftime("%d/%m/%Y")
    end   = df.index.max().strftime("%d/%m/%Y")
    cols  = df.shape[1]
    rows  = df.shape[0]
    return f"{cols} pays · {rows:,} lignes · {start} → {end}"


# ══════════════════════════════════════════════════════════════════════════════
#  FETCH API ENTSO-E
# ══════════════════════════════════════════════════════════════════════════════

def get_api_key() -> str:
    return st.secrets.get("ENTSOE_TOKEN", os.environ.get("ENTSOE_TOKEN", ""))


def fetch_one_api(code: str, start: date, end: date) -> pd.Series | None:
    """Télécharge les prix pour un pays depuis l'API ENTSO-E."""
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        from entsoe import EntsoePandasClient
        from entsoe.exceptions import NoMatchingDataError
        client   = EntsoePandasClient(api_key=api_key)
        ts_start = pd.Timestamp(start.isoformat(), tz="Europe/Paris")
        ts_end   = pd.Timestamp(end.isoformat(),   tz="Europe/Paris") + pd.Timedelta(days=1)
        sr = client.query_day_ahead_prices(code, start=ts_start, end=ts_end)
        return sr if (sr is not None and len(sr) > 0) else None
    except Exception:
        return None


def fetch_pays_api(pays_list: list[str], start: date, end: date) -> dict:
    """
    Télécharge plusieurs pays en parallèle via l'API.
    Retourne {nom_pays: Series}.
    Ajoute automatiquement les données dans la base locale.
    """
    results = {}

    def _fetch(nom):
        code = NOM_TO_CODE.get(nom)
        if not code:
            return nom, None
        sr = fetch_one_api(code, start, end)
        return nom, sr

    with ThreadPoolExecutor(max_workers=min(len(pays_list), 8)) as ex:
        futures = {ex.submit(_fetch, nom): nom for nom in pays_list}
        for future in as_completed(futures):
            nom, sr = future.result()
            if sr is not None:
                results[nom] = sr

    # Fusionner dans la base
    if results:
        df_new = pd.DataFrame(results)
        df_new.index = pd.to_datetime(df_new.index, utc=True)
        merge_into_db(df_new)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  RÉSOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "Horaire":   return df
    if freq == "Journalier":  return df.resample("1D").mean()
    if freq == "Hebdomadaire": return df.resample("1W").mean()
    if freq == "Mensuel":   return df.resample("1ME").mean()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ Prix Day-Ahead")
    st.markdown("---")

    # ── Uploader ───────────────────────────────────────────────────────────────
    st.markdown("### 📂 Importer des données")
    uploaded = st.file_uploader(
        "Fichier Parquet (index datetime, colonnes = pays)",
        type=["parquet"],
        help="Le fichier produit par le script de téléchargement est directement compatible.",
    )
    if uploaded is not None:
        try:
            df_up = pd.read_parquet(uploaded)
            # Normaliser l'index
            df_up.index = pd.to_datetime(df_up.index, utc=True)
            df_up.index.name = "timestamp"
            merged = merge_into_db(df_up)
            st.success(
                f"✓ Données importées : {df_up.shape[1]} pays, "
                f"{df_up.shape[0]:,} lignes\n\n"
                f"Base mise à jour : {db_info(merged)}"
            )
        except Exception as e:
            st.error(f"Erreur lors de l'import : {e}")

    st.markdown("---")

    # ── Dates ──────────────────────────────────────────────────────────────────
    st.markdown("### 📅 Période affichée")
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

    # ── Résolution ─────────────────────────────────────────────────────────────
    st.markdown("### 📊 Résolution")
    resolution = st.selectbox(
        "Agréger par",
        ["Horaire", "Journalier", "Hebdomadaire", "Mensuel"],
        index=1,
    )

    st.markdown("---")

    # ── Sélection pays ─────────────────────────────────────────────────────────
    st.markdown("### 🌍 Pays")

    col_a, col_b = st.columns(2)
    with col_a:
        tout_cocher   = st.button("Tout cocher",   use_container_width=True)
    with col_b:
        tout_decocher = st.button("Tout décocher", use_container_width=True)

    if "selection" not in st.session_state:
        st.session_state.selection = {p: p in ["France", "Allemagne"] for p in PAYS.values()}
    if tout_cocher:
        st.session_state.selection = {p: True for p in PAYS.values()}
    if tout_decocher:
        st.session_state.selection = {p: False for p in PAYS.values()}

    # Charger la base pour savoir quels pays sont déjà disponibles
    db = load_db()
    pays_en_base = set(db.columns.tolist()) if not db.empty else set()

    for i, nom in enumerate(PAYS.values()):
        en_base = "✓" if nom in pays_en_base else "○"
        checked = st.checkbox(
            f"{en_base} {nom}",
            value=st.session_state.selection.get(nom, False),
            key=f"cb_{nom}",
        )
        st.session_state.selection[nom] = checked

    pays_selectionnes = [p for p, v in st.session_state.selection.items() if v]

    st.markdown("---")
    st.caption("✓ = données en base  ·  ○ = à télécharger via API")

    # ── API fetch pour pays manquants ──────────────────────────────────────────
    pays_manquants = [p for p in pays_selectionnes if p not in pays_en_base]
    if pays_manquants and get_api_key():
        st.markdown("### 🔄 Données manquantes")
        st.warning(f"{len(pays_manquants)} pays pas encore en base :\n" +
                   ", ".join(pays_manquants))
        if st.button("Télécharger via API", use_container_width=True):
            with st.spinner("Téléchargement en cours…"):
                fetch_pays_api(pays_manquants, date_debut, date_fin)
                db = load_db()
            st.success("✓ Base mise à jour")
            st.rerun()
    elif pays_manquants and not get_api_key():
        st.info("Importe un fichier Parquet pour ces pays ou ajoute ton token ENTSO-E dans les secrets.")

    st.caption(f"Source : ENTSO-E  ·  Base : {db_info(db)}")


# ══════════════════════════════════════════════════════════════════════════════
#  ZONE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

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

# ── Préparer les données depuis la base ────────────────────────────────────────
if db.empty:
    st.warning("Base de données vide. Importe un fichier Parquet ou télécharge via l'API.")
    st.stop()

# Filtrer par pays sélectionnés et disponibles
pays_dispo = [p for p in pays_selectionnes if p in db.columns]
pays_absents = [p for p in pays_selectionnes if p not in db.columns]

if pays_absents:
    st.warning(f"Données non disponibles pour : {', '.join(pays_absents)}")

if not pays_dispo:
    st.error("Aucune donnée disponible. Importe un fichier ou télécharge via l'API.")
    st.stop()

# Filtrer par période
ts_debut = pd.Timestamp(date_debut.isoformat(), tz="UTC")
ts_fin   = pd.Timestamp(date_fin.isoformat(),   tz="UTC") + pd.Timedelta(days=1)
df_view  = db.loc[ts_debut:ts_fin, pays_dispo].copy()

if df_view.empty:
    st.warning("Pas de données pour cette période. Essaie une autre plage de dates.")
    st.stop()

# Convertir en heure de Paris et appliquer la résolution
df_view.index = df_view.index.tz_convert("Europe/Paris")
df_view = resample(df_view, resolution)

# ── Graphique ──────────────────────────────────────────────────────────────────
fig = go.Figure()

for nom in pays_dispo:
    idx   = list(PAYS.values()).index(nom) if nom in PAYS.values() else 0
    color = PALETTE[idx % len(PALETTE)]
    fig.add_trace(go.Scatter(
        x=df_view.index,
        y=df_view[nom].values,
        name=nom,
        line=dict(color=color, width=1.8),
        mode="lines",
    ))

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
for nom in pays_dispo:
    sr = df_view[nom].dropna()
    if len(sr) == 0:
        continue
    stats_rows.append({
        "Pays":              nom,
        "Moyenne (€/MWh)":  round(float(sr.mean()), 1),
        "Médiane":          round(float(sr.median()), 1),
        "Min":              round(float(sr.min()), 1),
        "Max":              round(float(sr.max()), 1),
        "Heures ≤ 0":      int((sr <= 0).sum()),
        "% heures ≤ 0":    f"{(sr <= 0).mean()*100:.1f} %",
    })

if stats_rows:
    df_stats = pd.DataFrame(stats_rows).set_index("Pays")

    def color_mean(v):
        try:
            f = float(str(v).replace(",", "."))
            if f < 0:     return "background-color:#7f1d1d;color:white"
            elif f < 30:  return "background-color:#1a3a2a;color:white"
            elif f > 150: return "background-color:#451a03;color:white"
        except Exception:
            pass
        return ""

    st.dataframe(
        df_stats.style.map(color_mean, subset=["Moyenne (€/MWh)"]),
        use_container_width=True,
    )

    total_neg = sum(int((df_view[n] <= 0).sum()) for n in pays_dispo)
    if total_neg > 0:
        detail = " · ".join(
            f"{n} : {int((df_view[n] <= 0).sum())}h"
            for n in pays_dispo if (df_view[n] <= 0).any()
        )
        st.info(f"⚠️ **{total_neg} heures à prix nul ou négatif** — {detail}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Base locale : {db_info(db)}  ·  "
    "Données ENTSO-E Transparency Platform  ·  "
    "Prix Day-Ahead SDAC (ex-PCR) en €/MWh"
)
