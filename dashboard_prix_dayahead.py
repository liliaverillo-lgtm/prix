import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import boto3
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from botocore.exceptions import ClientError

st.set_page_config(page_title="Prix Day-Ahead — ENTSO-E", page_icon="⚡", layout="wide")

R2_KEY    = "prix_dayahead/master_db.parquet"
LOCAL_TMP = "/tmp/master_db_prix.parquet"

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
NOM_TO_CODE = {v: k for k, v in PAYS.items()}

PALETTE = [
    "#4C9BE8", "#E24444", "#2DB87A", "#F5A623", "#A855F7",
    "#EC4899", "#14B8A6", "#F97316", "#84CC16", "#06B6D4",
    "#8B5CF6", "#EF4444", "#10B981", "#F59E0B", "#3B82F6",
]

@st.cache_resource
def get_r2_client():
    r2 = st.secrets["r2"]
    return boto3.client(
        "s3",
        endpoint_url          = f"https://{r2['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id     = r2["access_key_id"],
        aws_secret_access_key = r2["secret_access_key"],
        region_name           = "auto",
    )

def r2_bucket() -> str:
    return st.secrets["r2"]["bucket_name"]

def load_db() -> pd.DataFrame:
    if os.path.exists(LOCAL_TMP):
        df = pd.read_parquet(LOCAL_TMP)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    try:
        s3 = get_r2_client()
        s3.download_file(r2_bucket(), R2_KEY, LOCAL_TMP)
        df = pd.read_parquet(LOCAL_TMP)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return pd.DataFrame()
        raise

def save_db(df: pd.DataFrame):
    df.to_parquet(LOCAL_TMP)
    s3 = get_r2_client()
    s3.upload_file(LOCAL_TMP, r2_bucket(), R2_KEY)

def merge_into_db(new_df: pd.DataFrame) -> pd.DataFrame:
    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")
    else:
        new_df.index = new_df.index.tz_convert("UTC")
    existing = load_db()
    merged = new_df.combine_first(existing) if not existing.empty else new_df.copy()
    save_db(merged)
    return merged

def db_info(df: pd.DataFrame) -> str:
    if df.empty:
        return "Base vide"
    return (f"{df.shape[1]} pays · {df.shape[0]:,} lignes · "
            f"{df.index.min().strftime('%d/%m/%Y')} → "
            f"{df.index.max().strftime('%d/%m/%Y')}")

def pays_manquants_pour_periode(db: pd.DataFrame, pays_list: list, start: date, end: date) -> list:
    if db.empty:
        return pays_list
    manquants = []
    ts_start = pd.Timestamp(start.isoformat(), tz="UTC")
    ts_end   = pd.Timestamp(end.isoformat(),   tz="UTC") + pd.Timedelta(days=1)
    for pays in pays_list:
        if pays not in db.columns:
            manquants.append(pays)
            continue
        sr = db.loc[ts_start:ts_end, pays].dropna()
        if len(sr) == 0:
            manquants.append(pays)
    return manquants

@st.cache_resource
def get_entsoe_client():
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=st.secrets["ENTSOE_TOKEN"])

def fetch_pays(pays_list: list, start: date, end: date) -> int:
    from entsoe.exceptions import NoMatchingDataError
    ts_start = pd.Timestamp(start.isoformat(), tz="Europe/Paris")
    ts_end   = pd.Timestamp(end.isoformat(),   tz="Europe/Paris") + pd.Timedelta(days=1)
    results  = {}

    def _fetch_one(nom):
        code = NOM_TO_CODE.get(nom)
        if not code:
            return nom, None
        try:
            client = get_entsoe_client()
            sr = client.query_day_ahead_prices(code, start=ts_start, end=ts_end)
            return nom, sr if (sr is not None and len(sr) > 0) else (nom, None)
        except NoMatchingDataError:
            return nom, None
        except Exception:
            return nom, None

    with ThreadPoolExecutor(max_workers=min(len(pays_list), 8)) as ex:
        futures = {ex.submit(_fetch_one, nom): nom for nom in pays_list}
        for future in as_completed(futures):
            nom, sr = future.result()
            if sr is not None:
                results[nom] = sr

    if results:
        df_new = pd.DataFrame(results)
        df_new.index = pd.to_datetime(df_new.index, utc=True)
        merge_into_db(df_new)
    return len(results)

def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "Horaire":      return df
    if freq == "Journalier":   return df.resample("1D").mean()
    if freq == "Hebdomadaire": return df.resample("1W").mean()
    if freq == "Mensuel":      return df.resample("1ME").mean()
    return df

def _factor(sr: pd.Series) -> float:
    if len(sr) < 2:
        return 1.0
    diffs = sr.index.to_series().diff().dropna()
    return 4.0 if diffs.median() <= pd.Timedelta(minutes=15) else 1.0

def heures_negatives(sr: pd.Series) -> float:
    sr = sr.dropna()
    nb_pas = int((sr <= 0).sum())
    return nb_pas / _factor(sr)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Prix Day-Ahead")
    st.markdown("---")

    st.markdown("### 📅 Période")
    col1, col2 = st.columns(2)
    with col1:
        date_debut = st.date_input("Début", value=date.today() - timedelta(days=365),
                                   min_value=date(2015, 1, 1), max_value=date.today())
    with col2:
        date_fin = st.date_input("Fin", value=date.today() - timedelta(days=1),
                                 min_value=date(2015, 1, 1), max_value=date.today())
    if date_debut >= date_fin:
        st.error("La date de début doit être avant la date de fin.")
        st.stop()

    st.markdown("### 📊 Résolution")
    resolution = st.selectbox("Agréger par",
                               ["Horaire", "Journalier", "Hebdomadaire", "Mensuel"], index=1)
    st.markdown("---")

    # ── Sélection pays ────────────────────────────────────────────────────────
    st.markdown("### 🌍 Pays")

    # Initialisation des clés cb_* si première visite
    for nom in PAYS.values():
        if f"cb_{nom}" not in st.session_state:
            st.session_state[f"cb_{nom}"] = nom in ("France", "Allemagne")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Tout cocher", use_container_width=True):
            for nom in PAYS.values():
                st.session_state[f"cb_{nom}"] = True
    with col_b:
        if st.button("Tout décocher", use_container_width=True):
            for nom in PAYS.values():
                st.session_state[f"cb_{nom}"] = False

    db = load_db()
    pays_en_base = set(db.columns.tolist()) if not db.empty else set()
    for nom in PAYS.values():
        badge = "✓" if nom in pays_en_base else "○"
        st.checkbox(f"{badge} {nom}", key=f"cb_{nom}")

    # Lecture des pays sélectionnés depuis les clés session_state cb_*
    pays_selectionnes = [p for p in PAYS.values() if st.session_state.get(f"cb_{p}", False)]

    st.markdown("---")
    st.caption(f"✓ = en base R2 · ○ = à télécharger\n\n{db_info(db)}")

# ── AUTO-FETCH ────────────────────────────────────────────────────────────────
if pays_selectionnes:
    manquants = pays_manquants_pour_periode(db, pays_selectionnes, date_debut, date_fin)
    if manquants:
        with st.spinner(f"Téléchargement depuis ENTSO-E : {', '.join(manquants)}…"):
            n = fetch_pays(manquants, date_debut, date_fin)
            db = load_db()
            if n > 0:
                st.success(f"✓ {n} pays téléchargés et sauvegardés dans R2.")
            else:
                st.warning("Données non disponibles sur ENTSO-E pour cette sélection.")

# ── ZONE PRINCIPALE ───────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:1.8rem;margin-bottom:0'>⚡ Prix Day-Ahead ENTSO-E</h1>",
            unsafe_allow_html=True)
st.markdown(f"<p style='color:#888;margin-top:4px'>"
            f"{date_debut.strftime('%d/%m/%Y')} → {date_fin.strftime('%d/%m/%Y')} "
            f"· {resolution} · {len(pays_selectionnes)} pays</p>",
            unsafe_allow_html=True)

if not pays_selectionnes:
    st.info("Sélectionne au moins un pays dans le panneau gauche.")
    st.stop()

if db.empty:
    st.warning("Base R2 vide. Les données vont être téléchargées automatiquement.")
    st.stop()

pays_dispo   = [p for p in pays_selectionnes if p in db.columns]
pays_absents = [p for p in pays_selectionnes if p not in db.columns]
if pays_absents:
    st.warning(f"Données non disponibles pour : {', '.join(pays_absents)}")
if not pays_dispo:
    st.error("Aucune donnée disponible pour cette sélection.")
    st.stop()

ts_debut = pd.Timestamp(date_debut.isoformat(), tz="UTC")
ts_fin   = pd.Timestamp(date_fin.isoformat(),   tz="UTC") + pd.Timedelta(days=1)
df_view  = db.loc[ts_debut:ts_fin, pays_dispo].copy()
if df_view.empty:
    st.warning("Pas de données pour cette période.")
    st.stop()

df_view.index = df_view.index.tz_convert("Europe/Paris")
df_raw  = df_view.copy()
df_view = resample(df_view, resolution)

# ── Graphique ──────────────────────────────────────────────────────────────────
fig = go.Figure()
for nom in pays_dispo:
    idx   = list(PAYS.values()).index(nom) if nom in PAYS.values() else 0
    color = PALETTE[idx % len(PALETTE)]
    fig.add_trace(go.Scatter(x=df_view.index, y=df_view[nom].values,
                             name=nom, line=dict(color=color, width=1.8), mode="lines"))
fig.update_layout(height=520, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10))
fig.update_xaxes(rangeslider_visible=True)
fig.update_yaxes(title_text="€/MWh")
st.plotly_chart(fig, use_container_width=True)

# ── Statistiques ───────────────────────────────────────────────────────────────
st.markdown("### 📊 Statistiques")
rows = []
for nom in pays_dispo:
    sr = df_raw[nom].dropna()
    if len(sr) == 0:
        continue
    h_neg   = heures_negatives(sr)
    total_h = len(sr) / _factor(sr)
    pct     = h_neg / total_h * 100 if total_h > 0 else 0.0
    rows.append({
        "Pays":            nom,
        "Moyenne (€/MWh)": round(float(sr.mean()), 1),
        "Médiane":         round(float(sr.median()), 1),
        "Min":             round(float(sr.min()), 1),
        "Max":             round(float(sr.max()), 1),
        "Heures ≤ 0":      round(h_neg, 1),
        "% heures ≤ 0":    f"{pct:.1f} %",
    })

if rows:
    df_stats = pd.DataFrame(rows).set_index("Pays")

    def color_mean(v):
        try:
            f = float(str(v).replace(",", "."))
            if f < 0:   return "background-color:#7f1d1d;color:white"
            if f < 30:  return "background-color:#1a3a2a;color:white"
            if f > 150: return "background-color:#451a03;color:white"
        except Exception:
            pass
        return ""

    st.dataframe(df_stats.style.map(color_mean, subset=["Moyenne (€/MWh)"]),
                 use_container_width=True)

    total_neg_h = sum(heures_negatives(df_raw[n].dropna()) for n in pays_dispo)
    if total_neg_h > 0:
        detail = " · ".join(
            f"{n} : {round(heures_negatives(df_raw[n].dropna()), 1)}h"
            for n in pays_dispo if (df_raw[n] <= 0).any()
        )
        st.info(f"⚠️ **{round(total_neg_h, 1)} heures à prix nul ou négatif** — {detail}")

# ── Export ────────────────────────────────────────────────────────────────────
def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("Europe/Paris")
    out.index = out.index.strftime("%Y-%m-%d %H:%M")
    out.index.name = "timestamp (heure Paris)"
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.tz_localize(None)
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    buf = io.BytesIO()
    out.to_excel(buf, engine="openpyxl")
    return buf.getvalue()

@st.cache_data(show_spinner="Génération Excel…")
def build_excel_full(db: pd.DataFrame) -> bytes:
    return _df_to_excel_bytes(db)

@st.cache_data(show_spinner="Génération Excel…")
def build_excel_period(df_view: pd.DataFrame) -> bytes:
    return _df_to_excel_bytes(df_view)

st.markdown("### 💾 Télécharger les données")
if not db.empty:
    st.caption(f"Base disponible : {db_info(db)}")
    col1, col2 = st.columns(2)
    with col1:
        buf_pq = io.BytesIO()
        db.to_parquet(buf_pq)
        buf_pq.seek(0)
        st.download_button(label="⬇️ Télécharger en Parquet", data=buf_pq,
                           file_name="prix_dayahead_europe.parquet",
                           mime="application/octet-stream", use_container_width=True,
                           help="Format compact — recommandé pour Python/Pandas.")
        st.caption("⚡ Rapide · Petit fichier · Pour Python")
    with col2:
        st.download_button(label="⬇️ Télécharger en Excel — toute la période",
                           data=build_excel_full(db),
                           file_name="prix_dayahead_europe.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
        st.caption("📊 Format Excel · Tous pays · Toutes les dates")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.download_button(label="⬇️ Télécharger Excel — période sélectionnée",
                           data=build_excel_period(df_view),
                           file_name=f"prix_dayahead_{date_debut}_{date_fin}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True,
                           help=f"Uniquement du {date_debut} au {date_fin} pour les pays sélectionnés.")
    with col4:
        st.caption(f"Période : {date_debut.strftime('%d/%m/%Y')} → {date_fin.strftime('%d/%m/%Y')}\n\n"
                   f"Pays : {', '.join(pays_dispo)}\n\nRésolution : {resolution}")

st.markdown("---")
st.caption(f"Données ENTSO-E · Prix Day-Ahead SDAC · Base R2 : {db_info(db)}")
