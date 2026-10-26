# =============================================================
# PAGE DATAVIZ - ANALYSE DESCRIPTIVE ET VISUALISATIONS
# =============================================================
# Objectif : Explorer visuellement la base fusionnée (base_model.csv)
# =============================================================

import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
print("matplotlib fonctionne")

# ─── Constantes ────────────────────────────────────────────────────────────────

TOP_VARS = [
    "TOP.D.R", "TOP.VR", "TOP.Rappat.valide",
    "TOP.Poursuite", "TOP.Recup", "TOP.Autres.Garanties",
]
TOP_LABELS = [
    "Dépannage / Remorquage", "Véhicule de Remplacement",
    "Rapatriement", "Poursuite Voyage",
    "Récupération", "Autres Garanties",
]

JOURS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
JOURS_FR    = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

COULEURS_CAUSE = px.colors.qualitative.Set2
COULEUR_BLEU   = "#2d6a9f"
COULEUR_VERT   = "#2ecc71"
COULEUR_ROUGE  = "#e74c3c"

# ─── Helpers(fonctions utilitaires)────────────────────────────────────────────────────────────────

def fmt_duree(secondes: float) -> str:
    """Formate une durée en secondes → 'HHh MMmin' ou 'MMmin SSs'."""
    if pd.isna(secondes):
        return "—"
    s = int(secondes)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h {m:02d}min" if h else f"{m}min {sec:02d}s"

def _preparer_base(base: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les colonnes dérivées nécessaires aux graphiques.
    Appelé une seule fois à l'entrée de render_dataviz().
    """
    df = base.copy()

    # Colonne datetime
    df["date.ouverture"] = pd.to_datetime(df["date.ouverture"], errors="coerce")

    # Période mois (string pour l'affichage)
    df["periode"] = df["date.ouverture"].dt.to_period("M").astype(str)

    # Jour de la semaine (en français)
    df["jour_en"] = df["date.ouverture"].dt.day_name()
    jour_map = dict(zip(JOURS_ORDER, JOURS_FR))
    df["jour_fr"] = df["jour_en"].map(jour_map)

    # Heure d'ouverture
    df["heure"] = pd.to_datetime(
        df["heure.ouverture"], format="%H:%M:%S", errors="coerce"
    ).dt.hour

    # Durée en minutes (pour les affichages)
    df["duree_min"] = df["duree_corrigee_totale"] / 60

    return df

# ─── Onglet 1 : Temporel ───────────────────────────────────────────────────────

def _tab_temporel(df: pd.DataFrame):
    st.subheader("Évolution temporelle des dossiers")

    # --- Évolution mensuelle (volume + durée moyenne) ---
    monthly = (
        df.groupby("periode")
        .agg(
            nb_dossiers=("Numero_dossier_ID", "count"),
            duree_moy=("duree_corrigee_totale", "mean"),
        )
        .reset_index()
    )
    monthly["duree_moy_min"] = monthly["duree_moy"] / 60

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly["periode"], y=monthly["nb_dossiers"],
            name="Nb dossiers", marker_color=COULEUR_BLEU, opacity=0.8,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["periode"], y=monthly["duree_moy_min"],
            name="Durée moy. (min)", mode="lines+markers",
            line=dict(color=COULEUR_ROUGE, width=2),
            marker=dict(size=5),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Volume de dossiers et durée moyenne par mois",
        xaxis_title="Mois",
        legend=dict(orientation="h", y=1.1),
        height=420,
        xaxis_tickangle=-45,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Nombre de dossiers", secondary_y=False)
    fig.update_yaxes(title_text="Durée moyenne (min)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        " Les barres bleues montrent le volume mensuel de dossiers ouverts. "
        "La courbe rouge représente la durée moyenne de traitement. "
        "Des pics de volume en été (juillet-août) s'accompagnent souvent d'une durée plus élevée, "
        "ce qui peut refléter une surcharge des équipes en période estivale."
    )

    st.markdown("---")

    # --- Répartition par jour de la semaine ---
    st.subheader("Répartition par jour de la semaine")

    jours_fr_order = JOURS_FR[:5]  # Lundi → Vendredi (jours ouvrés dominants)
    jour_counts = (
        df[df["jour_fr"].isin(JOURS_FR)]
        .groupby("jour_fr")
        .agg(nb=("Numero_dossier_ID", "count"), duree_moy=("duree_min", "mean"))
        .reindex(JOURS_FR)
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        fig_jour = px.bar(
            jour_counts, x="jour_fr", y="nb",
            labels={"jour_fr": "Jour", "nb": "Nombre de dossiers"},
            title="Volume par jour",
            color="nb",
            color_continuous_scale="Blues",
        )
        fig_jour.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_jour, use_container_width=True)

    with col2:
        fig_dur_jour = px.bar(
            jour_counts, x="jour_fr", y="duree_moy",
            labels={"jour_fr": "Jour", "duree_moy": "Durée moy. (min)"},
            title="Durée moyenne par jour",
            color="duree_moy",
            color_continuous_scale="Reds",
        )
        fig_dur_jour.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_dur_jour, use_container_width=True)

    st.info(
        " Le volume de dossiers est concentré sur les jours ouvrés (lundi–vendredi). "
        "Le week-end, les dossiers ouverts sont moins nombreux mais peuvent présenter une durée "
        "de traitement différente (effectifs réduits)."
    )

    st.markdown("---")

    # --- Distribution par heure d'ouverture ---
    st.subheader("Répartition par heure d'ouverture")

    heure_counts = (
        df.dropna(subset=["heure"])
        .groupby("heure")
        .size()
        .reset_index(name="nb")
    )

    fig_heure = px.bar(
        heure_counts, x="heure", y="nb",
        labels={"heure": "Heure (0–23)", "nb": "Nombre de dossiers"},
        title="Dossiers ouverts par heure de la journée",
        color="nb",
        color_continuous_scale="Blues",
    )
    fig_heure.update_layout(height=350, coloraxis_showscale=False)
    st.plotly_chart(fig_heure, use_container_width=True)

    st.info(
        " L'activité est concentrée entre 8h et 18h, avec un pic en milieu de matinée. "
        "Peu de dossiers sont ouverts la nuit, ce qui reflète les horaires des équipes d'assistance."
    )


# ─── Onglet 2 : Causes & Services ─────────────────────────────────────────────

def _tab_causes_services(df: pd.DataFrame):
    st.subheader("Causes d'intervention")

    cause_stats = (
        df.groupby("Cause.intervention")
        .agg(
            nb=("Numero_dossier_ID", "count"),
            duree_moy=("duree_min", "mean"),
            duree_med=("duree_min", "median"),
        )
        .sort_values("nb", ascending=False)
        .reset_index()
    )
    cause_stats["duree_moy"] = cause_stats["duree_moy"].round(1)
    cause_stats["duree_med"] = cause_stats["duree_med"].round(1)

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            cause_stats, values="nb", names="Cause.intervention",
            title="Répartition des dossiers par cause",
            color_discrete_sequence=COULEURS_CAUSE,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar_cause = px.bar(
            cause_stats.sort_values("duree_moy"),
            x="duree_moy", y="Cause.intervention",
            orientation="h",
            labels={"duree_moy": "Durée moy. (min)", "Cause.intervention": "Cause"},
            title="Durée moyenne par cause",
            color="duree_moy",
            color_continuous_scale="Blues",
        )
        fig_bar_cause.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_bar_cause, use_container_width=True)

    # Tableau récapitulatif
    with st.expander("Tableau détaillé par cause"):
        affichage = cause_stats.rename(columns={
            "Cause.intervention": "Cause",
            "nb": "Nb dossiers",
            "duree_moy": "Durée moy. (min)",
            "duree_med": "Durée méd. (min)",
        })
        st.dataframe(affichage, use_container_width=True, hide_index=True)

    st.info(
        " La **Panne mécanique** représente la grande majorité des dossiers. "
        "Cependant, ce sont les causes rares comme **Bris de glace** et **Incendie** qui génèrent "
        "les durées de traitement les plus longues, car elles impliquent des services plus complexes."
    )

    st.markdown("---")

    # --- Services d'assistance (TOP) ---
    st.subheader("Services d'assistance activés (variables TOP)")

    taux_top = pd.DataFrame({
        "Service": TOP_LABELS,
        "Taux (%)": [df[v].mean() * 100 for v in TOP_VARS],
        "Durée moy. avec (min)": [
            df.loc[df[v] == 1, "duree_min"].mean() for v in TOP_VARS
        ],
        "Durée moy. sans (min)": [
            df.loc[df[v] == 0, "duree_min"].mean() for v in TOP_VARS
        ],
    }).round(1)

    fig_top = px.bar(
        taux_top.sort_values("Taux (%)"),
        x="Taux (%)", y="Service",
        orientation="h",
        title="Taux d'activation de chaque service d'assistance",
        color="Taux (%)",
        color_continuous_scale="Blues",
        text="Taux (%)",
    )
    fig_top.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_top.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig_top, use_container_width=True)

    # Comparaison durée avec / sans service
    st.markdown("**Impact des services sur la durée de traitement**")

    fig_impact = go.Figure()
    fig_impact.add_trace(go.Bar(
        name="Avec le service",
        x=taux_top["Service"],
        y=taux_top["Durée moy. avec (min)"],
        marker_color=COULEUR_BLEU,
    ))
    fig_impact.add_trace(go.Bar(
        name="Sans le service",
        x=taux_top["Service"],
        y=taux_top["Durée moy. sans (min)"],
        marker_color="#a8c8e8",
    ))
    fig_impact.update_layout(
        barmode="group",
        title="Durée moyenne selon l'activation du service",
        xaxis_title="Service",
        yaxis_title="Durée moyenne (min)",
        height=420,
        xaxis_tickangle=-30,
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_impact, use_container_width=True)

    st.info(
        " Le service **Dépannage/Remorquage** est activé dans presque tous les dossiers. "
        "Les services **Véhicule de Remplacement**, **Rapatriement** et **Poursuite Voyage** sont rares "
        "mais augmentent fortement la durée de traitement — ce sont des indicateurs de complexité du dossier."
    )


# ─── Onglet 3 : Agents & RH ───────────────────────────────────────────────────

def _tab_agents_rh(df: pd.DataFrame):
    st.subheader("Profil des agents intervenant sur les dossiers")

    col1, col2, col3 = st.columns(3)

    # Population CAC / CAS
    with col1:
        pop_counts = df["pop_mode"].value_counts().reset_index()
        pop_counts.columns = ["Population", "Nb dossiers"]
        fig_pop = px.pie(
            pop_counts, values="Nb dossiers", names="Population",
            title="Population (CAC / CAS)",
            color_discrete_sequence=["#2d6a9f", "#4a90d9"],
            hole=0.4,
        )
        fig_pop.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_pop, use_container_width=True)

    # Lieu de travail
    with col2:
        lieu_counts = df["prop_tele"].apply(
            lambda x: "Télétravail" if x > 0.5 else "Sur site"
        ).value_counts().reset_index()
        lieu_counts.columns = ["Lieu", "Nb dossiers"]
        fig_lieu = px.pie(
            lieu_counts, values="Nb dossiers", names="Lieu",
            title="Lieu de travail dominant",
            color_discrete_sequence=["#2ecc71", "#e67e22"],
            hole=0.4,
        )
        fig_lieu.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_lieu, use_container_width=True)

    # Type de contrat
    with col3:
        contrat_counts = df["type_contrat_mode"].value_counts().reset_index()
        contrat_counts.columns = ["Contrat", "Nb dossiers"]
        fig_contrat = px.pie(
            contrat_counts, values="Nb dossiers", names="Contrat",
            title="Type de contrat dominant",
            color_discrete_sequence=["#9b59b6", "#e74c3c", "#f39c12"],
            hole=0.4,
        )
        fig_contrat.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_contrat, use_container_width=True)

    st.markdown("---")

    # --- Expérience vs durée ---
    st.subheader("Expérience des agents et durée de traitement")

    # Découpage en tranches d'expérience
    df_exp = df.dropna(subset=["exp_moy", "duree_min"]).copy()
    df_exp["tranche_exp"] = pd.cut(
        df_exp["exp_moy"],
        bins=[0, 200, 500, 1000, 2000, 6200],
        labels=["< 200j", "200–500j", "500j–1an", "1–2 ans", "> 2 ans"],
    )

    exp_stats = (
        df_exp.groupby("tranche_exp", observed=True)
        .agg(
            nb=("Numero_dossier_ID", "count"),
            duree_moy=("duree_min", "mean"),
            duree_med=("duree_min", "median"),
        )
        .reset_index()
        .rename(columns={"tranche_exp": "Expérience"})
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig_exp_nb = px.bar(
            exp_stats, x="Expérience", y="nb",
            title="Nombre de dossiers par tranche d'expérience",
            labels={"nb": "Nb dossiers", "Expérience": "Expérience cumulée"},
            color="nb", color_continuous_scale="Blues",
        )
        fig_exp_nb.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_exp_nb, use_container_width=True)

    with col_b:
        fig_exp_dur = px.bar(
            exp_stats, x="Expérience", y="duree_moy",
            title="Durée moyenne par tranche d'expérience",
            labels={"duree_moy": "Durée moy. (min)", "Expérience": "Expérience cumulée"},
            color="duree_moy", color_continuous_scale="Reds",
        )
        fig_exp_dur.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_exp_dur, use_container_width=True)

    st.info(
        " Les agents les plus expérimentés (> 2 ans) traitent des dossiers plus nombreux. "
        "La durée de traitement tend à varier selon l'expérience — les agents juniors peuvent être plus "
        "lents, mais les agents seniors héritent souvent de dossiers plus complexes."
    )

    st.markdown("---")

    # --- Télétravail vs durée ---
    st.subheader("Impact du télétravail sur la durée de traitement")

    df_tele = df.copy()
    df_tele["mode_travail"] = df_tele["prop_tele"].apply(
        lambda x: "Majoritairement télétravail" if x > 0.5
        else ("Mix" if x > 0.1 else "Majoritairement sur site")
    )

    tele_stats = (
        df_tele.groupby("mode_travail")
        .agg(nb=("Numero_dossier_ID", "count"), duree_moy=("duree_min", "mean"))
        .reset_index()
        .round(1)
    )

    fig_tele = px.bar(
        tele_stats, x="mode_travail", y="duree_moy",
        title="Durée moyenne selon le mode de travail",
        labels={"mode_travail": "Mode de travail", "duree_moy": "Durée moy. (min)"},
        color="duree_moy", color_continuous_scale="Blues",
        text="duree_moy",
    )
    fig_tele.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
    fig_tele.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_tele, use_container_width=True)

    st.info(
        " La corrélation entre télétravail et durée est légèrement négative. "
        "Le mode de travail a un effet modeste sur la durée de traitement des dossiers."
    )


# ─── Onglet 4 : Durée de traitement ───────────────────────────────────────────

def _tab_duree(df: pd.DataFrame):
    st.subheader("Distribution de la durée de traitement")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Durée moyenne", fmt_duree(df["duree_corrigee_totale"].mean()))
    col2.metric("Durée médiane", fmt_duree(df["duree_corrigee_totale"].median()))
    col3.metric("Durée max", fmt_duree(df["duree_corrigee_totale"].max()))
    col4.metric("Nb dossiers", f"{len(df):,}")

    # Histogramme de la durée
    fig_hist = px.histogram(
        df, x="duree_min",
        nbins=120,
        range_x=[0, 120],
        labels={"duree_min": "Durée (minutes)"},
        title="Distribution de la durée de traitement (tronquée à 120 min)",
        color_discrete_sequence=[COULEUR_BLEU],
    )
    mediane_min = df["duree_min"].median()
    fig_hist.add_vline(
        x=mediane_min, line_dash="dash", line_color=COULEUR_ROUGE,
        annotation_text=f"Médiane = {mediane_min:.0f} min",
        annotation_position="top right",
    )
    fig_hist.update_layout(
        height=380, yaxis_title="Nombre de dossiers",
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.info(
        " La distribution est fortement asymétrique à droite - la majorité des dossiers "
        "sont traités en moins de 30 minutes, mais une queue longue reflète les dossiers complexes "
        "(rapatriement, véhicule de remplacement, etc.)."
    )

    st.markdown("---")

    # --- Durée par cause ---
    st.subheader("Durée de traitement par cause d'intervention")

    duree_cause = (
        df.groupby("Cause.intervention")
        .agg(
            mediane=("duree_min", "median"),
            moyenne=("duree_min", "mean"),
            q1=("duree_min", lambda x: x.quantile(0.25)),
            q3=("duree_min", lambda x: x.quantile(0.75)),
        )
        .sort_values("mediane")
        .reset_index()
        .round(1)
    )

    fig_box_cause = px.box(
        df[df["duree_min"] <= 200],
        x="Cause.intervention", y="duree_min",
        labels={
            "Cause.intervention": "Cause d'intervention",
            "duree_min": "Durée (minutes)",
        },
        title="Boxplot de la durée par cause (tronqué à 200 min)",
        color="Cause.intervention",
        color_discrete_sequence=COULEURS_CAUSE,
    )
    fig_box_cause.update_layout(
        height=450, showlegend=False,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_box_cause, use_container_width=True)

    st.info(
        " Les boxplots montrent la dispersion de la durée pour chaque cause. "
        "Clés/Carburant/Crevaison est la plus rapide à traiter. Incendie et Bris de glace "
        "sont les plus longs, avec une forte variabilité (nombreux outliers)."
    )

    st.markdown("---")

    # --- Durée par type d'énergie ---
    st.subheader("Durée de traitement par type d'énergie")

    col_e1, col_e2 = st.columns(2)

    energie_stats = (
        df.groupby("Type.d.energie")
        .agg(nb=("Numero_dossier_ID", "count"), duree_moy=("duree_min", "mean"))
        .sort_values("duree_moy", ascending=False)
        .reset_index()
        .round(1)
    )

    with col_e1:
        fig_energie_nb = px.bar(
            energie_stats.sort_values("nb"),
            x="nb", y="Type.d.energie",
            orientation="h",
            title="Nb dossiers par type d'énergie",
            labels={"nb": "Nb dossiers", "Type.d.energie": "Énergie"},
            color="nb", color_continuous_scale="Blues",
        )
        fig_energie_nb.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_energie_nb, use_container_width=True)

    with col_e2:
        fig_energie_dur = px.bar(
            energie_stats,
            x="duree_moy", y="Type.d.energie",
            orientation="h",
            title="Durée moy. par type d'énergie",
            labels={"duree_moy": "Durée moy. (min)", "Type.d.energie": "Énergie"},
            color="duree_moy", color_continuous_scale="Reds",
        )
        fig_energie_dur.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_energie_dur, use_container_width=True)

    st.info(
        " Les véhicules **Diesel** sont les plus représentés dans les dossiers. "
        "Les véhicules **Électriques** génèrent des durées de traitement plus longues en moyenne, "
        "probablement car leur assistance est plus spécialisée (moins de prestataires disponibles)."
    )

    st.markdown("---")

    # --- Nb d'intervenants vs durée ---
    st.subheader("Nombre d'intervenants et durée de traitement")

    interv_stats = (
        df[df["nb_intervenants"] <= 15]
        .groupby("nb_intervenants")
        .agg(
            nb=("Numero_dossier_ID", "count"),
            duree_moy=("duree_min", "mean"),
        )
        .reset_index()
        .round(1)
    )

    fig_interv = make_subplots(specs=[[{"secondary_y": True}]])
    fig_interv.add_trace(
        go.Bar(
            x=interv_stats["nb_intervenants"], y=interv_stats["nb"],
            name="Nb dossiers", marker_color=COULEUR_BLEU, opacity=0.7,
        ),
        secondary_y=False,
    )
    fig_interv.add_trace(
        go.Scatter(
            x=interv_stats["nb_intervenants"], y=interv_stats["duree_moy"],
            name="Durée moy. (min)", mode="lines+markers",
            line=dict(color=COULEUR_ROUGE, width=2),
        ),
        secondary_y=True,
    )
    fig_interv.update_layout(
        title="Durée moyenne selon le nombre d'intervenants",
        xaxis_title="Nombre d'intervenants",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )
    fig_interv.update_yaxes(title_text="Nombre de dossiers", secondary_y=False)
    fig_interv.update_yaxes(title_text="Durée moyenne (min)", secondary_y=True)
    st.plotly_chart(fig_interv, use_container_width=True)

    st.info(
        " Plus un dossier implique "
        "d'intervenants différents, plus sa durée de traitement est élevée. "
        "C'est l'un des prédicteurs les plus importants dans les modèles ML et GLM."
    )


# ─── Point d'entrée principal ─────────────────────────────────────────────────

def render_dataviz(base: pd.DataFrame):
    """
    Fonction principale appelée depuis app.py.
    Prend la base d'apprentissage (base_model.csv) en entrée.
    """
    st.title("DataViz — Analyse Descriptive")
    st.markdown(
        "Ce module propose une exploration visuelle des **86 950 dossiers** de la base d'apprentissage. "
        "Les graphiques sont interactifs : survolez, zoomez, filtrez."
    )

    # Préparation des colonnes dérivées
    df = _preparer_base(base)

    st.markdown("---")

    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Temporel",
        "🔍 Causes et Services",
        "👥 Agents et RH",
        "⏱️ Durée",
    ])

    with tab1:
        _tab_temporel(df)

    with tab2:
        _tab_causes_services(df)

    with tab3:
        _tab_agents_rh(df)

    with tab4:
        _tab_duree(df)
