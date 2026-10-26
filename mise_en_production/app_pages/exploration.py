# PAGE EXPLORATION - Filtres interactifs

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def rendu_exploration(base):
    """Affiche la page d'exploration interactive avec filtres dynamiques."""

    st.markdown("# Exploration interactive des données")
    st.markdown("Utilisez les filtres ci-dessous pour explorer les données librement.")

    # Filtres
    st.markdown("### Filtres")

    col1, col2, col3 = st.columns(3)

    with col1:
        causes = ["Toutes"] + sorted(base["Cause.intervention"].dropna().unique().tolist())
        cause_filter = st.selectbox("Cause d'intervention", causes)

    with col2:
        energies = ["Toutes"] + sorted(base["Type.d.energie"].dropna().unique().tolist())
        energie_filter = st.selectbox("Type d'énergie", energies)

    with col3:
        outils = ["Tous"] + sorted(base["Outil.d.assistance"].dropna().unique().tolist())
        outil_filter = st.selectbox("Outil d'assistance", outils)

    # Application des filtres
    filtered = base.copy()
    if cause_filter != "Toutes":
        filtered = filtered[filtered["Cause.intervention"] == cause_filter]
    if energie_filter != "Toutes":
        filtered = filtered[filtered["Type.d.energie"] == energie_filter]
    if outil_filter != "Tous":
        filtered = filtered[filtered["Outil.d.assistance"] == outil_filter]

    st.info(f"**{len(filtered):,} dossiers** correspondent aux filtres sélectionnés.")

    st.markdown("---")

    # Statistiques du sous-ensemble
    st.markdown("### Statistiques du sous-ensemble")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de dossiers", f"{len(filtered):,}")
    with col2:
        st.metric("Durée moyenne", f"{filtered['duree_corrigee_totale'].mean() / 60:.0f} min")
    with col3:
        st.metric("Durée médiane", f"{filtered['duree_corrigee_totale'].median() / 60:.0f} min")
    with col4:
        st.metric("Nombre moyen d'intervenants", f"{filtered['nb_intervenants'].mean():.1f}")

    # Visualisations
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Distribution de la durée filtrée")
        fig = px.histogram(
            filtered,
            x=filtered["duree_corrigee_totale"] / 60,
            nbins=80,
            color_discrete_sequence=["#e74c3c"],
        )
        fig.update_layout(
            xaxis_title="Durée (minutes)",
            yaxis_title="Fréquence",
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.markdown("#### Services activés")
        top_vars = [
            "TOP.D.R",
            "TOP.VR",
            "TOP.Rappat.valide",
            "TOP.Poursuite",
            "TOP.Recup",
            "TOP.Autres.Garanties",
        ]
        top_labels_short = ["D/R", "VR", "Rapat.", "Poursuite", "Récup.", "Autres"]
        rates_filtered = [filtered[v].mean() * 100 for v in top_vars]
        rates_all = [base[v].mean() * 100 for v in top_vars]

        fig = go.Figure(data=[
            go.Bar(name="Filtres", x=top_labels_short, y=rates_filtered, marker_color="#e74c3c"),
            go.Bar(name="Global", x=top_labels_short, y=rates_all, marker_color="#95a5a6"),
        ])
        fig.update_layout(barmode="group", yaxis_title="Taux (%)", height=350)
        st.plotly_chart(fig, width="stretch")

    # Aperçu des données
    st.markdown("### Aperçu des données filtrées")
    st.dataframe(
        filtered[
            [
                "Numero_dossier_ID",
                "Client",
                "Cause.intervention",
                "Type.d.energie",
                "duree_corrigee_totale",
                "nb_intervenants",
                "nb_interventions",
            ]
        ].head(20),
        width="stretch",
        hide_index=True,
    )
