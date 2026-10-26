# PAGE ACCUEIL : chiffres clés, contexte, navigation

# Import des bibliothèques
import streamlit as st
import plotly.express as px

# Fonction de rendu de la page d'accueil
def rendu_accueil(base):
    # Affichage de la page d'accueil avec KPIs, contexte et navigation
    
    # En-tête
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #4a90d9 100%); padding: 40px; 
        border-radius: 15px; margin-bottom: 30px; box-shadow: 0 4px 15 px rgba(0,0,0,0.2)">
            <h1 style="color: #b8d4f0; text-align: center; font-size: 2.5em; margin-bottom: 10px">
            Projet Assurance Assistance
            </h1>
            <h3 style="color: #b8d4f0; text-align: center; font-weight: normal; margin-bottom: 5px">
            Analyse et prédiction de la durée de traitement des dossiers
            </h3>
            <p style="color: #d0e4f7; text-align: center; font-size: 1.1em">
            DU BDAP 2025-2026 | Université de Montpellier
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Métriques principales
    st.markdown("### Chiffres clés")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total dossiers (Train)", value=f"{len(base):,}")
    with col2:
        duree_moy = base["duree_corrigee_totale"].mean() / 60
        st.metric(label="Durée moyenne", value=f"{duree_moy:.0f} min")
    with col3:
        st.metric(label="Clients", value=f"{base['Client'].nunique()}")
    with col4:
        st.metric(label="Causes d'intervention", value=f"{base['Cause.intervention'].nunique()}")
        
    st.markdown("---")
    
    # Description du projet
    st.markdown("""
            ## Contexte du projet
            
            Ce projet s'inscrit dans le cadre du **DU Big Data, Data Science et Analyse des Risques sous Python 2025 - 2026** 
            de l'Université de Montpellier. En tant que data scientists juniors dans une entreprise d'assurance, nous analysons les 
            **dossiers d'assistance automobile** ouverts entre 2021 et 2022. 
            
            ### Objectifs
            
            1. **Traitement de données** : Nettoyage, détection d'anomalies, jointures entre les 3 tables
            2. **Analyse descriptive** : comprendre les distributions et les relations entre variables
            3. **Économétrie** : modélisation paramétrique (GLM Gamma, Inverse Gaussien)
            4. **Machine Learning** : modèles classiques (Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost) et modèles de Deep Learning
            5. **Mise en production** : interface interactive Streamlit
            
            ### Données disponibles
            | Table | Description | Lignes |
            |-------|-------------|--------|
            | `dossier.csv` | Dossiers d'assistance (1 ligne = 1 dossier) | 101234 |
            | `temps.csv` | Temps passé par matricule sur chaque dossier | 431598 |
            | `ressources.csv` | Profil des agents (contrat, expérience, lieu, ...) | 389353 |
        """)
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### Distribution de la durée de traitement")
        fig_duree = px.histogram(
            base, x=base['duree_corrigee_totale'] / 60, nbins=100,
            labels={'x': "Durée (minutes)"}, color_discrete_sequence=['#3498db']
        )
        fig_duree.update_layout(xaxis_title="Durée (minutes)", yaxis_title="Nombre de dossiers", height=350, showlegend=False)
        mediane = base['duree_corrigee_totale'].median() / 60
        fig_duree.add_vline(x=mediane, line_dash="dash", line_color="red", annotation_text=f"Médiane = {mediane:.0f} min")
        st.plotly_chart(fig_duree, width='stretch')
        
    with col_b:
        st.markdown("### Evolution mensuelle du nombre de dossiers")
        monthly = base.groupby(base['date.ouverture'].dt.to_period('M')).size().reset_index()
        monthly.columns = ['Mois', 'Nombre']
        monthly['Mois'] = monthly['Mois'].astype(str)
        fig_monthly = px.line(monthly, x='Mois', y='Nombre', markers=True, color_discrete_sequence=['#2ecc71'])
        fig_monthly.update_layout(xaxis_title="Mois", yaxis_title="Nombre de dossiers", height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig_monthly, width='stretch')
    
    st.markdown("---")
    
    st.markdown("""
        **Équipe projet** :
        - Dorcasse Yankam
        - Djawadou Djobo
        - Bénédicte Wantineki
    """)
