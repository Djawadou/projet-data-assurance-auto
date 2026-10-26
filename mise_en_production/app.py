# =============================================================
# APPLICATION STREAMLIT - PROJET ASSURANCE ASSISTANCE

# Lancement : streamlit run app.py

# Import des bibliothèques
from pathlib import Path
import streamlit as st
import pandas as pd

# Import des fonctions de rendu (une par page)
from app_pages import accueil, dataviz, econometrie, exploration, machine_learning

ROOT_DIR = Path(__file__).resolve().parents[1]

# Configuration de la page
st.set_page_config(
    page_title="Assurance Assistance - DU BDAP",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des données (en cache pour la performance)
@st.cache_data
def load_data():
    """
    Chargement de base_model.csv (train, durée > 0) et base_test.csv (dossiers test, durée = 0).
    """
    
    base = pd.read_csv(ROOT_DIR / 'data' / 'processed' / 'base_model.csv')
    base["date.ouverture"] = pd.to_datetime(base["date.ouverture"])
    base["heure"] = pd.to_datetime(base["heure.ouverture"], format='%H:%M:%S').dt.hour
    
    base_test = pd.read_csv(ROOT_DIR / 'data' / 'processed' / 'base_test.csv')
    return base, base_test

@st.cache_data
def load_ml_comparaison():
    """
    Chargement du tableau de comparaison ML/DL 
    """
    
    return pd.read_csv(ROOT_DIR / 'data' / 'models' / 'comparaison_modeles.csv')

base, base_test = load_data()
ml_comparaison = load_ml_comparaison()

# Barre latérale navigation
st.sidebar.markdown("""
                    # Assurance assistance 
                    ### Projet Data - DU BDAP 2025-2026
                    ---
                    """)

page = st.sidebar.radio(
    "Navigation",
    [
        "Accueil",
        "Exploration des données",
        "DataViz - Analyse descriptive",
        "Économétrie",
        "Machine Learning"
    ],
    index=0   
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
 **Données :** 
 - Dossiers d'assistance 2021-2022
 - Déplacements avec survenance en France  
""")
st.sidebar.markdown(f"**Base d'entraînement (durée > 0) :** {len(base):,}")
st.sidebar.markdown(f"**Base à prédire (durée = 0) :** {len(base_test):,}")

# Routage des pages
if page == "Accueil":
    accueil.rendu_accueil(base)
elif page == "Exploration des données":
    exploration.rendu_exploration(base)
elif page == "DataViz - Analyse descriptive":
    dataviz.render_dataviz(base)
elif page == "Économétrie":
    econometrie.render_econometrie(base, base_test)
elif page == "Machine Learning":
    machine_learning.rendu_machine_learning(base, ml_comparaison)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; padding: 20px;">
    <p>Projet Data - DU BDAP 2025-2026 | Université de Montpellier</p>
    <p>Données : dossiers d'assistance automobile 2021-2022</p> 
</div>   
""", unsafe_allow_html=True)   
    
