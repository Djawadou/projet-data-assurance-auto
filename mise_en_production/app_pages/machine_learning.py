# PAGE MACHINE LEARNING - Comparaison ML/DL + Prédiction

# Import des bibliothèques
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# Définition des chemins
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
COMPARISON_PATH = MODEL_DIR / "comparaison_modeles.csv"
METADATA_PATH = MODEL_DIR / "ml_metadata.joblib"

# Définition des variables
FEATURE_LABELS = {
    "nb_interventions": "Nombre d'interventions",
    "nb_intervenants": "Nombre d'intervenants",
    "prop_tele": "Proportion télétravail",
    "exp_moy": "Expérience moyenne (jours)",
    "Cause.intervention": "Cause d'intervention",
    "Type.d.energie": "Type d'énergie",
    "Outil.d.assistance": "Outil d'assistance",
    "Assistance.ou.Administratif": "Assistance ou Administratif",
    "pop_mode": "Population (agent)",
    "site_mode": "Site de travail",
    "type_contrat_mode": "Type de contrat",
    "TOP.D.R": "Dépannage / Remorquage",
    "TOP.VR": "Véhicule de remplacement",
    "TOP.Rappat.valide": "Rapatriement",
    "TOP.Poursuite": "Poursuite de voyage",
    "TOP.Recup": "Récupération",
    "TOP.Autres.Garanties": "Autres garanties",
}

MODEL_DETAILS = {
    "Regression Lineaire": ("Linéaire", "Baseline simple"),
    "Régression Linéaire": ("Linéaire", "Baseline simple"),
    "Ridge": ("Linéaire régularisé", "Gère la multicolinéarité"),
    "Lasso": ("Linéaire régularisé", "Sélection de variables"),
    "Random Forest": ("Ensemble (Bagging)", "Robuste, peu de réglages"),
    "Gradient Boosting": ("Ensemble (Boosting)", "Très performant"),
    "XGBoost": ("Ensemble (Boosting)", "État de l'art"),
}

DL_DETAILS = {
    "MLP Simple": ("2 couches (64, 32)", "Baseline DL"),
    "MLP Deep": ("4 couches + Dropout + L2", "Régularisation contre l'overfitting"),
    "Wide & Deep": ("Branche linéaire + profonde", "Effets linéaires et non linéaires"),
    "Residual MLP": ("Skip connections", "Apprentissage plus stable"),
}

PREDICTION_THRESHOLDS_MIN = [
    (10, "success", "Ce dossier devrait être traité **rapidement** (< 10 min)."),
    (30, "info", "Durée de traitement **modérée** (10-30 min)."),
    (60, "warning", "Durée de traitement **élevée** (30-60 min). Dossier complexe."),
]


# Fonction de rendu de la page Machine Learning
def rendu_machine_learning(base, ml_comparison):
    """Affiche la page ML avec comparaison des modèles et prédiction interactive."""

    metadata = _load_metadata()
    results_df = _load_results(ml_comparison)
    ml_count = _count_models(metadata, results_df, "ML")
    dl_count = _count_models(metadata, results_df, "DL")
    total_count = ml_count + dl_count

    st.markdown("# Machine Learning & Deep Learning")
    st.markdown(f"""
    Nous avons testé **{total_count} modèles** pour prédire la durée de traitement :
    **{ml_count} modèles ML classiques** (scikit-learn) et **{dl_count} architectures Deep Learning** (TensorFlow).

    - Séparation : 80 % entraînement / 20 % test sur `base_model.csv`
    - Cible : `log(1 + durée)` (inversion pour les métriques en secondes)
    - Métriques : **MAE**, **RMSE**, **MAPE**
    """)

    # Tableau des modèles
    st.markdown("### Modèles testés")

    _render_tested_models(metadata, results_df, ml_count, dl_count)

    st.markdown("---")

    # Résultats de la comparaison
    st.markdown("### Comparaison des performances")

    if results_df.empty:
        st.info("Aucun résultat de comparaison à afficher.")
    elif 'RMSE Test (s)' in results_df.columns:
        display_cols = [c for c in ['Modele', 'MAE Train (s)', 'MAE Test (s)',
                                    'RMSE Train (s)', 'RMSE Test (s)', 'MAPE Test (%)',
                                    'Type', 'Temps (s)']
                        if c in results_df.columns]
        results_sorted = results_df[display_cols].sort_values('RMSE Test (s)')
        st.dataframe(results_sorted.round(1), width="stretch", hide_index=True)

        best = results_sorted.iloc[0]
        st.success(f"**Meilleur modèle : {best['Modele']}** | "
                   f"RMSE = {best.get('RMSE Test (s)', 'N/A')}s | "
                   f"MAE = {best.get('MAE Test (s)', 'N/A')}s | "
                   f"MAPE = {best.get('MAPE Test (%)', 'N/A')}%")
    else:
        st.dataframe(results_df.round(1), width="stretch", hide_index=True)

    st.markdown("---")

    # Graphiques comparatifs
    _render_charts(results_df)

    st.markdown("---")

    # Synthèse
    _render_summary(results_df, metadata)

    st.markdown("---")

    # Prédiction interactive
    _render_prediction(base, metadata)


# ============================================================
# Fonctions privées

def _load_metadata():
    """Charge les métadonnées du notebook ML si elles sont disponibles."""
    if METADATA_PATH.exists():
        return joblib.load(METADATA_PATH)
    return {}


def _load_results(ml_comparison):
    """Charge les résultats de comparaison depuis le CSV généré par le notebook."""
    if ml_comparison is not None:
        results_df = ml_comparison.copy()
    elif COMPARISON_PATH.exists():
        results_df = pd.read_csv(COMPARISON_PATH)
    else:
        st.warning("Le fichier `comparaison_modeles.csv` n'a pas été trouvé. "
                   "Exécutez d'abord le notebook `machine_learning.ipynb`.")
        return pd.DataFrame()

    if "Modele" not in results_df.columns and "Modèle" in results_df.columns:
        results_df = results_df.rename(columns={"Modèle": "Modele"})

    if "Modele" not in results_df.columns:
        first_col = results_df.columns[0]
        results_df = results_df.rename(columns={first_col: "Modele"})

    return results_df


def _count_models(metadata, results_df, model_type):
    """Compte les modèles depuis les métadonnées, puis depuis le CSV si besoin."""
    metadata_key = "models_ml" if model_type == "ML" else "models_dl"
    if metadata.get(metadata_key):
        return len(metadata[metadata_key])

    if "Type" in results_df.columns:
        return int((results_df["Type"] == model_type).sum())

    return 0


def _render_tested_models(metadata, results_df, ml_count, dl_count):
    """Affiche les modèles testés à partir des métadonnées et du CSV."""
    tab_ml, tab_dl = st.tabs([
        f"Machine Learning ({ml_count} modèles)",
        f"Deep Learning ({dl_count} modèles)",
    ])

    ml_models = metadata.get("models_ml", _models_from_results(results_df, "ML"))
    dl_models = metadata.get("models_dl", _models_from_results(results_df, "DL"))

    with tab_ml:
        ml_info = pd.DataFrame([
            {
                "Modele": model,
                "Type": _model_detail(model, MODEL_DETAILS, 0, "ML"),
                "Avantage": _model_detail(model, MODEL_DETAILS, 1, "Modèle entraîné"),
            }
            for model in ml_models
        ])
        st.dataframe(ml_info, width="stretch", hide_index=True)

    with tab_dl:
        dl_info = pd.DataFrame([
            {
                "Modele": model,
                "Architecture": _model_detail(model, DL_DETAILS, 0, "Architecture DL"),
                "Particularité": _model_detail(model, DL_DETAILS, 1, "Modèle entraîné"),
            }
            for model in dl_models
        ])
        st.dataframe(dl_info, width="stretch", hide_index=True)


def _models_from_results(results_df, model_type):
    """Récupère les noms des modèles depuis le CSV de comparaison."""
    if {"Modele", "Type"}.issubset(results_df.columns):
        return results_df.loc[results_df["Type"] == model_type, "Modele"].tolist()
    return []


def _model_detail(model_name, details, item_index, default):
    """Retourne une description de modèle en tolérant les suffixes de paramètres."""
    for key, values in details.items():
        if key.lower() in model_name.lower():
            return values[item_index]
    return default


def _render_charts(results_df):
    """Affiche les graphiques comparatifs RMSE et MAE."""
    st.markdown("### Visualisation comparative")

    col1, col2 = st.columns(2)

    with col1:
        if {'Modele', 'RMSE Test (s)'}.issubset(results_df.columns):
            fig = px.bar(
                results_df.sort_values('RMSE Test (s)'), x='Modele', y='RMSE Test (s)',
                color='Type' if 'Type' in results_df.columns else 'RMSE Test (s)',
                color_discrete_map={'ML': '#3498db', 'DL': '#e74c3c'},
                title='RMSE Test (plus bas = meilleur)'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)
            st.plotly_chart(fig, width="stretch")

    with col2:
        if {'Modele', 'MAE Test (s)'}.issubset(results_df.columns):
            fig = px.bar(
                results_df.sort_values('MAE Test (s)'), x='Modele', y='MAE Test (s)',
                color='Type' if 'Type' in results_df.columns else 'MAE Test (s)',
                color_discrete_map={'ML': '#3498db', 'DL': '#e74c3c'},
                title='MAE Test (plus bas = meilleur)'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=True)
            st.plotly_chart(fig, width="stretch")


def _render_summary(results_df, metadata):
    """Affiche une synthèse calculée depuis les résultats disponibles."""
    st.markdown("### Synthèse")

    if results_df.empty or "RMSE Test (s)" not in results_df.columns:
        st.info("La synthèse sera disponible quand les résultats de comparaison seront chargés.")
        return

    best = results_df.sort_values("RMSE Test (s)").iloc[0]
    best_model = best["Modele"]
    best_type = best.get("Type", metadata.get("best_model_type", "N/A"))
    best_rmse = best.get("RMSE Test (s)", np.nan)
    best_mae = best.get("MAE Test (s)", np.nan)

    st.markdown(f"""
    - Le meilleur modèle chargé est **{best_model}** ({best_type}), avec un RMSE test de **{best_rmse:.1f} s**.
    - Son MAE test est de **{best_mae:.1f} s**.
    - Les variables de prédiction sont lues depuis `ml_metadata.joblib` pour rester synchronisées avec le notebook.
    - Les fichiers de modèles disponibles sont détectés automatiquement dans `data/models/`.
    """)


def _render_prediction(base, metadata):
    """Affiche le widget de prédiction interactive."""

    st.markdown("### Prédiction interactive de la durée")
    st.markdown("""
    Renseignez les caractéristiques d'un dossier ci-dessous pour obtenir une **estimation
    de la durée de traitement** grâce au meilleur modèle entraîné.
    """)

    feat_num = metadata.get('FEAT_NUM', [])
    feat_cat = metadata.get('FEAT_CAT', [])
    feat_bin = metadata.get('FEAT_BIN', [])
    all_features = metadata.get('ALL_FEATURES', feat_num + feat_cat + feat_bin)

    if not all_features:
        st.error("Les métadonnées des variables sont introuvables (`ml_metadata.joblib`).")
        return

    # Formulaire
    st.markdown("#### Caractéristiques du dossier")

    input_values = {}
    cat_col1, cat_col2, num_col = st.columns(3)

    for idx, feature in enumerate(feat_cat):
        target_col = cat_col1 if idx % 2 == 0 else cat_col2
        with target_col:
            input_values[feature] = _render_categorical_input(base, feature)

    with num_col:
        for feature in feat_num:
            input_values[feature] = _render_numeric_input(base, feature)

    # Services d'assistance
    st.markdown("#### Services d'assistance demandés")
    service_cols = st.columns(3)
    for idx, feature in enumerate(feat_bin):
        with service_cols[idx % len(service_cols)]:
            input_values[feature] = int(_render_binary_input(base, feature))

    # Sélection du modèle
    st.markdown("#### Choix du modèle")

    available_models = _scan_models(MODEL_DIR)

    if available_models:
        best_default = 0
        best_name = metadata.get('best_model_name', '')
        for idx, n in enumerate(available_models.keys()):
            if best_name.lower().replace(' ', '') in n.lower().replace(' ', ''):
                best_default = idx
                break

        selected_model = st.selectbox("Modèle de prédiction",
                                      list(available_models.keys()),
                                      index=best_default, key="pred_model_choice")
    else:
        selected_model = None
        st.warning("Aucun modèle trouvé dans `data/models/`. "
                   "Exécutez d'abord le notebook `machine_learning.ipynb`.")

    # Bouton de prédiction
    if st.button("Prédire la durée de traitement", type="primary", width="stretch"):

        input_data = pd.DataFrame([{feature: input_values[feature] for feature in all_features}])

        prediction = _predict(input_data, selected_model, available_models,
                              all_features, MODEL_DIR, base)

        # Affichage du résultat
        st.markdown("---")
        st.markdown("#### Résultat de la prédiction")

        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric("Durée estimée (secondes)", f"{prediction:.0f} s")
        with res2:
            st.metric("Durée estimée (minutes)", f"{prediction / 60:.1f} min")
        with res3:
            moyenne = base['duree_corrigee_totale'].mean()
            ecart = ((prediction - moyenne) / moyenne) * 100
            st.metric("Écart vs moyenne", f"{ecart:+.1f} %")

        duree_min = prediction / 60
        _show_prediction_message(duree_min)

        if selected_model and available_models:
            model_type = available_models[selected_model][0]
            st.caption(f"Modèle utilisé : **{selected_model}** | "
                       f"Type : {'ML (scikit-learn)' if model_type == 'ml' else 'DL (TensorFlow)'}")


def _render_categorical_input(base, feature):
    """Crée un selectbox pour une variable catégorielle."""
    label = FEATURE_LABELS.get(feature, _clean_label(feature))
    values = sorted(base[feature].dropna().unique())
    return st.selectbox(label, values, key=f"pred_{feature}")


def _render_numeric_input(base, feature):
    """Crée un input numérique avec bornes déduites de la base."""
    label = FEATURE_LABELS.get(feature, _clean_label(feature))
    series = pd.to_numeric(base[feature], errors="coerce").dropna()

    if series.empty:
        return st.number_input(label, value=0.0, key=f"pred_{feature}")

    min_value = float(series.min())
    max_value = float(series.max())
    default = float(series.median())

    if feature == "prop_tele" or 0 <= min_value <= max_value <= 1:
        return st.slider(label, min_value=0.0, max_value=1.0,
                         value=float(np.clip(default, 0, 1)), step=0.05,
                         key=f"pred_{feature}")

    if pd.api.types.is_integer_dtype(series) or feature.startswith("nb_"):
        return st.number_input(label, min_value=int(min_value), max_value=int(max_value),
                               value=int(round(default)), step=1, key=f"pred_{feature}")

    return st.number_input(label, min_value=min_value, max_value=max_value,
                           value=default, step=max((max_value - min_value) / 100, 1.0),
                           key=f"pred_{feature}")


def _render_binary_input(base, feature):
    """Crée une checkbox pour une variable binaire."""
    label = FEATURE_LABELS.get(feature, _clean_label(feature))
    series = pd.to_numeric(base[feature], errors="coerce").dropna()
    default = bool(round(series.mode().iloc[0])) if not series.empty else False
    return st.checkbox(label, value=default, key=f"pred_{feature}")


def _clean_label(feature):
    """Transforme un nom de colonne technique en libellé lisible."""
    return feature.replace(".", " ").replace("_", " ").title()


def _show_prediction_message(duree_min):
    """Affiche un message qualifiant la durée estimée."""
    for threshold, level, message in PREDICTION_THRESHOLDS_MIN:
        if duree_min < threshold:
            getattr(st, level)(message)
            return

        st.error("Durée de traitement **très élevée** (> 1 h). Dossier très complexe.")


def _scan_models(model_dir):
    """Scanne le répertoire des modèles et retourne un dictionnaire {nom: (type, path)}."""
    available = {}
    if not model_dir.exists():
        return available
    for path in sorted(model_dir.iterdir()):
        f = path.name
        if f.endswith('.pkl') and f.startswith('ml_'):
            name = f.replace('ml_', '').replace('.pkl', '').replace('_', ' ')
            available[name] = ('ml', path)
        elif f.endswith('.keras') and f.startswith('dl_'):
            name = f.replace('dl_', '').replace('.keras', '').replace('_', ' ')
            available[name] = ('dl', path)
    return available


def _predict(input_data, selected_model, available_models, all_features, model_dir, base):
    """Exécute la prédiction avec le modèle sélectionné, ou repli sur la moyenne."""
    prediction = None

    if selected_model and available_models:
        model_type, model_path = available_models[selected_model]

        with st.spinner(f"Chargement du modèle {selected_model}..."):
            try:
                if model_type == 'ml':
                    model = joblib.load(model_path)
                    X_pred = input_data[all_features]
                    y_pred_log = model.predict(X_pred)[0]
                    prediction = float(np.expm1(y_pred_log))

                elif model_type == 'dl':
                    preprocessor_path = model_dir / 'preprocessor.joblib'
                    if preprocessor_path.exists():
                        from tensorflow import keras
                        model = keras.models.load_model(model_path)
                        preprocessor = joblib.load(preprocessor_path)
                        X_pred = input_data[all_features]
                        X_pred_p = preprocessor.transform(X_pred)
                        y_pred_log = model.predict(X_pred_p, verbose=0).ravel()[0]
                        prediction = float(np.expm1(y_pred_log))
                    else:
                        st.error("Préprocesseur non trouvé (`preprocessor.joblib`).")
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

    if prediction is None:
        st.warning("Prédiction approximative par moyenne conditionnelle (modèle non disponible).")
        cause = input_data['Cause.intervention'].iloc[0]
        mask = base['Cause.intervention'] == cause
        if mask.sum() > 0:
            prediction = float(base.loc[mask, 'duree_corrigee_totale'].mean())
        else:
            prediction = float(base['duree_corrigee_totale'].mean())

    return max(prediction, 0.0)
