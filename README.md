# Projet Data - Assurance Assistance Automobile

## DU Big Data Data Sciences et Analyse des Risques sous Python 2025-2026 | Université de Montpellier

> ⚠️ **Données confidentielles** : les données brutes et les bases traitées (`data/raw/`, `data/processed/`) ont été fournies dans un cadre pédagogique et **ne sont pas publiées** dans ce dépôt. Les notebooks sont publiés avec leurs sorties (graphiques, tableaux, résultats des modèles), ce qui permet de suivre l'ensemble de la démarche. Le modèle Random Forest (46 Mo) n'est pas inclus pour alléger le dépôt.

---

## Contexte

Ce projet s'inscrit dans le cadre du **Diplôme Universitaire Big Data Data Sciences et Analyse des Risques sous Python 2025-2026** de l'Université de Montpellier.
En tant que data scientists juniors dans une entreprise d'assurance assistance, nous analysons les **dossiers d'assistance automobile** ouverts entre 2021 et 2022 sur le territoire français.

L'objectif est de mener une étude complète de data science : du traitement des données brutes jusqu'à la mise en production d'une application interactive, en passant par l'analyse statistique, la modélisation économétrique et le machine learning.

---

## Objectif métier

L'objectif métier du projet est d'identifier les facteurs qui influencent la durée de traitement des dossiers d'assistance automobile, puis de construire un outil capable d'estimer cette durée à partir des caractéristiques du dossier.

---

## Organisation du projet

```text
projet-data-assurance-auto/
|
|-- data/                              # Données du projet
|   |-- raw/                           # Données brutes non modifiées
|   |   |-- dossier.csv                # Table des dossiers d'assistance (~101 000 lignes)
|   |   |-- temps.csv                  # Table des temps passés par matricule (~431 000 lignes)
|   |   |-- ressources.csv             # Table des ressources/agents (~389 000 lignes)
|   |-- processed/                     # Données nettoyées et enrichies
|   |   |-- base_model.csv             # Base d'entraînement (durée > 0) pour la modélisation
|   |   |-- base_test.csv              # Base à prédire (durée == 0)
|   |   |-- anomalies_retraitements.csv # Tableau des anomalies et retraitements
|   |-- models/                        # Modèles sauvegardés
|   |   |-- ml_*.pkl                   # Modèles ML (pipelines scikit-learn), dont XGBoost optimisé
|   |   |-- dl_*.keras                 # Modèles DL (TensorFlow)
|   |   |-- preprocessor.joblib        # Pipeline de prétraitement
|   |   |-- ml_metadata.joblib         # Métadonnées (variables explicatives, meilleur modèle)
|   |   |-- comparaison_modeles.csv    # Tableau comparatif ML vs DL
|   |-- Glossaire.pdf                  # Dictionnaire des données
|
|-- traitement_de_donnees/             # Notebooks de traitement et nettoyage
|   |-- traitement_donnees.ipynb       # Traitement principal
|
|-- econometrie/                       # Notebooks d'économétrie et d'analyses statistiques
|   |-- econometrie.ipynb              # Analyse descriptive, AFDM et GLM
|
|-- machine_learning/                  # Notebooks de Machine Learning et Deep Learning
|   |-- machine_learning.ipynb         # Modèles ML, optimisation XGBoost et architectures DL
|
|-- mise_en_production/                # Application Streamlit
|   |-- app.py                         # Application web interactive
|   |-- app_pages/                     # Pages de l'application
|       |-- accueil.py
|       |-- exploration.py
|       |-- dataviz.py
|       |-- econometrie.py
|       |-- machine_learning.py
|
|-- requirements.txt                   # Dépendances Python du projet
|-- README.md                          # Ce fichier
|-- LICENSE                            # Licence du projet
```

---

## Description des données

### Table `dossier.csv` - Dossiers d'assistance

Ensemble des dossiers d'assistance ouverts de 2021 à 2022 pour des déplacements avec survenance en France. **1 ligne = 1 dossier**.

| Variable | Description |
|----------|-------------|
| `Numero_dossier_ID` | Identifiant unique du dossier (anonymisé) |
| `Client` | Client qui délègue son activité d'assistance (anonymisé) |
| `Formule` | Formule d'assistance liée au contrat d'assurance |
| `date.ouverture` | Date d'ouverture du dossier (AAAA/MM/JJ) |
| `heure.ouverture` | Heure d'ouverture (HH:MM:SS) |
| `Matricule.de.traitement` | Personne ayant créé le dossier (171 = création automatique) |
| `Cause.intervention` | Cause de la demande (panne, accident, vol, etc.) |
| `date.de.survenance` | Date de l'incident |
| `Type.d.energie` | Énergie du véhicule (Essence, Diesel, Électrique, etc.) |
| `Outil.d.assistance` | Outil utilisé (MCS ou Higgins) |
| `Assistance.ou.Administratif` | Type de dossier (assistance directe ou administratif) |
| `TOP.D.R` | Service dépannage/remorquage (0/1) |
| `TOP.VR` | Service véhicule de remplacement (0/1) |
| `TOP.Rappat.valide` | Service rapatriement d'un valide (0/1) |
| `TOP.Poursuite` | Service poursuite de voyage (0/1) |
| `TOP.Recup` | Service récupération de véhicule (0/1) |
| `TOP.Autres.Garanties` | Autres services d'assistance (0/1) |

### Table `temps.csv` - Temps par dossier

Temps passé par chaque agent (matricule) sur chaque dossier. Le matricule 171 (système) n'apparaît pas dans cette table.

| Variable | Description |
|----------|-------------|
| `Numero.dossier` | Identifiant du dossier |
| `Matricule` | Agent ayant travaillé sur le dossier |
| `Date.debut.traitement` | Date de début de l'intervention |
| `heure.debut.traitement` | Heure de début |
| `duree.corrigee` | Durée en secondes passée sur le dossier |

### Table `ressources.csv` - Ressources

Profil des agents (chargés d'accueil et chargés d'assistance) pour chaque jour de présence.

| Variable | Description |
|----------|-------------|
| `Matricule` | Identifiant de l'agent |
| `Date.presence` | Jour de présence (JJ/MM/AAAA) |
| `Lieu.travail` | TELE (télétravail) ou SITE (entreprise) |
| `Population` | CAC (chargé d'accueil) ou CAS (chargé d'assistance) |
| `Site` | Lieu de rattachement (A ou B) |
| `Type.de.contrat` | CDI, CDD ou CDS (saisonnier) |
| `Duree.travail` | Durée de travail en heures (journée = 7,33 h) |
| `Temps.travail` | Temps de présence (100 = temps plein) |
| `Experience` | Jours de présence cumulés |

---

## Méthodologie

### 1. Traitement des données (`traitement_donnees.ipynb`)

Traitement séquentiel : **Dossier** -> **Temps** -> **Ressources**, avec 7 étapes par table :

1. Lecture et visualisation
2. Vérification et correction des types
3. Retraitement des variables mal encodées
4. Vérification de la délimitation temporelle (2021-2022)
5. Détection et retraitement des doublons
6. Détection et imputation des valeurs manquantes
7. Détection et correction des incohérences

**Sorties** :
- `base_model.csv` : base d'entraînement (durée > 0, environ 89 % des dossiers)
- `base_test.csv` : base à prédire (durée == 0, environ 11 % des dossiers)
- `anomalies_retraitements.csv` : tableau de traçabilité des anomalies et retraitements

### 2. Analyse statistique et économétrie (`econometrie.ipynb`)

**Partie 1 - Analyse descriptive**
- Analyse univariée : distributions des variables numériques et catégorielles
- Analyse bivariée : tests statistiques (Kruskal-Wallis, Mann-Whitney, Spearman, Tukey HSD)
- Analyse temporelle : évolution mensuelle, distribution par heure et par jour
- Matrice de corrélation de Spearman

**Partie 2 - Analyse multivariée**
- AFDM (Analyse Factorielle des Données Mixtes) avec 5 composantes (bibliothèque `prince`)

**Partie 3 - Modélisation économétrique (GLM)**
- GLM Gamma (lien log) avec backward elimination (p < 0,05)
- GLM Inverse Gaussien (lien log) avec backward elimination (p < 0,05)
- Comparaison par AIC et métriques de validation (MAE, RMSE, MAPE)
- Analyse VIF (multicolinéarité) sur les variables retenues
- Diagnostic des résidus (QQ-plot, résidus vs valeurs ajustées)

**Vidéo de présentation** :  
[Voir la vidéo sur le traitement des données et l'économétrie](https://drive.google.com/file/d/1MC2cIQLi_N2mwhNnUT7DROEDcsve4NeF/view?usp=sharing)

### 3. Machine Learning et Deep Learning (`machine_learning.ipynb`)

**Modèles ML classiques** (scikit-learn) :

| Modèle | Type | Description |
|--------|------|-------------|
| Régression linéaire | Baseline | Référence de performance |
| Ridge (L2) | Régularisé | Pénalité L2 contre la multicolinéarité |
| Lasso (L1) | Régularisé | Sélection automatique de variables |
| Random Forest | Ensemble (Bagging) | Combinaison de 200 arbres |
| Gradient Boosting | Ensemble (Boosting) | Arbres séquentiels correcteurs |
| XGBoost | Ensemble (Boosting) | État de l'art du boosting |
| XGBoost optimisé | Ensemble (Boosting) | Recherche ciblée d'hyperparamètres |

**4 architectures Deep Learning** (TensorFlow / Keras) :

| Modèle | Architecture | Particularité |
|--------|-------------|---------------|
| MLP Simple | 2 couches (64, 32) | Baseline DL |
| MLP Deep | 4 couches + Dropout + L2 | Régularisation contre l'overfitting |
| Wide & Deep | Branche linéaire + profonde | Capture des effets linéaires et non linéaires |
| Residual MLP | Connexions résiduelles | Apprentissage plus stable |

**Démarche** :
- Séparation 80 % entraînement / 20 % test sur `base_model.csv`
- Transformation de la cible : `log(1 + durée)` pour stabiliser la variance
- Métriques en secondes : MAE, RMSE, MAPE
- Optimisation ciblée de XGBoost par recherche d'hyperparamètres
- Courbes de loss (train vs test) pour chaque modèle DL
- Importance des variables par permutation
- Sauvegarde de tous les modèles dans `data/models/`

### 4. Mise en production (Streamlit)

Application Streamlit avec plusieurs pages :

| Page | Contenu |
|------|---------|
| Accueil | Chiffres clés, contexte, navigation |
| Exploration des données | Filtres interactifs et statistiques sur un sous-ensemble |
| DataViz | Exploration visuelle des données |
| Économétrie | Analyses univariées, bivariées, résultats GLM |
| Machine Learning | Comparaison ML vs DL, visualisations et prédiction interactive |

---

## Installation et exécution

### Prérequis

- Python 3.9+
- pip (gestionnaire de paquets Python)

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/Djawadou/projet-data-assurance-auto.git
cd projet-data-assurance-auto

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution des notebooks

```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir les notebooks dans l'ordre :
# 1. traitement_de_donnees/traitement_donnees.ipynb
# 2. econometrie/econometrie.ipynb
# 3. machine_learning/machine_learning.ipynb
```

### Reproduction des fichiers générés

Les fichiers présents dans `data/processed/` et `data/models/` sont générés à partir des données brutes situées dans `data/raw/`. Ces données n'étant pas publiées, l'exécution complète des notebooks et de l'application nécessite d'y avoir accès.

Pour reproduire l'ensemble du pipeline depuis les données brutes, exécuter les notebooks dans l'ordre suivant :

1. `traitement_de_donnees/traitement_donnees.ipynb`  
   Génère les bases nettoyées et enrichies dans `data/processed/`, notamment `base_model.csv`, `base_test.csv` et `anomalies_retraitements.csv`.

2. `econometrie/econometrie.ipynb`  
   Réalise les analyses statistiques, l'AFDM et les modèles économétriques.

3. `machine_learning/machine_learning.ipynb`  
   Entraîne les modèles de machine learning et deep learning, puis sauvegarde les modèles et métriques dans `data/models/`.

Les fichiers générés principaux sont :

| Fichier | Description |
|---------|-------------|
| `data/processed/base_model.csv` | Base d'entraînement avec durée de traitement connue |
| `data/processed/base_test.csv` | Base des dossiers sans durée observée à prédire |
| `data/processed/anomalies_retraitements.csv` | Journal des anomalies détectées et retraitées |
| `data/models/ml_*.pkl` | Modèles de machine learning sauvegardés, dont le XGBoost optimisé |
| `data/models/dl_*.keras` | Modèles de deep learning sauvegardés |
| `data/models/preprocessor.joblib` | Pipeline de prétraitement |
| `data/models/ml_metadata.joblib` | Métadonnées utilisées par l'application Streamlit |
| `data/models/comparaison_modeles.csv` | Comparaison des performances des modèles |


### Exécution de l'application Streamlit

```bash
cd mise_en_production
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

---

## Résultats principaux

### Modélisation prédictive

Le **XGBoost optimisé** obtient le meilleur compromis prédictif sur les métriques MAE et RMSE.
Le **Gradient Boosting** reste toutefois intéressant lorsque l'on privilégie un entraînement plus rapide et une mise en production plus simple.

### Variables les plus influentes sur la durée de traitement

1. **Nombre d'intervenants** : plus il y a d'agents, plus le dossier est complexe et long
2. **Nombre d'interventions** : indicateur de la complexité du dossier
3. **Services demandés** : VR, rapatriement et poursuite de voyage augmentent fortement la durée
4. **Expérience des agents** : les agents plus expérimentés traitent les dossiers plus vite
5. **Télétravail** : légère réduction de la durée en télétravail

### Facteurs clés

- Les **pannes mécaniques** et les **problèmes de clés/carburant/crevaison** sont traités plus rapidement
- Les **accidents** et **incendies** nécessitent des durées de traitement significativement plus longues
- Les **chargés d'assistance (CAS)** traitent des dossiers plus complexes que les **chargés d'accueil (CAC)**
- L'outil **Higgins** est en phase d'adoption croissante depuis 2021

---

## Technologies utilisées

| Catégorie | Outils |
|-----------|--------|
| Langage | Python 3.9+ |
| Manipulation de données | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Statistiques | scipy, statsmodels |
| Machine Learning | scikit-learn, xgboost |
| Deep Learning | TensorFlow / Keras |
| Analyse factorielle | prince |
| Application web | Streamlit |
| Versionnement | Git / GitLab |

---
## Auteurs

Projet réalisé par :

- Dorcasse Yankam
- Djawadou Djobo
- Bénédicte Wantineki


## Licence

Ce projet est réalisé dans un cadre académique (DU BDAP Python 2025-2026, Université de Montpellier).
