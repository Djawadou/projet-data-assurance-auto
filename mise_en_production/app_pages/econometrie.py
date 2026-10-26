# =============================================================
# PAGE ÉCONOMÉTRIE - ANALYSE STATISTIQUE ET MODÉLISATION GLM
# =============================================================

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, spearmanr, shapiro, levene, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import prince




# ─── Palette & style ───────────────────────────────────────────────────────────
PALETTE = sns.color_palette("Blues_d", 8)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# ─── Constantes variables ──────────────────────────────────────────────────────
NUM_VARS  = ['duree_corrigee_totale', 'nb_interventions', 'nb_intervenants', 'prop_tele', 'exp_moy']
CAT_VARS  = ['Cause.intervention', 'Type.d.energie', 'Outil.d.assistance', 'Assistance.ou.Administratif']
TOP_VARS  = ['TOP.D.R', 'TOP.VR', 'TOP.Rappat.valide', 'TOP.Poursuite', 'TOP.Recup', 'TOP.Autres.Garanties']
TOP_LABELS = ['Dépannage/Remorquage', 'Véhicule de Remplacement', 'Rapatriement', 'Poursuite Voyage', 'Récupération', 'Autres Garanties']

VAR_QUANTI_GLM = ['nb_interventions', 'nb_intervenants', 'exp_moy', 'prop_tele']
VAR_QUALI_GLM  = ['Cause.intervention', 'Type.d.energie', 'Outil.d.assistance',
                  'Assistance.ou.Administratif', 'pop_mode', 'site_mode', 'type_contrat_mode']
VAR_BIN_GLM    = ['TOP.D.R', 'TOP.VR', 'TOP.Rappat.valide', 'TOP.Poursuite', 'TOP.Recup', 'TOP.Autres.Garanties']


def close_fig(fig):
    plt.close(fig)

def fmt_seconds(s: float) -> str:
    """Formatte des secondes en 'HHh MMmin' (ou 'MMmin SSs' si < 1h)."""
    if pd.isna(s):
        return "—"
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h {m:02d}min" if h else f"{m}min {sec:02d}s"


# ─── Cache : modèles GLM ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_glm_models(base_hash: str, _base: pd.DataFrame):
    """
    Ajuste GLM Gamma et Inverse Gaussien avec backward elimination.
    Retourne les résultats, colonnes retenues, métriques et VIF.
    """
    all_vars = VAR_QUANTI_GLM + VAR_QUALI_GLM + VAR_BIN_GLM

    train, valid = train_test_split(_base, test_size=0.2, random_state=42)
    y_train = train['duree_corrigee_totale']
    y_valid = valid['duree_corrigee_totale']

    X_train_raw = pd.get_dummies(train[all_vars], drop_first=True) * 1
    X_valid_raw = pd.get_dummies(valid[all_vars], drop_first=True) * 1
    X_valid_raw = X_valid_raw.reindex(columns=X_train_raw.columns, fill_value=0)

    X_train = sm.add_constant(X_train_raw)
    X_valid = sm.add_constant(X_valid_raw)

    def backward_elim(X, y, family, seuil=0.05):
        cols = list(X.columns)
        historique = []
        while True:
            model = sm.GLM(endog=y, exog=X[cols], family=family).fit(disp=False)
            pvals = model.pvalues.drop('const', errors='ignore')
            pmax = pvals.max()
            var_pmax = pvals.idxmax()
            if pmax > seuil:
                historique.append({
                    'Variable retirée': var_pmax,
                    'p-value': round(pmax, 4),
                    'AIC avant': round(model.aic, 1)
                })
                cols.remove(var_pmax)
            else:
                break
        model_final = sm.GLM(endog=y, exog=X[cols], family=family).fit(disp=False)
        historique_df = pd.DataFrame(historique) if historique else pd.DataFrame()
        return model_final, cols, historique_df
    

    # --- GLM Gamma ---
    fam_gamma = sm.families.Gamma(link=sm.families.links.Log())
    full_gamma = sm.GLM(y_train, X_train, family=fam_gamma).fit(disp=False)
    res_gamma, cols_gamma, hist_gamma = backward_elim(X_train, y_train, fam_gamma)

    # --- GLM Inverse Gaussien ---
    fam_ig = sm.families.InverseGaussian(link=sm.families.links.Log())
    full_ig = sm.GLM(y_train, X_train, family=fam_ig).fit(disp=False)
    res_ig, cols_ig, hist_ig = backward_elim(X_train, y_train, fam_ig)

    # --- Métriques sur validation ---
    def metrics(y_true, y_pred, name, aic, aic_full, n_vars):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        return {'Modèle': name, 'AIC complet': round(aic_full, 0),
                'AIC final': round(aic, 0), 'Nb vars': n_vars,
                'MAE (s)': round(mae, 1), 'RMSE (s)': round(rmse, 1),
                'MAPE (%)': round(mape, 2), 'R²': round(r2, 4)}

    pred_g  = res_gamma.predict(X_valid[cols_gamma])
    pred_ig = res_ig.predict(X_valid[cols_ig])

    comparaison = pd.DataFrame([
        metrics(y_valid, pred_g,  'Gamma (Log)',       res_gamma.aic, full_gamma.aic, len(cols_gamma)),
        metrics(y_valid, pred_ig, 'Inv. Gaussien (Log)', res_ig.aic,  full_ig.aic,   len(cols_ig)),
    ]).sort_values('AIC final').reset_index(drop=True)

    # --- VIF ---
    def compute_vif(cols, label):
        X_vif = X_train[cols].drop(columns='const', errors='ignore')
        vif = pd.DataFrame({
            'Variable': X_vif.columns,
            'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        }).sort_values('VIF', ascending=False).reset_index(drop=True)
        vif['Modèle'] = label
        return vif

    vif_gamma = compute_vif(cols_gamma, 'Gamma')
    vif_ig    = compute_vif(cols_ig,    'Inv. Gaussien')

    # --- Coefficients ---
    def coef_df(result, cols):
        df = pd.DataFrame({
            'Variable': result.params.index,
            'Coef':     result.params.values,
            'exp(Coef)': np.exp(result.params.values),
            'p-value':  result.pvalues.values,
        }).query("Variable != 'const'").reset_index(drop=True)
        return df

    coef_gamma = coef_df(res_gamma, cols_gamma)
    coef_ig    = coef_df(res_ig,    cols_ig)

    return {
        'comparaison': comparaison,
        'vif_gamma': vif_gamma,
        'vif_ig': vif_ig,
        'coef_gamma': coef_gamma,
        'coef_ig': coef_ig,
        'hist_gamma': hist_gamma,
        'hist_ig': hist_ig,
        'n_train': len(train),
        'n_valid': len(valid),
        'n_vars_full': X_train.shape[1],
        'aic_full_gamma': round(full_gamma.aic, 1),
        'aic_full_ig': round(full_ig.aic, 1),
    }


# ─── Sections ─────────────────────────────────────────────────────────────────

def _tab_univarie(base):
    # ── 1. Variables numériques : grille (n_vars × 2) comme dans le notebook ──
    st.subheader("Distributions des variables numériques")
    st.caption("Colonne gauche : distribution brute | Colonne droite : distribution log(1 + x)")

    fig, axes = plt.subplots(len(NUM_VARS), 2, figsize=(14, 4 * len(NUM_VARS)))

    for i, var in enumerate(NUM_VARS):
        data = base[var].dropna()
        median_val = data.median()

        # Colonne gauche — distribution brute
        axes[i, 0].hist(data, bins=min(100, data.nunique()),
                        color='steelblue', edgecolor='white', alpha=0.8)
        axes[i, 0].set_title(f'{var} — Distribution brute', fontsize=11)
        axes[i, 0].set_ylabel('Fréquence')
        axes[i, 0].axvline(median_val, color='red', linestyle='--',
                           label=f'Médiane = {median_val:.1f}')
        axes[i, 0].legend(fontsize=9)

        # Colonne droite — distribution log(1+x)
        axes[i, 1].hist(np.log1p(data), bins=80,
                        color='coral', edgecolor='white', alpha=0.8)
        axes[i, 1].set_title(f'log(1 + {var})', fontsize=11)
        axes[i, 1].set_ylabel('Fréquence')

    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)

    st.info(
        "**Lecture :** La colonne gauche montre la distribution brute de chaque variable — toutes sont "
        "fortement asymétriques à droite (queue longue), ce qui est typique des durées et des comptages. "
        "La colonne droite applique une transformation **log(1+x)** qui rapproche la distribution d'une "
        "loi normale et justifie l'utilisation d'un **lien log** dans les modèles GLM."
    )

    st.divider()

    # ── 2. Variables catégorielles : grille (2 × 2) comme dans le notebook ──
    st.subheader("Distributions des variables catégorielles")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, var in enumerate(CAT_VARS):
        counts = base[var].value_counts()
        counts.plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='white')
        axes[i].set_title(f'Distribution de {var}', fontsize=11)
        axes[i].set_ylabel('Nombre de dossiers')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)

    st.info(
        "**Lecture :** La **cause d'intervention** est très concentrée sur quelques modalités (Panne mécanique, "
        "Clés/Carburant/Crevaison). Le **type d'énergie** est dominé par l'essence et le diesel. "
        "L'**outil d'assistance** et la distinction **Assistance/Administratif** révèlent la structure "
        "opérationnelle des dossiers."
    )

    st.divider()

    # ── 3. Variables TOP : barplot avec valeurs affichées ──
    st.subheader("Taux d'activation des services d'assistance (variables TOP)")

    top_labels = ['Dépannage\nRemorquage', 'Véhicule\nRemplacement',
                        'Rapatriement', 'Poursuite\nVoyage', 'Récupération', 'Autres\nGaranties']
    taux = [base[v].mean() * 100 for v in TOP_VARS]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(top_labels, taux, color='steelblue', edgecolor='white')

    for bar, val in zip(bars, taux):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=11)

    ax.set_ylabel("Taux d'activation (%)")
    ax.set_title("Taux d'Activation des Services d'Assistance", fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)

    st.info(
        "**INTERPRETATION :** Le service **Dépannage/Remorquage** est de loin le plus fréquent, activé dans la "
        "grande majorité des dossiers. Les services **VR** (Véhicule de Remplacement), **Rapatriement** "
        "et **Poursuite de Voyage** sont nettement moins fréquents mais correspondent aux dossiers les "
        "plus complexes et donc les plus longs à traiter."
    )


def _tab_bivarie(base):
    # --- Durée par cause ---
    st.subheader("Durée de traitement par cause d'intervention")
    
    order_cause = (base.groupby('Cause.intervention')['duree_corrigee_totale']
                    .median().sort_values().index.tolist())
    groups = [base.loc[base['Cause.intervention'] == c, 'duree_corrigee_totale'].dropna()
              for c in order_cause]
    # Création de log_duree pour les graphiques (log(1 + x) pour gérer les zéros et les grandes valeurs)
  
    base = base.copy()
    base['log_duree'] = np.log1p(base['duree_corrigee_totale'])

    
    # Boxplot + Violinplot côte à côte comme dans le notebook
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.boxplot(data=base, x='Cause.intervention', y='log_duree',
                hue='Cause.intervention', palette='bright', legend=False, ax=axes[0])
    axes[0].set_title('Boxplot : log(durée) par Cause', fontsize=11)
    axes[0].set_xlabel("Cause d'intervention")
    axes[0].set_ylabel("log_durée)")
    axes[0].tick_params(axis='x', rotation=45)

    sns.violinplot(data=base, x='Cause.intervention', y='log_duree',
                   hue='Cause.intervention', palette='bright', legend=False, ax=axes[1])
    axes[1].set_title('Violinplot : log(durée) par Cause', fontsize=11)
    axes[1].set_xlabel("Cause d'intervention")
    axes[1].set_ylabel("log_durée)")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)

    # Tests statistiques
    with st.expander("Tests statistiques : Normalité (Shapiro), Variances (Levene), Kruskal-Wallis"):
        rows_shapiro = []
        for name, group in base.groupby('Cause.intervention'):
            g = group['duree_corrigee_totale'].dropna()
            stat_sh, p_sh = shapiro(g.sample(min(5000, len(g)), random_state=42))
            rows_shapiro.append({'Cause': name, 'Stat Shapiro': round(stat_sh, 4),
                                 'p-value': f'{p_sh:.3e}',
                                 'Normalité': 'Normale' if p_sh > 0.05 else 'Non normale'})
        st.markdown("**Test de Normalité — Shapiro-Wilk** (H0 : distribution normale)")
        st.dataframe(pd.DataFrame(rows_shapiro), use_container_width=True, hide_index=True)


        stat_lev, p_lev = levene(*groups)
        st.markdown("**Test d'Égalité des Variances — Levene** (H0 : variances égales)")
        c1, c2 = st.columns(2)
        c1.metric("Statistique Levene", f"{stat_lev:.3f}")
        c2.metric("p-value", f"{p_lev:.3e}",
                  delta="Variances inégales" if p_lev < 0.05 else "Variances égales")

        stat_kw, pval_kw = kruskal(*groups)
        st.markdown("**Test de Kruskal-Wallis** (H0 : durée médiane identique dans tous les groupes)")
        st.markdown("H1 : au moins une cause a une durée médiane différente")
        c1, c2 = st.columns(2)
        c1.metric("Statistique H", f"{stat_kw:.3f}")
        c2.metric("p-value", f"{pval_kw:.3e}",
                  delta="Différence significative ✓" if pval_kw < 0.05 else "Pas de différence")
    with st.expander("Test post-hoc Tukey HSD (quels couples de causes diffèrent ?)"):
        tukey = pairwise_tukeyhsd(base['duree_corrigee_totale'].dropna(),
                                  base.loc[base['duree_corrigee_totale'].notna(), 'Cause.intervention'])
        df_tukey = pd.DataFrame(data=tukey._results_table.data[1:],
                                columns=tukey._results_table.data[0])
        df_tukey['reject'] = df_tukey['reject'].map({True: 'Oui', False: 'Non'})
        df_sig = df_tukey[df_tukey['reject'] == 'Oui']
        #df_sig = df_tukey
        st.write(f"{len(df_sig)} paires  ")
        st.dataframe(df_sig.reset_index(drop=True), use_container_width=True)

    st.info(
        "**INTERPRETATION:** La p-value < 0.05 du test de Kruskal-Wallis confirme que la durée de traitement "
        "diffère **significativement** selon la cause d'intervention. Shapiro et Levene confirment "
        "que les données ne sont pas normales et que les variances sont inégales — ce qui justifie "
        "l'utilisation de tests non-paramétriques."
        "- Le test de Tukey identifie les couples de causes qui different significativement."
    )

    st.divider()
    st.markdown("**Impact du type d'energie**")

    # --- Durée par type d'énergie ---
    st.markdown("**Violinplot pour visualiser la distribution de log(durée) selon le type d'énergie**")
    st.subheader("log(durée) par type d'énergie")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=base, x='Type.d.energie', y='log_duree',
                   hue='Type.d.energie', palette='bright', legend=False, ax=ax)
    ax.set_title("Distribution de log(durée) par Type d'Énergie", fontsize=12, fontweight='bold')
    ax.set_xlabel("Type d'énergie")
    ax.set_ylabel("log_durée")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)


# Test de Kruskal-Wallis
    st.markdown("**Test de Kruskal-Wallis**")
    groups_e = [g['duree_corrigee_totale'].dropna() for _, g in base.groupby('Type.d.energie')]
    stat_e, p_e = kruskal(*groups_e)
    c1, c2 = st.columns(2)
    c1.metric("Kruskal-Wallis H", f"{stat_e:.3f}")
    c2.metric("p-value", f"{p_e:.3e}",
              delta="Différence significative ✓" if p_e < 0.05 else "Pas de différence")

    with st.expander("Test post-hoc Tukey HSD — Type d'énergie"):
        tukey_e = pairwise_tukeyhsd(base['duree_corrigee_totale'].dropna(),
                                    base.loc[base['duree_corrigee_totale'].notna(), 'Type.d.energie'])
        df_te = pd.DataFrame(data=tukey_e._results_table.data[1:],
                             columns=tukey_e._results_table.data[0])
        df_te['reject'] = df_te['reject'].map({True: 'Oui', False: 'Non'})
        st.dataframe(df_te.reset_index(drop=True), use_container_width=True)

    st.info(
        "**Interprétation :** Le type d'énergie impacte significativement la durée (p < 0.001). "
        "Les véhicules **électriques et hybrides** tendent à avoir des durées de traitement plus longues, "
        "probablement en raison de la complexité technique plus élevée des interventions sur ces motorisations."
    )
     
    st.divider()
    st.markdown("**4.3. Impact des garanties (variables TOP)**")

    # --- Durée par services TOP : boxplot log_duree (2×3) ---
    st.subheader("log(durée) selon l'activation des services d'assistance")
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    axes = axes.flatten()
    mw_rows = []
    for i, var in enumerate(TOP_VARS):
        sns.boxplot(data=base, x=var, y='log_duree',
                    hue=var, palette='Set2', legend=False, ax=axes[i])
        axes[i].set_title(f'log(durée) selon {var}', fontsize=9)
        axes[i].set_xlabel(f'{var} (0=Non, 1=Oui)')
        axes[i].set_ylabel("log_durée)")
        g0 = base.loc[base[var] == 0, 'duree_corrigee_totale'].dropna()
        g1 = base.loc[base[var] == 1, 'duree_corrigee_totale'].dropna()
        stat_mw, p_mw = mannwhitneyu(g0, g1)
        mw_rows.append({'Variable': var, 'Stat Mann-Whitney': round(stat_mw, 0),
                            'p-value': f'{p_mw:.3e}',
                            'Conclusion': 'Différence significative' if p_mw < 0.05 else 'Pas de différence'})
    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)
    with st.expander("Tests de Mann-Whitney U — résultats par service"):
        st.markdown("**H0 : pas de différence de durée entre les 2 groupes (activé / non activé)**")
        st.dataframe(pd.DataFrame(mw_rows), use_container_width=True, hide_index=True)

    st.info(
        "**Interprétation :** Tous les tests sont significatifs (p < 0.05). "
        "Le service **D/R (Dépannage/Remorquage) réduit** la durée car il correspond aux dossiers simples. "
        "À l'inverse, les services **VR, Rapatriement et Poursuite de Voyage augmentent** fortement "
        "la durée : ce sont des prises en charge complexes, multi-intervenants."
    )

 
    st.divider()
    st.markdown("**Impact des variables numeriques**")
    
    # --- Corrélations : X vs log_duree ---
    st.subheader("Corrélations avec log(durée) (Spearman)")
    expl_vars   = ['nb_intervenants', 'nb_interventions', 'prop_tele', 'exp_moy']
    expl_labels = ['Nb intervenants', 'Nb interventions', 'Prop. télétravail', 'Expérience moy.']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (var, lbl) in enumerate(zip(expl_vars, expl_labels)):
        sub = base[[var, 'log_duree', 'duree_corrigee_totale']].dropna()
        rho, pv = spearmanr(sub[var], sub['duree_corrigee_totale'])
        axes[i].scatter(sub[var], sub['log_duree'], alpha=0.05, s=3, color='steelblue')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('log(1 + durée)')
        axes[i].set_title(f'{var} vs durée\nSpearman rho={rho:.3f}, p={pv:.2e}', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)

    st.info(
        "**Interprétation :** Le **nombre d'intervenants** (ρ = +0.786) et le **nombre d'interventions** "
        "(ρ = +0.788) sont les prédicteurs numériques les plus fortement corrélés à la durée : "
        "plus il y a d'intervenants/interventions, plus le dossier est long. "
        "La **proportion de télétravail** a une corrélation quasi-nulle (ρ = -0.019). "
        "L'**expérience moyenne** a une corrélation faiblement positive (ρ = +0.079)."
    )
 # --- Analyse temporelle : boxplot log_duree/mois + volume/heure ---
    st.subheader("Analyse temporelle de la durée")
    base['date.ouverture'] = pd.to_datetime(base['date.ouverture'])
    base['mois'] = base['date.ouverture'].dt.month
    base['heure'] = pd.to_datetime(base['heure.ouverture'], format='%H:%M:%S', errors='coerce').dt.hour
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=base, x='mois', y='log_duree',
                hue='mois', palette='coolwarm', legend=False, ax=axes[0])
    axes[0].set_title('log(durée) par Mois', fontweight='bold')
    axes[0].set_xlabel('Mois (1=Jan, 12=Déc)')
    axes[0].set_ylabel('log(1 + durée)')
    
    # --- Volume par heure ---
    heure_counts = base['heure'].value_counts().sort_index()
    axes[1].bar(heure_counts.index, heure_counts.values, color='coral', edgecolor='white')
    axes[1].set_title("Nombre de Dossiers par Heure d'Ouverture", fontweight='bold')
    axes[1].set_xlabel('Heure (0-23)')
    axes[1].set_ylabel('Nombre de dossiers')

    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)

    st.info(
        "**Interprétation :** La durée **augmente entre juin et août** (période estivale avec plus de pannes "
        "sur route), et **diminue en janvier**. Le graphique de droite montre que les dossiers sont "
        "généralement ouverts entre **8h et 17h**, avec un pic en milieu de matinée."
    )

    st.divider()
    st.markdown("**Matrice de Corrélation**")
    st.markdown("On utilise Spearman car les variables ne sont pas normales")
    st.subheader("Matrice de corrélation de Spearman")
    corr_vars = ['duree_corrigee_totale', 'nb_interventions', 'nb_intervenants',
                 'prop_tele', 'exp_moy'] + TOP_VARS
    corr_matrix = base[corr_vars].corr(method='spearman')

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={'size': 8})
    ax.set_title('Matrice de Corrélation de Spearman', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)

    st.info(
        "**Interprétation :** La matrice confirme que **nb_intervenants** et **nb_interventions** "
        "sont les variables les plus corrélées à la durée (rouge foncé). Ces deux variables sont aussi "
        "corrélées entre elles (ce que l'analyse VIF vérifiera). Les variables TOP montrent des "
        "corrélations de signe opposé selon qu'elles correspondent à des dossiers simples (D/R, négatif) "
        "ou complexes (VR, Rapatriement, positif)."
    )
    

def _tab_afdm(base):
    st.subheader("Analyse Factorielle des Données Mixtes (AFDM / FAMD)")
    st.markdown("""
    L'AFDM réduit la dimensionnalité sur des données **mixtes** (quantitatives + qualitatives).
    Elle synthétise l'information en composantes factorielles non corrélées.
    """)

    @st.cache_data(show_spinner=False)
    def _run_famd(_base):
        target = 'duree_corrigee_totale'
        var_quanti = ['nb_interventions', 'nb_intervenants', 'exp_moy', 'prop_tele']
        var_quali  = ['Client', 'Formule', 'Cause.intervention', 'Type.d.energie',
                      'Outil.d.assistance', 'Assistance.ou.Administratif',
                      'pop_mode', 'site_mode', 'type_contrat_mode']
        var_top    = ['TOP.D.R', 'TOP.VR', 'TOP.Rappat.valide', 'TOP.Poursuite', 'TOP.Recup', 'TOP.Autres.Garanties']

        features_afdm = _base[var_quanti + var_quali + var_top + [target]].dropna()
        for v in var_top:
            features_afdm[v] = features_afdm[v].astype(str)
        for v in var_quali:
            features_afdm[v] = features_afdm[v].astype(str)

        X_famd = features_afdm.drop(columns=target)
        y_famd = features_afdm[target]

        famd = prince.FAMD(n_components=5, random_state=42)
        famd = famd.fit(X_famd)
        coords = famd.row_coordinates(X_famd)
        coords.columns = ['Dim1', 'Dim2', 'Dim3', 'Dim4', 'Dim5']

        eig = famd.eigenvalues_
        explained = eig / eig.sum()
        var_contrib = famd.column_contributions_
        return coords, explained, y_famd, var_contrib

    with st.spinner("Calcul de l'AFDM en cours…"):
        coords, explained, y_famd, var_contrib = _run_famd(base)

    cumul = np.cumsum(explained)

    # VISUALISATION DE L'AFDM
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 1. Eboulis des valeurs propres ---
    axes[0].bar(range(1, len(explained)+1), explained * 100, color='steelblue', edgecolor='white')
    axes[0].plot(range(1, len(explained)+1), cumul * 100, 'ro-', label='Cumul')
    axes[0].set_title("Éboulis des valeurs propres", fontweight='bold')
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Variance Expliquée (%)")
    axes[0].legend()

    # --- 2. Projection des dossiers (Dim1 x Dim2) ---
    axes[1].scatter(coords['Dim1'], coords['Dim2'], alpha=0.05, s=3, color='steelblue')
    axes[1].axhline(0, color='grey', linewidth=0.5)
    axes[1].axvline(0, color='grey', linewidth=0.5)
    axes[1].set_xlabel(f'Dim1 ({explained[0]*100:.1f}%)')
    axes[1].set_ylabel(f'Dim2 ({explained[1]*100:.1f}%)')
    axes[1].set_title("Projection des Dossiers", fontweight='bold')

# --- 3. Contribution des variables ---
    sns.heatmap(var_contrib, cmap='coolwarm', ax=axes[2])
    axes[2].set_title('Contribution des Variables', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)
    close_fig(fig)

    st.info(
        "**Interprétation des graphiques AFDM :**\n\n"
        "**1. Éboulis des valeurs propres (gauche)** — Chaque barre représente la part de variance "
        "expliquée par une composante. La courbe rouge cumule l'information retenue. "
        "On retient généralement les composantes jusqu'au coude de la courbe.\n\n"
        "**2. Projection des dossiers (Dim1 vs Dim2) (centre)** — Chaque point est un dossier projeté "
        "sur les deux premières composantes. Un **nuage diffus et homogène** (sans clusters séparés) "
        "indique l'absence de sous-populations naturellement distinctes, ce qui **justifie l'approche GLM** "
        "(modélisation continue) plutôt qu'un clustering.\n\n"
        "**3. Contribution des variables (droite)** — La heatmap montre la contribution de chaque variable "
        "aux dimensions factorielles. Les variables à contribution élevée structurent le plus les données."
    )

    # Variance expliquée
    df_exp = pd.DataFrame({
        'Composante': [f"F{i+1}" for i in range(len(explained))],
        'Variance (%)': (explained * 100).round(2),
        'Cumulée (%)': (cumul * 100).round(2)
    })
    st.dataframe(df_exp, use_container_width=True, hide_index=True)


def _tab_glm(base):
    st.subheader("Modélisation GLM — Gamma et Inverse Gaussien")
    st.markdown("""
    | Modèle | Famille | Lien | Hypothèse variance |
    |--------|---------|------|--------------------|
    | **GLM Gamma** | Gamma | Log | $Var(Y) \\propto \\mu^2$ |
    | **GLM Inv. Gaussien** | Inverse Gaussienne | Log | $Var(Y) \\propto \\mu^3$ |

    **Démarche** : ajustement du modèle complet → **Backward Elimination** (retrait itératif des variables avec p > 5%).
    """)

    with st.spinner("Ajustement des modèles GLM (mise en cache après le premier calcul)…"):
        # On passe un hash simple pour le cache
        base_hash = str(len(base)) + str(base.columns.tolist())
        glm = run_glm_models(base_hash, base)

    st.success("Modèles ajustés avec succès.")

    # --- Contexte train / validation ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Dossiers TRAIN (80%)", f"{glm['n_train']:,}")
    c2.metric("Dossiers VALIDATION (20%)", f"{glm['n_valid']:,}")
    c3.metric("Variables initiales", glm['n_vars_full'])

    st.divider()

    # --- Historique backward elimination ---
    st.subheader("Historique de la Backward Elimination")
    tab_h_g, tab_h_ig = st.tabs(["Gamma — variables retirées", "Inv. Gaussien — variables retirées"])

    def show_historique(hist_df, aic_full, aic_final, n_retire, n_garde, label):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AIC modèle complet",  f"{aic_full:,.1f}")
        c2.metric("AIC modèle final",    f"{aic_final:,.1f}",
                  delta=f"{aic_final - aic_full:+.1f}")
        c3.metric("Variables retirées", n_retire)
        c4.metric("Variables retenues", n_garde)
        if not hist_df.empty:
            st.caption(f"Étapes de retrait — {label}")
            st.dataframe(hist_df, width='stretch', hide_index=True)
        else:
            st.info("Aucune variable retirée : toutes étaient significatives dès le départ.")

    comp_g  = glm['comparaison'][glm['comparaison']['Modèle'] == 'Gamma (Log)'].iloc[0]
    comp_ig = glm['comparaison'][glm['comparaison']['Modèle'] == 'Inv. Gaussien (Log)'].iloc[0]
    with tab_h_g:
        show_historique(glm['hist_gamma'],
                        glm['aic_full_gamma'], comp_g['AIC final'],
                        len(glm['hist_gamma']), comp_g['Nb vars'], 'Gamma')
    with tab_h_ig:
        show_historique(glm['hist_ig'],
                        glm['aic_full_ig'], comp_ig['AIC final'],
                        len(glm['hist_ig']), comp_ig['Nb vars'], 'Inv. Gaussien')
    
    def plot_aic_backward(hist_df, titre, color):
        if hist_df.empty:
            st.caption(f"{titre} : aucune variable retirée.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(hist_df) + 1), hist_df['AIC avant'].values,
                marker='o', color=color, linewidth=2, markersize=6)
        for i, row in hist_df.iterrows():
            ax.annotate(row['Variable retirée'], (i + 1, row['AIC avant']),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha='center', rotation=30)
        ax.set_xlabel("Étape d'élimination")
        ax.set_ylabel("AIC")
        ax.set_title(f"Évolution de l'AIC — {titre}", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        close_fig(fig)

    col1, col2 = st.columns(2)
    with col1:
        plot_aic_backward(glm['hist_gamma'], "Gamma", "steelblue")
    with col2:
        plot_aic_backward(glm['hist_ig'], "Inv. Gaussien", "coral")

    st.divider()

    # --- Tableau comparaison ---
    st.subheader("Comparaison des modèles finaux")
    comp = glm['comparaison'].copy()
    best_model = comp.iloc[0]['Modèle']
    def highlight_best(row):
        color = 'background-color: #d4edda; color: #155724;' if row.name == 0 else ''
        return [color] * len(row)
    st.dataframe(
        comp.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.info(f"**Meilleur modèle : {best_model}** (AIC le plus bas + meilleures métriques)")

    st.markdown(
        "> **INTERPRETATION DU TABLEAU :** L'**AIC** (Akaike Information Criterion) mesure la qualité "
        "d'ajustement en pénalisant la complexité — **plus il est bas, meilleur est le modèle**. "
        "La **backward elimination** a réduit le nombre de variables en ne gardant que celles "
        "significatives (p < 5%), ce qui améliore l'AIC final par rapport au modèle complet. "
        "La **MAE** et la **RMSE** mesurent l'erreur de prédiction en secondes sur le jeu de validation."
    )

    st.divider()

    # --- Coefficients ---
    st.subheader("Coefficients des modèles finaux")
    tab_g, tab_ig = st.tabs(["Gamma (Log)", "Inv. Gaussien (Log)"])

    st.markdown(
        "> **LES COEFFICIENTS :** Dans un GLM à lien log, l'effet d'une variable s'interprète "
        "via **exp(Coef)** — le **multiplicateur** appliqué à la durée prédite. "
        "exp(Coef) > 1 → la variable **augmente** la durée | exp(Coef) < 1 → elle la **réduit**. "
        "Exemple : exp(Coef) = 1.5 signifie que la durée est multipliée par 1.5 (+ 50%)."
    )

    def show_coefs(coef_df, label):
        df = coef_df.copy()
        df['Significatif'] = df['p-value'].apply(
            lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        )
        df = df.sort_values('exp(Coef)', ascending=False)
        st.dataframe(df.style.format({'Coef': '{:.4f}', 'exp(Coef)': '{:.4f}', 'p-value': '{:.4f}'}),
                     width='stretch', hide_index=True)

        # Graphique des 15 plus forts effets
        top15 = df.head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71' if v >= 1 else '#e74c3c' for v in top15['exp(Coef)']]
        ax.barh(top15['Variable'], top15['exp(Coef)'], color=colors)
        ax.axvline(1, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel("exp(Coef) — effet multiplicatif sur la durée")
        ax.set_title(f"Top 15 variables — {label}", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        close_fig(fig)
        st.info(
            "** Graphoque :** Les barres **vertes** (exp(Coef) > 1) indiquent les variables "
            "qui **augmentent** la durée, les barres **rouges** (exp(Coef) < 1) celles qui la **réduisent**. "
            "La ligne pointillée à x = 1 est le seuil d'effet nul."
        )

    with tab_g:
        show_coefs(glm['coef_gamma'], 'Gamma (Log)')
    with tab_ig:
        show_coefs(glm['coef_ig'], 'Inv. Gaussien (Log)')

    st.divider()

    # --- VIF ---
    st.subheader("Analyse VIF — Multicolinéarité")
    st.markdown(
        "Le **Variance Inflation Factor (VIF)** mesure à quel point chaque variable explicative "
        "est corrélée aux autres. Un VIF élevé signale que la variable est en partie redondante, "
        "ce qui peut **biaiser les coefficients** GLM. Seuils : VIF > 5 = modéré | VIF > 10 = fort."
    )
    tab_vg, tab_vi = st.tabs(["VIF Gamma", "VIF Inv. Gaussien"])

    def show_vif(vif_df):
        vif_df = vif_df.copy()
        vif_df['Niveau'] = vif_df['VIF'].apply(
            lambda v: '🔴 Fort (>10)' if v > 10 else ('🟡 Modéré (>5)' if v > 5 else '🟢 OK (<5)')
        )
        st.dataframe(vif_df[['Variable', 'VIF', 'Niveau']].style.format({'VIF': '{:.2f}'}),
                     use_container_width=True, hide_index=True)
        n_ok = (vif_df['VIF'] < 5).sum()
        n_mod = ((vif_df['VIF'] >= 5) & (vif_df['VIF'] < 10)).sum()
        n_fort = (vif_df['VIF'] >= 10).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("VIF < 5 (OK)", n_ok)
        c2.metric("VIF 5-10 (Modéré)", n_mod)
        c3.metric("VIF > 10 (Fort)", n_fort)

    with tab_vg:
        show_vif(glm['vif_gamma'])
    with tab_vi:
        show_vif(glm['vif_ig'])


def _tab_predictions(pred_test: pd.DataFrame):
    """pred_test = predictions_test.csv (contient pred_gamma et pred_invgauss)."""
    st.subheader("Prédictions GLM sur les dossiers TEST")
    st.markdown(f"""
    Les **{len(pred_test):,} dossiers TEST** ont une durée initiale inconnue (durée = 0).
    Les deux modèles GLM (Gamma et Inverse Gaussien) ont été appliqués pour prédire leur durée de traitement.
    """)

    pred = pred_test[['Numero_dossier_ID', 'Cause.intervention', 'Type.d.energie',
                       'pred_gamma', 'pred_invgauss']].copy()
    
    # # Détection instabilité Inv. Gaussien (valeurs aberrantes)
    # seuil_aberrant = 3600 * 24 * 365  # 1 an en secondes
    # ig_instable = (pred['pred_invgauss'] > seuil_aberrant).any()

    # --- KPIs ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nb dossiers prédits", f"{len(pred):,}")
    c2.metric("Durée médiane Gamma", fmt_seconds(pred['pred_gamma'].median()))
    c3.metric("Durée maximale Gamma", fmt_seconds(pred['pred_gamma'].max()))
    c4.metric("Durée médiane Inv. Gauss", fmt_seconds(pred['pred_invgauss'].median()))
    c5.metric("Durée maximale Inv. Gauss", fmt_seconds(pred['pred_invgauss'].max()))

    #if ig_instable:
        # #st.warning(
        #     "Le modèle **Inv. Gaussien** présente une **instabilité numérique** : certaines prédictions "
        #     "atteignent des valeurs aberrantes (milliards de secondes). Ce phénomène est lié à la "
        #     "sensibilité de la famille Inverse Gaussienne aux valeurs extrêmes. "
        #     "**Le modèle Gamma est retenu comme modèle de référence.**"
        # )
    st.divider()

    # --- Statistiques descriptives ---
    st.subheader("Statistiques descriptives des prédictions")
    pred_stats = pd.DataFrame({
        'Gamma': pred['pred_gamma'].describe(),
        'Inv. Gaussien': pred['pred_invgauss'].describe(),
    })
    
    st.dataframe(pred_stats.round(1))

    st.divider()
    
    # --- Tableau des prédictions par dossier ---
    st.subheader("Détail des prédictions par dossier")
    pred_display = pred.copy()
    pred_display['pred_gamma_fmt'] = pred_display['pred_gamma'].apply(fmt_seconds)
    pred_display['pred_invgauss_fmt'] = pred_display['pred_invgauss'].apply(fmt_seconds)
    
    # --- Filtres ---
    col_filtre_1, col_filtre_2 = st.columns(2)
    with col_filtre_1:
        causes = ['Toutes'] + sorted(pred_display['Cause.intervention'].dropna().unique().tolist())
        filtre_cause = st.selectbox("Filtrer par Cause d'intervention", causes)
    with col_filtre_2:
        energies = ['Toutes'] + sorted(pred_display['Type.d.energie'].dropna().unique().tolist())
        filtre_energie = st.selectbox("Filtrer par Type d'énergie", energies)

    if filtre_cause != 'Toutes':
        pred_display = pred_display[pred_display['Cause.intervention'] == filtre_cause]
    if filtre_energie != 'Toutes':
        pred_display = pred_display[pred_display['Type.d.energie'] == filtre_energie]

    st.caption(f"{len(pred_display):,} dossiers affichés")
    st.dataframe(
        pred_display[['Numero_dossier_ID', 'Cause.intervention', 'Type.d.energie',
                      'pred_gamma_fmt', 'pred_invgauss_fmt']]
        .rename(columns={
            'pred_gamma_fmt': 'Durée Gamma',
            'pred_invgauss_fmt': 'Durée Inv. Gaussien',
        }),
        use_container_width=True,
    )

    st.subheader("Top 10 dossiers avec la durée Gamma prédite la plus longue")

    top10 = (pred.nlargest(10, 'pred_gamma')
             [['Numero_dossier_ID', 'Cause.intervention', 'Type.d.energie',
               'pred_gamma', 'pred_invgauss']]
             .copy())
    top10['Durée Gamma']        = top10['pred_gamma'].apply(fmt_seconds)
    top10['Durée Inv. Gaussien'] = top10['pred_invgauss'].apply(fmt_seconds)

    st.dataframe(
        top10[['Numero_dossier_ID', 'Cause.intervention', 'Type.d.energie',
               'Durée Gamma', 'Durée Inv. Gaussien']].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    
# ─── Point d'entrée principal ─────────────────────────────────────────────────

def render_econometrie(base: pd.DataFrame, base_test: pd.DataFrame):
    st.title("Économétrie — Analyse Statistique & Modélisation")
    st.markdown("""
    Ce module réalise l'analyse statistique des dossiers d'assistance et la modélisation
    économétrique de la **durée de traitement** via des **GLM à lien log**.
    """)

    # Chargement de predictions_test.csv (contient pred_gamma et pred_invgauss)
    try:
        pred_test = pd.read_csv('../data/processed/predictions_test.csv')
    except FileNotFoundError:
        pred_test = None

        # ← BANNIÈRE ICI ↓
    duree_med = base['duree_corrigee_totale'].median()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dossiers TRAIN", f"{len(base):,}")
    c2.metric("Dossiers TEST", f"{len(base_test):,}" if base_test is not None else "—")
    c3.metric("Durée médiane globale", fmt_seconds(duree_med))
    c4.metric("Variables GLM candidates", str(len(VAR_QUANTI_GLM) + len(VAR_QUALI_GLM) + len(VAR_BIN_GLM)))



    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Analyse Univariée",
        "🔗 Analyse Bivariée",
        "🧩 AFDM",
        "📈 Modèles GLM",
        "🔮 Prédictions TEST",
    ])

    with tab1:
        _tab_univarie(base)
    with tab2:
        _tab_bivarie(base)
    with tab3:
        _tab_afdm(base)
    with tab4:
        _tab_glm(base)
    with tab5:
        if pred_test is None:
            st.error("Fichier `predictions_test.csv` introuvable. Veuillez d'abord exécuter le notebook.")
        else:
            _tab_predictions(pred_test)