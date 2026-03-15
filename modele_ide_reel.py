# =============================================================================
#  MODÈLE ÉCONOMÉTRIQUE — IDE AU MAROC EN AFRIQUE
#  Données réelles : WDI (Banque Mondiale) + WGI (Gouvernance)
#  Méthode : Panel à effets fixes (within) + Effets aléatoires (GLS)
#            + Test de Hausman
#  Auteur   : Projet Finance Internationale — ENCG Settat 2025–2026
#  Dépend.  : numpy, pandas, scipy
# =============================================================================
#
#  STRUCTURE DU SCRIPT
#  ───────────────────
#  Section 1 : Chargement & nettoyage des données
#  Section 2 : Spécification du modèle (justification des variables)
#  Section 3 : Estimateur Within (Effets Fixes)
#  Section 4 : Estimateur GLS (Effets Aléatoires)
#  Section 5 : Test de Hausman
#  Section 6 : Modèle 2 — IDE sortants Maroc (OLS)
#  Section 7 : Statistiques descriptives & corrélations
#  Section 8 : Profil Maroc vs moyenne panel
#
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1 — CHARGEMENT DES DONNÉES
# =============================================================================
# Charger le panel construit à partir des fichiers WDI téléchargés
# sur https://databank.worldbank.org/source/world-development-indicators
# + WGI inclus dans le fichier étendu

DATA_PATH = 'panel_ide_final.csv'   # ← Chemin vers votre fichier panel
df = pd.read_csv(DATA_PATH)

print("=" * 68)
print("  MODÈLE IDE — DONNÉES RÉELLES WDI/WGI")
print("=" * 68)
print(f"\n  Observations totales : {len(df)}")
print(f"  Pays                 : {df['country'].nunique()}")
print(f"  Années               : {df['year'].min()} — {df['year'].max()}")
print(f"  Variables            : {df.shape[1]}")


# =============================================================================
# SECTION 2 — SPÉCIFICATION DU MODÈLE
# =============================================================================
#
#  ÉQUATION PRINCIPALE (IDE entrants) :
#
#  IDE_in_it = α_i + β₁·ln(PIB/hab)_it + β₂·INF_it + β₃·OUV_it
#            + β₄·GOUV_it + β₅·ln(XRATE)_it + β₆·DEV_FIN_it
#            + β₇·CC_it + β₈·DETTE_it + β₉·RES_it + ε_it
#
#  Justification de chaque variable :
#  ──────────────────────────────────
#  ln(PIB/hab)   : Effet taille de marché (paradigme OLI, Dunning 1981)
#                  Un revenu élevé = demande locale forte = attractivité IDE
#
#  Inflation     : Instabilité macro → décote sur la rentabilité IDE
#                  Signe attendu : négatif
#
#  Ouverture     : (Exports + Imports) / PIB — intégration commerciale
#                  Signe attendu : positif (marchés ouverts attirent l'IDE)
#
#  Gouvernance   : Indice WGI composite (6 dimensions, échelle -2.5 à +2.5)
#                  Institutions solides → prime de risque faible
#                  Signe attendu : positif
#
#  ln(Taux de change) : Log du taux officiel LCU/USD (proxy REER)
#                  Dépréciation → coûts locaux baissent → IDE ↑
#                  Signe attendu : ambivalent selon la théorie
#
#  Dév. financier : Crédit privé / PIB — profondeur du système bancaire
#                  Facilite rapatriement profits et financement local
#                  Signe attendu : positif
#
#  Cpte courant  : Déficit = besoin financement externe → absorbe IDE
#                  Signe attendu : négatif (déficit attire les flux)
#
#  Dette ext.    : Stock dette externe / GNI
#                  Fardeau = signal de risque, mais aussi lié au besoin de FX
#                  Signe attendu : ambigu
#
#  Réserves      : Mois d'importations couverts
#                  Signal de solvabilité → attire les investisseurs
#                  Signe attendu : positif

VARS_X = [
    'ln_gdp_pc',        # ln(PIB par habitant)
    'inflation',         # Inflation GDP deflator (%)
    'trade_openness',    # Ouverture commerciale (X+M)/PIB
    'governance',        # Indice WGI composite
    'ln_xrate',          # ln(Taux de change officiel LCU/USD)
    'dev_fin',           # Crédit privé / PIB
    'current_account',   # Solde compte courant / PIB
    'external_debt',     # Dette extérieure / GNI
    'reserves',          # Réserves en mois d'importations
]

LABELS = {
    'ln_gdp_pc':        'ln(PIB/hab)',
    'inflation':        'Inflation (%)',
    'trade_openness':   'Ouverture commerciale',
    'governance':       'Gouvernance (WGI)',
    'ln_xrate':         'ln(Taux de change)',
    'dev_fin':          'Développement financier',
    'current_account':  'Solde compte courant',
    'external_debt':    'Dette extérieure/GNI',
    'reserves':         'Réserves (mois import.)',
}

df_clean = df.dropna(subset=['ide_in'] + VARS_X).copy()
print(f"\n  Observations (après suppression NaN) : {len(df_clean)}")
print(f"  Pays retenus : {df_clean['country'].nunique()}")


# =============================================================================
# UTILITAIRES ÉCONOMÉTRIQUES
# =============================================================================

def ols_hc1(Y, X):
    """MCO avec erreurs robustes HC1 (sandwich hétéroscédasticité)."""
    n, k = X.shape
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ b
    ss_res = e @ e
    ss_tot = ((Y - Y.mean())**2).sum()
    R2     = 1 - ss_res / ss_tot
    R2adj  = 1 - (1 - R2) * (n-1) / (n-k)
    XtXi   = np.linalg.pinv(X.T @ X)
    meat   = sum(float(e[i])**2 * np.outer(X[i], X[i]) for i in range(n))
    V      = (n / (n-k)) * XtXi @ meat @ XtXi
    se     = np.sqrt(np.diag(V))
    t_     = b / se
    p_     = 2 * (1 - stats.t.cdf(np.abs(t_), df=n-k))
    return b, se, t_, p_, R2, R2adj


def within_transform(Y, X, groups):
    """Transformée Within : soustrait les moyennes individuelles (demeaning)."""
    Yw, Xw = Y.copy().astype(float), X.copy().astype(float)
    for g in np.unique(groups):
        m      = groups == g
        Yw[m] -= Y[m].mean()
        Xw[m] -= X[m].mean(axis=0)
    return Yw, Xw


def sig_stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else '   '


def print_table(title, var_names, b, se, p, extras=None):
    print(f"\n  {title}")
    print(f"  {'Variable':<26} {'Coef.':>9} {'Std.Err.':>10} {'p-val':>7} {'sig':>4}")
    print("  " + "-" * 58)
    for v, ci, si, pi in zip(var_names, b, se, p):
        lab = LABELS.get(v, 'Constante' if v == 'const' else v)
        print(f"  {lab:<26} {ci:>9.4f} {si:>10.4f} {pi:>7.3f} {sig_stars(pi):>4}")
    print("  " + "-" * 58)
    if extras:
        for k_, v_ in extras.items():
            print(f"  {k_:<26} {v_}")


# =============================================================================
# SECTION 3 — ESTIMATEUR WITHIN (EFFETS FIXES)
# =============================================================================
#
#  PRINCIPE :
#  L'estimateur Within (ou Within Groups) élimine les effets individuels
#  non observés α_i en soustrayant les moyennes individuelles de chaque
#  variable. Cela permet de contrôler toute hétérogénéité fixe dans le
#  temps propre à chaque pays (culture, géographie, histoire).
#
#  AVANTAGE   : Élimine les biais de variables omises invariantes dans
#               le temps (ex. dotation en ressources naturelles).
#  INCONVÉNIENT : Ne peut pas estimer les effets de variables invariantes
#                 dans le temps (ex. superficie du pays).

print("\n" + "=" * 68)
print("  SECTION 3 — EFFETS FIXES (Within estimator)")
print("=" * 68)

groups = df_clean['country'].values
Y      = df_clean['ide_in'].values.astype(float)
X      = df_clean[VARS_X].values.astype(float)
N_c    = len(np.unique(groups))
T_avg  = len(Y) / N_c

Yw, Xw         = within_transform(Y, X, groups)
b_ef, se_ef, t_ef, p_ef, R2w, _ = ols_hc1(Yw, Xw)

# R² Between (variation inter-pays)
g_uniq = np.unique(groups)
Yb     = np.array([Y[groups==g].mean() for g in g_uniq])
Xb     = np.column_stack([np.ones(N_c),
         np.array([X[groups==g].mean(axis=0) for g in g_uniq])])
_, _, _, _, R2b, _ = ols_hc1(Yb, Xb)

print_table(
    "Effets Fixes (Within) — IDE entrants", VARS_X,
    b_ef, se_ef, p_ef,
    extras={
        'R² Within':    f'{R2w:.4f}',
        'R² Between':   f'{R2b:.4f}',
        'Observations': str(len(Y)),
        'Erreurs':      'HC1 robustes',
    }
)


# =============================================================================
# SECTION 4 — ESTIMATEUR GLS (EFFETS ALÉATOIRES)
# =============================================================================
#
#  PRINCIPE :
#  L'estimateur GLS (Generalized Least Squares) de Swamy-Arora suppose
#  que α_i ~ N(0, σ²_b) — les effets individuels sont des variables
#  aléatoires non corrélées avec les régresseurs.
#  Il effectue un quasi-demeaning avec θ ∈ [0,1] :
#      θ = 1 - √(σ²_w / (T·σ²_b + σ²_w))
#  θ = 0 → MCO classique | θ = 1 → Within (EF)
#
#  AVANTAGE   : Plus efficace que EF si l'hypothèse d'orthogonalité tient.
#  INCONVÉNIENT : Biaisé si α_i est corrélé avec les X (à tester via Hausman).

print("\n" + "=" * 68)
print("  SECTION 4 — EFFETS ALÉATOIRES (GLS Swamy-Arora)")
print("=" * 68)

sig2_w = (Yw - Xw @ b_ef) @ (Yw - Xw @ b_ef) / (len(Y) - N_c - len(VARS_X))
b_btw_full, _, _, _, _, _ = ols_hc1(Yb, Xb)
e_btw  = Yb - Xb @ b_btw_full
sig2_b = max(e_btw @ e_btw / (N_c - len(VARS_X) - 1) - sig2_w / T_avg, 1e-8)
theta  = 1 - np.sqrt(sig2_w / (T_avg * sig2_b + sig2_w))

print(f"\n  Composantes de variance :")
print(f"    σ²_within   = {sig2_w:.4f}")
print(f"    σ²_between  = {sig2_b:.4f}")
print(f"    θ (quasi-demeaning) = {theta:.4f}")

Yea, Xea = Y.copy(), X.copy()
for g in g_uniq:
    m = groups == g
    Yea[m] -= theta * Y[m].mean()
    Xea[m] -= theta * X[m].mean(axis=0)
Xea_c = np.column_stack([(1-theta) * np.ones(len(Yea)), Xea])
b_ea_f, se_ea_f, _, p_ea_f, R2ea, _ = ols_hc1(Yea, Xea_c)
b_ea, se_ea, p_ea = b_ea_f[1:], se_ea_f[1:], p_ea_f[1:]

print_table(
    f"Effets Aléatoires (GLS, θ={theta:.3f})", VARS_X,
    b_ea, se_ea, p_ea,
    extras={'R² GLS': f'{R2ea:.4f}', 'Erreurs': 'HC1 robustes'}
)


# =============================================================================
# SECTION 5 — TEST DE HAUSMAN
# =============================================================================
#
#  PRINCIPE :
#  Le test de Hausman (1978) vérifie si la différence entre les estimateurs
#  EF et EA est systématique.
#
#  H₀ : E[α_i | X_it] = 0  →  les effets individuels ne sont pas corrélés
#       avec les régresseurs → EA est CONSISTANT et EFFICACE
#  H₁ : E[α_i | X_it] ≠ 0  →  EF est PRÉFÉRABLE (même si moins efficace)
#
#  Statistique : H = (b_EF - b_EA)' [V_EF - V_EA]⁻¹ (b_EF - b_EA) ~ χ²(K)

print("\n" + "=" * 68)
print("  SECTION 5 — TEST DE HAUSMAN")
print("=" * 68)

diff = b_ef - b_ea
Vd   = np.diag(se_ef**2) - np.diag(se_ea**2)
try:
    H_stat = float(diff @ np.linalg.pinv(Vd) @ diff)
    H_df   = len(VARS_X)
    H_p    = 1 - stats.chi2.cdf(H_stat, df=H_df)
    print(f"\n  H statistique    = {H_stat:.4f}")
    print(f"  Degrés de liberté= {H_df}")
    print(f"  p-value          = {H_p:.4f}")
    if H_p < 0.05:
        print(f"\n  → REJET de H₀ (p < 0.05)")
        print(f"  → Effets FIXES retenus : endogénéité des effets individuels détectée")
        print(f"  → Interprétation : les caractéristiques non observées des pays")
        print(f"     (institutions, histoire, géographie) sont corrélées avec les")
        print(f"     déterminants des IDE → l'estimateur Within est nécessaire.")
    else:
        print(f"\n  → NON-REJET de H₀ (p ≥ 0.05)")
        print(f"  → Effets ALÉATOIRES retenus : plus efficaces ici")
except Exception as ex:
    print(f"  Hausman non convergé : {ex}")

# Tableau comparatif
print(f"\n  Synthèse EF vs EA :")
print(f"  {'Variable':<26} {'EF coef':>9} {'sig':>4}   {'EA coef':>9} {'sig':>4}")
print("  " + "-" * 56)
for i, v in enumerate(VARS_X):
    print(f"  {LABELS[v]:<26} {b_ef[i]:>9.4f} {sig_stars(p_ef[i]):>4}   "
          f"{b_ea[i]:>9.4f} {sig_stars(p_ea[i]):>4}")
print("  Note : *** p<0.01  ** p<0.05  * p<0.10")


# =============================================================================
# SECTION 6 — MODÈLE 2 : IDE SORTANTS MAROC (OLS)
# =============================================================================
#
#  ÉQUATION :
#  IDE_out_t = α + β₁·ln(PIB/hab)_t + β₂·DEV_FIN_t
#            + β₃·ln(XRATE)_t + β₄·RES_t + ε_t
#
#  Justification :
#  ── ln(PIB/hab)   : Push factor — richesse interne → capacité à investir
#                     à l'étranger (théorie d'Uppsala : expansion graduelle)
#  ── Dév. financier: Système bancaire mature → financement de l'expansion
#                     africaine (Attijariwafa 44 pays, BMCE 20 pays)
#  ── ln(Taux change): Dépréciation → compétitivité des firmes marocaines
#                     à l'international
#  ── Réserves      : Abondance → espace pour investir sans contrainte BDP

print("\n" + "=" * 68)
print("  SECTION 6 — IDE SORTANTS MAROC (OLS)")
print("=" * 68)
print("  Firmes : Attijariwafa Bank, BMCE/Bank of Africa, OCP, Maroc Telecom")

VARS_OUT   = ['ln_gdp_pc', 'dev_fin', 'ln_xrate', 'reserves']
LABELS_OUT = {
    'ln_gdp_pc': 'ln(PIB/hab)',
    'dev_fin':   'Dév. financier',
    'ln_xrate':  'ln(Taux de change)',
    'reserves':  'Réserves',
}

df_m = (df[df['country'] == 'Morocco']
        .dropna(subset=['ide_out'] + VARS_OUT)
        .sort_values('year').copy())

Y2   = df_m['ide_out'].values.astype(float)
X2   = np.column_stack([np.ones(len(df_m)), df_m[VARS_OUT].values.astype(float)])
b2, se2, t2, p2, R2_2, R2adj_2 = ols_hc1(Y2, X2)
fs   = (R2_2 / len(VARS_OUT)) / ((1 - R2_2) / (len(Y2) - len(VARS_OUT) - 1))
fp   = 1 - stats.f.cdf(fs, len(VARS_OUT), len(Y2) - len(VARS_OUT) - 1)

print_table(
    f"OLS HC1 — IDE sortants Maroc (T={len(Y2)})",
    ['const'] + VARS_OUT, b2, se2, p2,
    extras={
        'R²':           f'{R2_2:.4f}',
        'R² ajusté':    f'{R2adj_2:.4f}',
        'F-stat':       f'{fs:.4f}   p = {fp:.4f}',
        'Observations': str(len(Y2)),
    }
)


# =============================================================================
# SECTION 7 — STATISTIQUES DESCRIPTIVES & CORRÉLATIONS
# =============================================================================

print("\n" + "=" * 68)
print("  SECTION 7 — STATISTIQUES DESCRIPTIVES")
print("=" * 68)

desc = df_clean[['ide_in'] + VARS_X].describe().T[['mean','std','min','50%','max']]
desc.columns = ['Moyenne', 'Éc.-type', 'Min', 'Médiane', 'Max']
desc.index   = ['IDE entrants'] + [LABELS[v] for v in VARS_X]
print("\n" + desc.round(3).to_string())

print("\n  Corrélations avec IDE entrants :")
corr = df_clean[['ide_in'] + VARS_X].corr()['ide_in'].drop('ide_in').sort_values(ascending=False)
for v, val in corr.items():
    bar = '█' * int(abs(val) * 24)
    print(f"    {LABELS.get(v,v):<28} {val:+.3f}  {bar}")


# =============================================================================
# SECTION 8 — PROFIL MAROC vs MOYENNE PANEL
# =============================================================================

print("\n" + "=" * 68)
print("  SECTION 8 — PROFIL MAROC vs MOYENNE PANEL")
print("=" * 68)

m_avg = df_clean[df_clean['country'] == 'Morocco'][['ide_in'] + VARS_X].mean()
p_avg = df_clean[['ide_in'] + VARS_X].mean()

print(f"\n  {'Variable':<28} {'Maroc':>9} {'Panel':>9} {'Écart':>9} {'Rang'}")
print("  " + "-" * 62)
for v in ['ide_in'] + VARS_X:
    lab   = 'IDE entrants' if v == 'ide_in' else LABELS[v]
    ecart = m_avg[v] - p_avg[v]
    arrow = '↑↑' if ecart > p_avg[v]*0.5 else ('↑' if ecart > 0 else '↓')
    print(f"  {lab:<28} {m_avg[v]:>9.3f} {p_avg[v]:>9.3f} {ecart:>+9.3f}  {arrow}")

print("\n  Lecture :")
print("  ↑↑ = Maroc nettement au-dessus de la moyenne africaine")
print("  ↑  = Maroc légèrement au-dessus")
print("  ↓  = Maroc en-dessous de la moyenne")

print("\n" + "=" * 68)
print("  FIN — Données réelles WDI/WGI | ENCG Settat 2025–2026")
print("=" * 68)
