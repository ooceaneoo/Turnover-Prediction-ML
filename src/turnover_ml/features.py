import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Feature 1
    df["ratio_anciennete_poste"] = df["annees_dans_le_poste_actuel"] / (df["annees_dans_l_entreprise"] + 1)

    # Feature 2
    satisfaction_cols = [
        "satisfaction_employee_environnement",
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso",
    ]
    df["score_satisfaction_global"] = df[satisfaction_cols].mean(axis=1)

    # Feature 3
    df["stagnation_carriere"] = df["annees_dans_l_entreprise"] - df["annees_depuis_la_derniere_promotion"]

    # Feature 4
    df["salaire_par_annee_experience"] = df["revenu_mensuel"] / (df["annee_experience_totale"] + 1)

    return df