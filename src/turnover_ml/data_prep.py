import pandas as pd


def nettoyer_pourcentage(x):
    if isinstance(x, str):
        return float(x.replace(" %", "").strip())
    return None


def load_raw_data(data_dir: str = "data") -> pd.DataFrame:
    df_sirh = pd.read_csv(f"{data_dir}/extrait_sirh.csv")
    df_eval = pd.read_csv(f"{data_dir}/extrait_eval.csv")
    df_sondage = pd.read_csv(f"{data_dir}/extrait_sondage.csv")

    # id_employee depuis eval_number
    df_eval = df_eval.copy()
    df_eval["id_employee"] = (
        df_eval["eval_number"].str.replace("E_", "", regex=False).astype(int)
    )
    df_eval = df_eval.drop(columns=["eval_number"], errors="ignore")

    # rename sondage
    df_sondage = df_sondage.copy().rename(columns={"code_sondage": "id_employee"})

    # merge
    df_temp = df_sirh.merge(df_sondage, on="id_employee", how="inner")
    df = df_temp.merge(df_eval, on="id_employee", how="inner")

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # target binaire
    df["a_quitte_l_entreprise_num"] = df["a_quitte_l_entreprise"].apply(
        lambda x: 1 if x == "Oui" else 0
    )

    # salaire % → float
    df["augmentation_salaire_num"] = df["augementation_salaire_precedente"].apply(nettoyer_pourcentage)
    df = df.drop(columns=["augementation_salaire_precedente"], errors="ignore")

    # nettoyage texte
    cat_cols = [
        "genre",
        "statut_marital",
        "departement",
        "poste",
        "domaine_etude",
        "frequence_deplacement",
        "heure_supplementaires",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.strip()
                .str.lower()
                .str.replace("  ", " ")
            )

    # drop constantes
    constant_cols = df.columns[df.nunique(dropna=False) <= 1]
    df = df.drop(columns=constant_cols, errors="ignore")

    # drop id et target texte
    df = df.drop(columns=["id_employee", "a_quitte_l_entreprise"], errors="ignore")

    return df