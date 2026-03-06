import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from turnover_ml.features import add_engineered_features


def test_training_pipeline_smoke_runs():
    X = pd.DataFrame({
        "age": [30, 40, 35, 45],
        "genre": ["f", "m", "f", "m"],
        "revenu_mensuel": [3000, 4000, 3500, 4200],
        "statut_marital": ["célibataire", "marié(e)", "célibataire", "divorcé(e)"],
        "departement": ["commercial", "consulting", "commercial", "ressources humaines"],
        "poste": ["manager", "consultant", "manager", "tech lead"],
        "nombre_experiences_precedentes": [1, 2, 1, 3],
        "annee_experience_totale": [5, 8, 6, 10],
        "annees_dans_l_entreprise": [3, 5, 4, 6],
        "annees_dans_le_poste_actuel": [1, 2, 1, 3],
        "nombre_participation_pee": [0, 1, 0, 1],
        "nb_formations_suivies": [1, 2, 1, 3],
        "distance_domicile_travail": [5, 10, 7, 12],
        "niveau_education": [2, 3, 2, 4],
        "domaine_etude": ["marketing", "infra & cloud", "marketing", "autre"],
        "frequence_deplacement": ["aucun", "occasionnel", "aucun", "frequent"],
        "annees_depuis_la_derniere_promotion": [1, 2, 1, 3],
        "annes_sous_responsable_actuel": [2, 3, 2, 4],
        "satisfaction_employee_environnement": [2, 4, 3, 5],
        "note_evaluation_precedente": [3, 4, 3, 5],
        "niveau_hierarchique_poste": [1, 2, 1, 3],
        "satisfaction_employee_nature_travail": [3, 4, 3, 5],
        "satisfaction_employee_equipe": [2, 5, 3, 5],
        "satisfaction_employee_equilibre_pro_perso": [3, 4, 3, 5],
        "note_evaluation_actuelle": [3, 4, 3, 5],
        "heure_supplementaires": ["oui", "non", "oui", "non"],
        "augmentation_salaire_num": [10.0, 5.0, 8.0, 12.0],
    })
    y = pd.Series([1, 0, 0, 0])

    X = add_engineered_features(X)

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("under", RandomUnderSampler(random_state=42)),
        ("logreg", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape[0] == X.shape[0]