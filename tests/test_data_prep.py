import pandas as pd

from turnover_ml.data_prep import clean_dataset


def test_clean_dataset_creates_binary_target_and_numeric_salary():
    df = pd.DataFrame({
        "id_employee": [1, 2],
        "a_quitte_l_entreprise": ["Oui", "Non"],
        "augementation_salaire_precedente": ["10 %", "5 %"],
        "genre": ["F", "M"],
        "statut_marital": ["Célibataire", "Marié(e)"],
        "departement": ["Commercial", "Consulting"],
        "poste": ["Manager", "Consultant"],
        "domaine_etude": ["Marketing", "Infra & Cloud"],
        "frequence_deplacement": ["Occasionnel", "Aucun"],
        "heure_supplementaires": ["Oui", "Non"],
    })

    out = clean_dataset(df)

    assert "a_quitte_l_entreprise_num" in out.columns
    assert out["a_quitte_l_entreprise_num"].tolist() == [1, 0]
    assert "augmentation_salaire_num" in out.columns
    assert "a_quitte_l_entreprise" not in out.columns
    assert "id_employee" not in out.columns