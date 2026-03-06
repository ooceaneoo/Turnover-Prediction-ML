from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_with_valid_payload():
    payload = {
        "features": {
            "age": 41,
            "genre": "f",
            "revenu_mensuel": 5993,
            "statut_marital": "célibataire",
            "departement": "commercial",
            "poste": "cadre commercial",
            "nombre_experiences_precedentes": 8,
            "annee_experience_totale": 8,
            "annees_dans_l_entreprise": 6,
            "annees_dans_le_poste_actuel": 4,
            "nombre_participation_pee": 0,
            "nb_formations_suivies": 0,
            "distance_domicile_travail": 1,
            "niveau_education": 2,
            "domaine_etude": "infra & cloud",
            "frequence_deplacement": "occasionnel",
            "annees_depuis_la_derniere_promotion": 0,
            "annes_sous_responsable_actuel": 5,
            "satisfaction_employee_environnement": 2,
            "note_evaluation_precedente": 3,
            "niveau_hierarchique_poste": 2,
            "satisfaction_employee_nature_travail": 4,
            "satisfaction_employee_equipe": 1,
            "satisfaction_employee_equilibre_pro_perso": 1,
            "note_evaluation_actuelle": 3,
            "heure_supplementaires": "oui",
            "augmentation_salaire_num": 11.0,
            "ratio_anciennete_poste": 0.5714285714285714,
            "score_satisfaction_global": 2.0,
            "stagnation_carriere": 6,
            "salaire_par_annee_experience": 665.8888888888889
        }
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "probability" in data
    assert "prediction" in data
    assert "threshold" in data
    assert "model_info" in data

    assert isinstance(data["probability"], float)
    assert data["prediction"] in [0, 1]