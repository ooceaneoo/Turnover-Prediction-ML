# Schéma de la base de données


Le projet utilise une base PostgreSQL locale pour stocker :


- le dataset préparé utilisé dans le projet
- les requêtes envoyées au modèle
- les réponses générées par le modèle


---


## Vue d’ensemble


```mermaid
erDiagram
    DATASET_EMPLOYES {
        int id PK
        int age
        string genre
        int revenu_mensuel
        string statut_marital
        string departement
        string poste
        int nombre_experiences_precedentes
        int annee_experience_totale
        int annees_dans_l_entreprise
        int annees_dans_le_poste_actuel
        int nombre_participation_pee
        int nb_formations_suivies
        int distance_domicile_travail
        int niveau_education
        string domaine_etude
        string frequence_deplacement
        int annees_depuis_la_derniere_promotion
        int annes_sous_responsable_actuel
        int satisfaction_employee_environnement
        int note_evaluation_precedente
        int niveau_hierarchique_poste
        int satisfaction_employee_nature_travail
        int satisfaction_employee_equipe
        int satisfaction_employee_equilibre_pro_perso
        int note_evaluation_actuelle
        string heure_supplementaires
        float augmentation_salaire_num
        float ratio_anciennete_poste
        float score_satisfaction_global
        int stagnation_carriere
        float salaire_par_annee_experience
        int a_quitte_l_entreprise_num
    }


    PREDICTION_REQUESTS {
        int id PK
        datetime created_at
        string source
        json payload
    }


    PREDICTION_OUTPUTS {
        int id PK
        int request_id FK
        datetime created_at
        float probability
        int prediction
        float threshold
        string model_name
        float test_average_precision
        json response_payload
    }


    PREDICTION_REQUESTS ||--|| PREDICTION_OUTPUTS : genere
```


---


# Description des tables


## dataset_employes


Cette table contient le dataset complet préparé pour le projet, après nettoyage et feature engineering.


Elle permet de stocker les données du projet dans PostgreSQL plutôt que de dépendre uniquement de fichiers CSV.


---


## prediction_requests


Cette table enregistre chaque input envoyé au modèle via l’API.


Elle contient :


- la date de la requête
- la source
- le payload complet des features au format JSON


---


## prediction_outputs


Cette table enregistre chaque prédiction produite par le modèle.


Elle contient :


- la probabilité prédite
- la classe prédite
- le seuil utilisé
- le nom du modèle
- la métrique de référence
- la réponse complète au format JSON


---


# Relation entre les tables


- chaque ligne de `prediction_requests` correspond à une requête envoyée au modèle
- chaque ligne de `prediction_outputs` correspond à la sortie associée à une requête
- `prediction_outputs.request_id` référence `prediction_requests.id`


---


# Logique de traçabilité


Le fonctionnement de l’API suit ce cycle :


1. l’utilisateur envoie des données à l’endpoint `/predict`
2. les données sont enregistrées dans `prediction_requests`
3. le modèle effectue la prédiction
4. le résultat est enregistré dans `prediction_outputs`


Cette structure permet de garantir la traçabilité des échanges entre l’API, la base de données et le modèle de machine learning.
