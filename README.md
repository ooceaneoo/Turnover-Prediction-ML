---
title: Turnover Prediction API
sdk: docker
app_port: 7860
---

# Turnover Prediction API

API permettant de prédire le départ d’un employé à partir de données RH à l’aide d’un modèle de machine learning.

Le projet comprend :

- un pipeline de machine learning reproductible pour l'entraînement du modèle
- une API FastAPI exposant le modèle en service de prédiction
- une validation des entrées avec Pydantic
- une base PostgreSQL locale pour la traçabilité des prédictions
- des tests unitaires et tests API avec Pytest
- une intégration continue (CI) avec GitHub Actions
- un déploiement Docker sur Hugging Face Spaces

---

## API déployée

L'API est déployée publiquement sur Hugging Face Spaces :

https://oceaneo-turnover-prediction-api.hf.space/docs

Vous pouvez tester directement l'API via la documentation interactive Swagger.

Endpoints :

- `GET /health`
- `GET /schema`
- `GET /example`
- `POST /predict`
- `POST /predict_csv`

---

# Architecture du projet

Le projet expose un modèle de machine learning sous forme de service API.

```
Client
   │
   ▼
FastAPI API
   │
   ▼
Machine Learning Pipeline
(StandardScaler + OneHotEncoder + Logistic Regression)
   │
   ▼
Prediction
   │
   ├── Response retournée à l'utilisateur
   │
   └── (Local uniquement)
        ▼
      PostgreSQL
      (logging des requêtes et prédictions)
```

Cette architecture permet d'exposer un modèle de machine learning sous forme de service API afin de rendre les prédictions accessibles à des applications externes.

---

# Modèle utilisé

Le modèle utilisé est une **régression logistique** avec :

- preprocessing  
  - `StandardScaler` pour les variables numériques  
  - `OneHotEncoder` pour les variables catégorielles  
- gestion du déséquilibre des classes avec **RandomUnderSampler**
- optimisation du seuil de décision pour maximiser le **F1-score**

Les métriques et paramètres du modèle sont sauvegardés dans :

`reports/metrics.json`

Le pipeline entraîné est sauvegardé dans :

`models/pipeline.joblib`

---

# Structure du projet

```
Turnover-Prediction-ML
│
├── app
│   ├── main.py                 # API FastAPI
│   │
│   ├── core
│   │   ├── config.py           # configuration des chemins
│   │   └── model.py            # chargement du modèle
│   │
│   ├── schemas
│   │   └── predict.py          # schémas Pydantic pour l’API
│   │
│   └── db
│       ├── database.py         # connexion PostgreSQL
│       └── models.py           # modèles SQLAlchemy
│
├── src
│   └── turnover_ml             # package contenant le pipeline de machine learning
│       ├── data_prep.py        # préparation et nettoyage des données
│       ├── features.py         # feature engineering
│       └── train.py            # entraînement du modèle
│
├── tests                       # tests unitaires et tests API
│
├── data                        # dataset source utilisé pour l'entraînement (CSV)
│
├── models                      # pipeline ML entraîné (.joblib)
│
├── reports                     # métriques du modèle (metrics.json)
│
├── .github
│   └── workflows
│       └── ci.yml              # pipeline CI GitHub Actions
│
├── create_db.py                # création des tables PostgreSQL
├── load_dataset_to_db.py       # insertion du dataset dans PostgreSQL
│
├── requirements.txt
├── README.md
├── Dockerfile                  # image Docker utilisée pour le déploiement
└── .gitignore
```

---

# Installation

Cloner le dépôt :

```
git clone https://github.com/ooceaneoo/Turnover-Prediction-ML.git
cd Turnover-Prediction-ML
```

Installer les dépendances :

```
pip install -r requirements.txt
```

---

# Entraîner le modèle

Depuis la racine du projet :

```
export PYTHONPATH="$(pwd)/src"
python -m turnover_ml.train
```

Cela va :

- charger les données
- nettoyer le dataset
- créer les features
- entraîner le modèle
- sauvegarder le pipeline et les métriques

---

# Lancer l'API

```
python -m uvicorn app.main:app --reload
```

L’API sera accessible ici :

```
http://127.0.0.1:8000
```

Documentation interactive Swagger :

```
http://127.0.0.1:8000/docs
```

---

# Endpoints

### GET `/health`

Vérifie que l’API fonctionne.

```
{
  "status": "ok"
}
```

---

### POST `/predict`

Prédiction pour un employé.

Exemple de requête :

```
{
  "features": {
    "age": 41,
    "genre": "f",
    "revenu_mensuel": 5993
  }
}
```

Réponse :

```
{
  "probability": 0.86,
  "prediction": 1,
  "threshold": 0.61,
  "model_info": {
    "model": "LogReg + UnderSampling + Preprocessing",
    "test_average_precision": 0.59
  }
}
```

---

### GET `/schema`

Retourne la liste des features attendues par le modèle.

---

### GET `/example`

Retourne un exemple de payload valide pour `/predict`.

---

### POST `/predict_csv`

Permet de faire des prédictions batch à partir d’un fichier CSV.

Chaque ligne du CSV correspond à un employé.

La réponse retourne les données d’origine avec :

- `probability`
- `prediction`

---

# Base de données PostgreSQL

Le projet utilise une base de données **PostgreSQL locale** afin de garantir la **traçabilité des interactions avec le modèle de machine learning**.

Lorsqu’une base PostgreSQL est configurée, chaque appel à l’API est enregistré afin de conserver un historique :

- des **entrées envoyées au modèle**
- des **prédictions produites**

Ces informations permettent :

- de suivre l’utilisation du modèle
- d’analyser les prédictions produites
- d’assurer la traçabilité des décisions du modèle.

⚠️ La base PostgreSQL est utilisée **uniquement en environnement local**.

Dans l’environnement de déploiement (Hugging Face Spaces), la base de données n’est pas activée.

---

## Structure de la base

La base contient trois tables principales :

### `dataset_employes`


Contient le dataset complet utilisé pour l'entraînement du modèle.


Cette table permet de stocker les données dans une base relationnelle plutôt que uniquement dans des fichiers CSV.


### `prediction_requests`


Enregistre chaque requête envoyée au modèle via l’API.


Chaque entrée contient :


- la date de la requête
- la source de la requête
- les features envoyées au modèle (stockées en JSON)


### `prediction_outputs`


Enregistre les résultats produits par le modèle.


Chaque sortie contient :


- la probabilité prédite
- la classe prédite
- le seuil utilisé
- les informations du modèle
- la réponse complète retournée par l’API


La table `prediction_outputs` est reliée à `prediction_requests` afin de conserver le lien entre l’entrée et la prédiction correspondante.

---

# Schéma de la base de données

Un schéma simplifié de la base PostgreSQL est disponible ici :

[Voir le schéma de la base](docs_bdd.md)

---

## Installation et configuration de la base
### 1. Créer la base PostgreSQL


Depuis `psql` :


```sql
CREATE DATABASE turnover_db;
```

### 2. Créer les tables


Le script suivant crée les tables nécessaires :

```
python create_db.py
```

### 3. Insérer le dataset dans la base


Le script suivant charge les fichiers CSV du projet et insère les données dans la table `dataset_employes` :

```
export PYTHONPATH="$(pwd)/src"
python load_dataset_to_db.py
```

---

## Vérification de la base

Se connecter à PostgreSQL :

```
psql -U postgres -d turnover_db
```

Lister les tables :

```
\dt
```

Vérifier le dataset :

```
SELECT COUNT(*) FROM dataset_employes;
```

Vérifier les prédictions enregistrées :

```
SELECT * FROM prediction_requests;
SELECT * FROM prediction_outputs;
```

---

## Traçabilité des prédictions

Lorsqu’un utilisateur appelle l’endpoint `/predict` :

1. les données envoyées sont enregistrées dans `prediction_requests`
2. le modèle effectue la prédiction
3. la réponse du modèle est enregistrée dans `prediction_outputs`


Cela permet :


- de conserver un historique des prédictions
- d’analyser les performances du modèle
- d’assurer la traçabilité des décisions du modèle

---

# Tests

Le projet inclut des **tests automatisés avec Pytest** afin de vérifier le bon fonctionnement du pipeline de machine learning et de l’API.

### Tests unitaires

Les tests unitaires vérifient les principales fonctions du pipeline :

- préparation et nettoyage des données
- création des variables de feature engineering
- bon fonctionnement du pipeline d’entraînement du modèle

### Tests fonctionnels

Les tests fonctionnels vérifient le comportement de l’API FastAPI :

- disponibilité de l’endpoint `/health`
- fonctionnement de l’endpoint `/predict`
- validation des données d’entrée et structure des réponses JSON

Pour exécuter les tests :

```
pytest
```

---

# Intégration continue

Les tests sont exécutés automatiquement via **GitHub Actions** à chaque push.

Workflow :

```
.github/workflows/ci.yml
```

---

# Technologies utilisées

- Python
- pandas
- scikit-learn
- imbalanced-learn
- FastAPI
- Uvicorn
- PostgreSQL
- SQLAlchemy
- Pytest
- GitHub Actions
- Docker

---

# Déploiement

L’API est déployée via **Hugging Face Spaces** en utilisant un environnement **Docker**.

Le conteneur :

- installe les dépendances Python
- charge le pipeline de machine learning
- expose l’API FastAPI via Uvicorn

La base PostgreSQL utilisée pour la traçabilité est uniquement utilisée **en local** et est désactivée dans l’environnement de déploiement.














