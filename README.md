# Turnover Prediction API

API permettant de prédire le départ d’un employé à partir de données RH à l’aide d’un modèle de machine learning.

Le projet comprend :

* un pipeline de machine learning reproductible
* une API FastAPI pour exposer le modèle
* des tests unitaires
* une intégration continue (CI) avec GitHub Actions

---

# Modèle utilisé

Le modèle utilisé est une **régression logistique** avec :

* preprocessing

  * `StandardScaler` pour les variables numériques
  * `OneHotEncoder` pour les variables catégorielles
* gestion du déséquilibre des classes avec **RandomUnderSampler**
* optimisation du seuil de décision pour maximiser le **F1-score**

Les métriques et paramètres du modèle sont sauvegardés dans :

```
reports/metrics.json
```

Le pipeline entraîné est sauvegardé dans :

```
models/pipeline.joblib
```

---

# Structure du projet

```
Turnover-Prediction-ML
│
├── app
│   ├── main.py                # API FastAPI
│   ├── core
│   │   ├── config.py         # configuration des chemins
│   │   └── model.py          # chargement du modèle
│   └── schemas
│       └── predict.py        # schémas Pydantic
│
├── src
│   └── turnover_ml
│       ├── data_prep.py      # préparation et nettoyage des données
│       ├── features.py       # feature engineering
│       └── train.py          # entraînement du modèle
│
├── tests                     # tests unitaires
│
├── data                      # données sources
│
├── requirements.txt
├── README.md
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

* charger les données
* nettoyer le dataset
* créer les features
* entraîner le modèle
* sauvegarder le pipeline et les métriques

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

# Endpoints disponibles

### GET `/health`

Vérifie que l’API fonctionne.

Réponse :

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
    "revenu_mensuel": 5993,
    ...
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

* `probability`
* `prediction`

---

# Tests

Les tests unitaires couvrent :

* la préparation des données
* le feature engineering
* le pipeline d’entraînement

Pour lancer les tests :

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

* Python
* pandas
* scikit-learn
* imbalanced-learn
* FastAPI
* Uvicorn
* Pytest
* GitHub Actions
