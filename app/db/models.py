from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.database import Base


class DatasetEmploye(Base):
    __tablename__ = "dataset_employes"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    genre = Column(String)
    revenu_mensuel = Column(Integer)
    statut_marital = Column(String)
    departement = Column(String)
    poste = Column(String)
    nombre_experiences_precedentes = Column(Integer)
    annee_experience_totale = Column(Integer)
    annees_dans_l_entreprise = Column(Integer)
    annees_dans_le_poste_actuel = Column(Integer)
    nombre_participation_pee = Column(Integer)
    nb_formations_suivies = Column(Integer)
    distance_domicile_travail = Column(Integer)
    niveau_education = Column(Integer)
    domaine_etude = Column(String)
    frequence_deplacement = Column(String)
    annees_depuis_la_derniere_promotion = Column(Integer)
    annes_sous_responsable_actuel = Column(Integer)
    satisfaction_employee_environnement = Column(Integer)
    note_evaluation_precedente = Column(Integer)
    niveau_hierarchique_poste = Column(Integer)
    satisfaction_employee_nature_travail = Column(Integer)
    satisfaction_employee_equipe = Column(Integer)
    satisfaction_employee_equilibre_pro_perso = Column(Integer)
    note_evaluation_actuelle = Column(Integer)
    heure_supplementaires = Column(String)
    augmentation_salaire_num = Column(Float)
    ratio_anciennete_poste = Column(Float)
    score_satisfaction_global = Column(Float)
    stagnation_carriere = Column(Integer)
    salaire_par_annee_experience = Column(Float)
    a_quitte_l_entreprise_num = Column(Integer)


class PredictionRequest(Base):
    __tablename__ = "prediction_requests"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="api_predict")
    payload = Column(JSON, nullable=False)

    output = relationship("PredictionOutput", back_populates="request", uselist=False)


class PredictionOutput(Base):
    __tablename__ = "prediction_outputs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("prediction_requests.id"), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    probability = Column(Float)
    prediction = Column(Integer)
    threshold = Column(Float)
    model_name = Column(String)
    test_average_precision = Column(Float)
    response_payload = Column(JSON)

    request = relationship("PredictionRequest", back_populates="output")