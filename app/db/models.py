from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.database import Base


class PredictionRequest(Base):
    __tablename__ = "prediction_requests"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    age = Column(Integer)
    genre = Column(String)
    revenu_mensuel = Column(Integer)

    outputs = relationship("PredictionOutput", back_populates="request")


class PredictionOutput(Base):
    __tablename__ = "prediction_outputs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey("prediction_requests.id"))

    probability = Column(Float)
    prediction = Column(Integer)
    threshold = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    request = relationship("PredictionRequest", back_populates="outputs")