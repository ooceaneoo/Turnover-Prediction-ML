import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:OpenClassRoom@localhost:5432/turnover_db"
)

DISABLE_DB = os.getenv("DISABLE_DB", "false").lower() == "true"

if DISABLE_DB:
    engine = None
    SessionLocal = None
else:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

Base = declarative_base()