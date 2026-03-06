from app.db.database import Base, engine
from app.db import models


def create_tables():
    Base.metadata.create_all(bind=engine)
    print("Tables créées avec succès")


if __name__ == "__main__":
    create_tables()