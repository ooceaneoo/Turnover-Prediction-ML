from app.db.database import SessionLocal
from app.db.models import DatasetEmploye
from turnover_ml.data_prep import load_raw_data, clean_dataset
from turnover_ml.features import add_engineered_features


def main():
    df_raw = load_raw_data("data")
    df = clean_dataset(df_raw)
    df = add_engineered_features(df)

    session = SessionLocal()

    try:
        for _, row in df.iterrows():
            employe = DatasetEmploye(**row.to_dict())
            session.add(employe)

        session.commit()
        print(f"{len(df)} lignes insérées dans dataset_employes.")
    finally:
        session.close()


if __name__ == "__main__":
    main()