import pandas as pd
import os

def full_data_analysis(filepath, target_col=None):
    df = pd.read_csv(filepath)
    
    print("=== Taille du dataset ===")
    print(df.shape)
    print("\n=== Aperçu des premières lignes ===")
    print(df.head())
    
    print("\n=== Types des colonnes ===")
    print(df.dtypes)
    
    print("\n=== Statistiques descriptives numériques ===")
    print(df.describe())

    # Analyse des valeurs manquantes
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if missing_cols.empty:
        print("\nAucune valeur manquante détectée.")
    else:
        print("\n=== Valeurs manquantes par colonne ===")
        print(missing_cols)
        print("\n=== Pourcentage de valeurs manquantes ===")
        print((missing_cols / len(df)) * 100)
    
    # Distribution de la variable cible si spécifiée
    if target_col and target_col in df.columns:
        print(f"\n=== Distribution de la variable cible '{target_col}' ===")
        print(df[target_col].value_counts())
        print("\nProportions :")
        print(df[target_col].value_counts(normalize=True))


def prepare_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Supprimer colonne inutile
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Imputation des valeurs manquantes
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
    
    # Créer dossier processed s’il n’existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sauvegarder fichier nettoyé
    df.to_csv(output_path, index=False)
    print(f"✅ Données nettoyées sauvegardées dans {output_path}")
    
    return df  # retourner le df nettoyé pour usage ultérieur (ex: imputations test)

def prepare_test(input_path, output_path, train_df):
    df = pd.read_csv(input_path)
    
    # Supprimer colonne inutile
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Imputation avec les médianes du train
    df['MonthlyIncome'].fillna(train_df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(train_df['NumberOfDependents'].median(), inplace=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Données test nettoyées sauvegardées dans {output_path}")

if __name__ == "__main__":
    raw_train = "data/raw/cs-training.csv"
    processed_train = "data/processed/train_clean.csv"
    
    raw_test = "data/raw/cs-test.csv"
    processed_test = "data/processed/test_clean.csv"
    
    print("🔍 Analyse des données brutes (train) :")
    full_data_analysis(raw_train, target_col="SeriousDlqin2yrs")
    
    print("\n🧹 Préparation des données (train) :")
    train_df = prepare_data(raw_train, processed_train)
    
    print("\n🧹 Préparation des données (test) :")
    prepare_test(raw_test, processed_test, train_df)
