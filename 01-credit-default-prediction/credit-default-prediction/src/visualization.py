import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_data(filepath="", output_dir=""):
    # Créer dossier pour sauvegarder les figures
    
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # Remonter à la racine du projet
    base_path = os.path.join(BASE_DIR, "01-credit-default-prediction", "credit-default-prediction")
    print("base_path",base_path)
    filepath = os.path.join(base_path, "data", "processed", "train_clean.csv")
    print("base_path",filepath)
    output_dir = os.path.join(base_path, "reports", "figures")
    

    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données
    df = pd.read_csv(filepath)
    
    # 1. Aperçu général
    print("Taille du dataset :", df.shape)
    print(df.head())
    
    # 2. Distribution de la cible
    plt.figure(figsize=(6,4))
    sns.countplot(x='SeriousDlqin2yrs', data=df)
    plt.title("Distribution de la variable cible")
    plt.savefig(f"{output_dir}/target_distribution.png")
    plt.close()
    
    # 3. Histogrammes des variables numériques
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'SeriousDlqin2yrs' in num_cols:
        num_cols.remove('SeriousDlqin2yrs')
    
    plt.figure(figsize=(15,12))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(4, 3, i)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution de {col}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/numeric_distributions.png")
    plt.close()
    
    # 4. Boxplots des variables par rapport à la cible
    plt.figure(figsize=(15,12))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(4, 3, i)
        sns.boxplot(x='SeriousDlqin2yrs', y=col, data=df)
        plt.title(f"{col} vs SeriousDlqin2yrs")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/boxplots_by_target.png")
    plt.close()
    
    # 5. Matrice de corrélation
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Matrice de corrélation")
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    print(f"✅ Visualisations sauvegardées dans {output_dir}")

if __name__ == "__main__":
    visualize_data()
