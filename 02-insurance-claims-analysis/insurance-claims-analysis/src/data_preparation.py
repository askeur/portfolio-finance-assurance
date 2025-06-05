import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class InsuranceDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, bene_file_path, inpatient_claims_path):
        """Charger les fichiers CSV"""
        print("📊 Chargement des données...")
        
        try:
            self.bene_df = pd.read_csv(bene_file_path)
            self.inpatient_df = pd.read_csv(inpatient_claims_path)
            
            print(f"✅ Beneficiaires chargés: {self.bene_df.shape}")
            print(f"✅ Claims chargés: {self.inpatient_df.shape}")
            
            return self.bene_df, self.inpatient_df
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return None, None
    
    def explore_data(self):
        """Explorer la structure des données"""
        print("\n" + "="*50)
        print("🔍 EXPLORATION DES DONNÉES")
        print("="*50)
        
        print("\n📋 BENEFICIAIRES:")
        print(f"Shape: {self.bene_df.shape}")
        print(f"Colonnes: {list(self.bene_df.columns)}")
        print(f"Types: \n{self.bene_df.dtypes}")
        print(f"Valeurs manquantes: \n{self.bene_df.isnull().sum()}")
        
        print("\n🏥 INPATIENT CLAIMS:")
        print(f"Shape: {self.inpatient_df.shape}")
        print(f"Colonnes: {list(self.inpatient_df.columns)}")
        print(f"Types: \n{self.inpatient_df.dtypes}")
        print(f"Valeurs manquantes: \n{self.inpatient_df.isnull().sum()}")
        
    def merge_data(self):
        """Fusionner les données sur une clé commune"""
        print("\n🔗 Fusion des données...")
        
        # Identifier les colonnes communes
        common_cols = set(self.bene_df.columns) & set(self.inpatient_df.columns)
        print(f"Colonnes communes trouvées: {common_cols}")
        
        # Généralement 'BeneID' ou similar
        if 'BeneID' in common_cols:
            merge_key = 'BeneID'
        elif 'Provider' in common_cols:
            merge_key = 'Provider'
        else:
            merge_key = list(common_cols)[0] if common_cols else None
            
        if merge_key:
            self.merged_df = pd.merge(self.bene_df, self.inpatient_df, 
                                    on=merge_key, how='inner')
            print(f"✅ Fusion réussie sur '{merge_key}': {self.merged_df.shape}")
        else:
            print("⚠️ Pas de clé commune trouvée, concaténation des données")
            self.merged_df = pd.concat([self.bene_df, self.inpatient_df], axis=1)
            
        return self.merged_df
    
    def clean_data(self):
        """Nettoyer les données"""
        print("\n🧹 Nettoyage des données...")
        
        df = self.merged_df.copy()
        initial_shape = df.shape
        
        # 1. Supprimer les doublons
        df = df.drop_duplicates()
        print(f"Doublons supprimés: {initial_shape[0] - df.shape[0]}")
        
        # 2. Gérer les valeurs manquantes
        # Pour les numériques: médiane
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Pour les catégorielles: mode ou 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        
        # 3. Supprimer les colonnes avec trop de valeurs manquantes (>80%)
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > 0.8].index
        if len(cols_to_drop) > 0:
            df = df.drop(columns=cols_to_drop)
            print(f"Colonnes supprimées (>80% manquantes): {list(cols_to_drop)}")
        
        # 4. Détecter et traiter les outliers (IQR method)
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    print(f"Outliers détectés dans {col}: {outliers_count}")
                    # Remplacer par les limites
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        self.cleaned_df = df
        print(f"✅ Nettoyage terminé: {df.shape}")
        return df
    
    def feature_engineering(self):
        """Créer de nouvelles features"""
        print("\n⚡ Ingénierie des features...")
        
        df = self.cleaned_df.copy()
        
        # 1. Features temporelles si dates présentes
        date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
        for col in date_cols:
            if 'date' in col.lower() or 'dt' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                except:
                    pass
        
        # 2. Features d'agrégation si Provider présent
        if 'Provider' in df.columns:
            provider_stats = df.groupby('Provider').agg({
                df.select_dtypes(include=[np.number]).columns[0]: ['count', 'mean', 'sum']
            }).reset_index()
            provider_stats.columns = ['Provider', 'Provider_claim_count', 
                                    'Provider_avg_amount', 'Provider_total_amount']
            df = df.merge(provider_stats, on='Provider', how='left')
        
        # 3. Binning pour les variables continues
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Limiter aux 3 premières
            try:
                df[f'{col}_binned'] = pd.cut(df[col], bins=5, labels=['Low', 'Medium-Low', 
                                                                     'Medium', 'Medium-High', 'High'])
            except:
                pass
        
        # 4. Encoder les variables catégorielles
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() < 50:  # Seulement si pas trop de catégories
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        self.feature_df = df
        print(f"✅ Features créées: {df.shape}")
        return df
    
    def prepare_for_modeling(self, target_col=None):
        """Préparer les données pour le modeling"""
        print("\n🎯 Préparation pour le modeling...")
        
        df = self.feature_df.copy()
        
        # Si pas de target spécifiée, essayer de la détecter
        if target_col is None:
            # Chercher des colonnes qui pourraient être des targets
            potential_targets = [col for col in df.columns 
                               if any(word in col.lower() for word in 
                                     ['fraud', 'claim', 'target', 'label', 'outcome'])]
            if potential_targets:
                target_col = potential_targets[0]
                print(f"Target détectée automatiquement: {target_col}")
        
        if target_col and target_col in df.columns:
            # Séparer features et target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Sélectionner seulement les colonnes numériques pour X
            X = X.select_dtypes(include=[np.number])
            
            # Normaliser les features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y if y.nunique() < 10 else None
            )
            
            print(f"✅ Données préparées:")
            print(f"   X_train: {X_train.shape}")
            print(f"   X_test: {X_test.shape}")
            print(f"   Target: {target_col}")
            
            return X_train, X_test, y_train, y_test, X.columns.tolist()
        
        else:
            # Pas de target, préparer pour clustering ou analyse
            X = df.select_dtypes(include=[np.number])
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            print(f"✅ Données préparées pour analyse non supervisée: {X_scaled.shape}")
            return X_scaled, None, None, None, X.columns.tolist()
    
    def save_processed_data(self, output_dir="data/processed"):
        """Sauvegarder les données traitées"""
        os.makedirs(output_dir, exist_ok=True)
        
        if hasattr(self, 'feature_df'):
            self.feature_df.to_csv(f"{output_dir}/processed_data.csv", index=False)
            print(f"✅ Données sauvegardées dans {output_dir}/processed_data.csv")

def main():
    """Fonction principale pour tester le preprocessing"""
    preprocessor = InsuranceDataPreprocessor()
    
    # Chemins des fichiers (à adapter)
    bene_path = "data/raw/bene_file.csv"
    claims_path = "data/raw/Inpatient_Claim.csv"
    
    # Pipeline complet
    bene_df, claims_df = preprocessor.load_data(bene_path, claims_path)
    
    if bene_df is not None and claims_df is not None:
        preprocessor.explore_data()
        merged_df = preprocessor.merge_data()
        cleaned_df = preprocessor.clean_data()
        feature_df = preprocessor.feature_engineering()
        
        # Préparer pour modeling
        X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_for_modeling()
        
        # Sauvegarder
        preprocessor.save_processed_data()
        
        return preprocessor

if __name__ == "__main__":
    main()