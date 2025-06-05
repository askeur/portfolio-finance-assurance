import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, adjusted_rand_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class InsuranceModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.model_results = {}
        
    def load_processed_data(self, data_path="data/processed/processed_data.csv"):
        """Charger les données preprocessées"""
        print("📊 Chargement des données preprocessées...")
        
        try:
            self.df = pd.read_csv(data_path)
            print(f"✅ Données chargées: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return None
    
    def setup_classification_models(self):
        """Initialiser les modèles de classification"""
        print("\n🤖 Configuration des modèles de classification...")
        
        self.classification_models = {
            'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Grilles de paramètres pour optimisation
        self.param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        print(f"✅ {len(self.classification_models)} modèles configurés")
    
    def setup_clustering_models(self):
        """Initialiser les modèles de clustering"""
        print("\n🎯 Configuration des modèles de clustering...")
        
        self.clustering_models = {
            'KMeans': KMeans(random_state=42, n_init=10),
            'DBSCAN': DBSCAN()
        }
        
        self.clustering_params = {
            'KMeans': {
                'n_clusters': [2, 3, 4, 5, 6, 7, 8],
                'init': ['k-means++', 'random']
            },
            'DBSCAN': {
                'eps': [0.5, 1.0, 1.5, 2.0],
                'min_samples': [3, 5, 10, 15]
            }
        }
        
        print(f"✅ {len(self.clustering_models)} modèles de clustering configurés")
    
    def train_classification_models(self, X_train, X_test, y_train, y_test, optimize=True):
        """Entraîner les modèles de classification"""
        print("\n🚀 Entraînement des modèles de classification...")
        
        results = {}
        
        for name, model in self.classification_models.items():
            print(f"\n📈 Entraînement {name}...")
            
            try:
                if optimize and name in self.param_grids:
                    # Optimisation avec GridSearch
                    print(f"   🔍 Optimisation des hyperparamètres...")
                    grid_search = GridSearchCV(
                        model, self.param_grids[name], 
                        cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    print(f"   ✅ Meilleurs paramètres: {grid_search.best_params_}")
                else:
                    # Entraînement standard
                    best_model = model
                    best_model.fit(X_train, y_train)
                
                # Prédictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
                
                # Métriques
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                results[name] = {
                    'model': best_model,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"   ✅ {name} - F1: {metrics['f1']:.3f}, Accuracy: {metrics['accuracy']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Erreur avec {name}: {e}")
                continue
        
        self.classification_results = results
        return results
    
    def train_clustering_models(self, X, optimize=True):
        """Entraîner les modèles de clustering"""
        print("\n🎯 Entraînement des modèles de clustering...")
        
        results = {}
        
        for name, model in self.clustering_models.items():
            print(f"\n📊 Clustering avec {name}...")
            
            try:
                best_score = -1
                best_model = None
                best_params = None
                
                if optimize and name in self.clustering_params:
                    # Test différents paramètres
                    for param_combo in self._generate_param_combinations(self.clustering_params[name]):
                        temp_model = self.clustering_models[name].__class__(**param_combo, random_state=42)
                        
                        try:
                            labels = temp_model.fit_predict(X)
                            
                            if len(np.unique(labels)) > 1:  # Au moins 2 clusters
                                score = silhouette_score(X, labels)
                                
                                if score > best_score:
                                    best_score = score
                                    best_model = temp_model
                                    best_params = param_combo
                        except:
                            continue
                else:
                    # Paramètres par défaut
                    best_model = model
                    labels = best_model.fit_predict(X)
                    if len(np.unique(labels)) > 1:
                        best_score = silhouette_score(X, labels)
                
                if best_model is not None:
                    final_labels = best_model.fit_predict(X)
                    
                    results[name] = {
                        'model': best_model,
                        'labels': final_labels,
                        'silhouette_score': best_score,
                        'n_clusters': len(np.unique(final_labels)),
                        'best_params': best_params
                    }
                    
                    print(f"   ✅ {name} - Silhouette: {best_score:.3f}, Clusters: {len(np.unique(final_labels))}")
                
            except Exception as e:
                print(f"   ❌ Erreur avec {name}: {e}")
                continue
        
        self.clustering_results = results
        return results
    
    def _generate_param_combinations(self, param_grid):
        """Générer les combinaisons de paramètres"""
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in product(*values):
            yield dict(zip(keys, combination))
    
    def evaluate_models(self, X_test, y_test):
        """Évaluer et comparer les modèles"""
        print("\n📊 Évaluation des modèles...")
        
        if hasattr(self, 'classification_results'):
            # Comparaison des modèles de classification
            comparison_data = []
            
            for name, result in self.classification_results.items():
                metrics = result['metrics']
                comparison_data.append({
                    'Model': name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1'],
                    'ROC-AUC': metrics.get('roc_auc', 'N/A'),
                    'CV Mean': metrics['cv_mean'],
                    'CV Std': metrics['cv_std']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            print("\n🏆 Comparaison des modèles de classification:")
            print(comparison_df.round(3))
            
            # Sélectionner le meilleur modèle
            best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
            self.best_model = self.classification_results[best_model_name]['model']
            self.best_model_name = best_model_name
            
            print(f"\n🥇 Meilleur modèle: {best_model_name}")
            
            return comparison_df
    
    def generate_detailed_report(self, X_test, y_test, output_dir="reports"):
        """Générer un rapport détaillé"""
        print("\n📋 Génération du rapport détaillé...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if hasattr(self, 'best_model') and self.best_model is not None:
            # Rapport de classification
            y_pred = self.best_model.predict(X_test)
            
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            
            # Sauvegarder le rapport
            report_df.to_csv(f"{output_dir}/classification_report.csv")
            
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Matrice de Confusion - {self.best_model_name}')
            plt.ylabel('Vrai Label')
            plt.xlabel('Prédiction')
            plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance si disponible
            if hasattr(self.best_model, 'feature_importances_'):
                feature_names = [f'Feature_{i}' for i in range(len(self.best_model.feature_importances_))]
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                plt.figure(figsize=(12, 8))
                sns.barplot(data=importance_df.head(20), x='importance', y='feature')
                plt.title('Top 20 Feature Importances')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
            
            print(f"✅ Rapport sauvegardé dans {output_dir}/")
    
    def save_models(self, output_dir="models"):
        """Sauvegarder les modèles entraînés"""
        print("\n💾 Sauvegarde des modèles...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if hasattr(self, 'best_model') and self.best_model is not None:
            model_path = f"{output_dir}/best_model_{self.best_model_name}.joblib"
            joblib.dump(self.best_model, model_path)
            print(f"✅ Meilleur modèle sauvegardé: {model_path}")
        
        # Sauvegarder tous les modèles de classification
        if hasattr(self, 'classification_results'):
            for name, result in self.classification_results.items():
                model_path = f"{output_dir}/model_{name}.joblib"
                joblib.dump(result['model'], model_path)
            print(f"✅ {len(self.classification_results)} modèles de classification sauvegardés")
        
        # Sauvegarder les modèles de clustering
        if hasattr(self, 'clustering_results'):
            for name, result in self.clustering_results.items():
                model_path = f"{output_dir}/clustering_{name}.joblib"
                joblib.dump(result['model'], model_path)
            print(f"✅ {len(self.clustering_results)} modèles de clustering sauvegardés")
    
    def predict_new_data(self, X_new):
        """Faire des prédictions sur de nouvelles données"""
        if hasattr(self, 'best_model') and self.best_model is not None:
            predictions = self.best_model.predict(X_new)
            probabilities = None
            
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X_new)
            
            return predictions, probabilities
        else:
            print("❌ Aucun modèle entraîné disponible")
            return None, None

def main():
    """Fonction principale pour tester l'entraînement"""
    trainer = InsuranceModelTrainer()
    
    # Charger les données (à adapter selon votre preprocessing)
    data_path = "data/processed/processed_data.csv"
    df = trainer.load_processed_data(data_path)
    
    if df is not None:
        # Exemple d'utilisation avec des données factices
        print("⚠️ Utilisation de données factices pour la démonstration")
        
        # Créer des données d'exemple
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = np.random.choice([0, 1], 1000)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Classification
        trainer.setup_classification_models()
        classification_results = trainer.train_classification_models(X_train, X_test, y_train, y_test)
        
        # Clustering
        trainer.setup_clustering_models()
        clustering_results = trainer.train_clustering_models(X)
        
        # Évaluation
        comparison_df = trainer.evaluate_models(X_test, y_test)
        
        # Rapport et sauvegarde
        trainer.generate_detailed_report(X_test, y_test)
        trainer.save_models()
        
        return trainer

if __name__ == "__main__":
    main()