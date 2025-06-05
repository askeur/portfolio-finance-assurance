import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class InsuranceDataVisualizer:
    def __init__(self):
        self.setup_style()
        
    def setup_style(self):
        """Configuration du style des graphiques"""
        # Utiliser un style disponible
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
        
        # Configuration matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
    def visualize_insurance_data(self, 
                               processed_data_path="data/processed/processed_data.csv",
                               bene_data_path="",
                               claims_data_path="",
                               output_dir="reports/figures"):
        """Fonction principale de visualisation des données d'assurance"""
        
        # Configuration des chemins
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        base_path = os.path.join(BASE_DIR,"02-insurance-claims-analysis" , "insurance-claims-analysis")
              
        if not os.path.isabs(processed_data_path):
            processed_data_path = os.path.join(base_path, processed_data_path)
        
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(base_path, output_dir)
            
        print(f"📊 Chemin des données: {processed_data_path}")
        print(f"📁 Dossier de sortie: {output_dir}")
        
        # Créer le dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Charger les données
            df = pd.read_csv(processed_data_path)
            print(f"✅ Données chargées: {df.shape}")
            
            # Vérifier la qualité des données
            self.check_data_quality(df)
            
            # 1. Vue d'ensemble des données
            self.create_data_overview(df, output_dir)
            
            # 2. Analyse des réclamations
            self.analyze_claims_patterns(df, output_dir)
            
            # 3. Analyse des bénéficiaires
            self.analyze_beneficiaries(df, output_dir)
            
            # 4. Analyse geographic
            self.geographic_analysis(df, output_dir)
            
            # 5. Corrélations et patterns
            self.correlation_analysis(df, output_dir)
            
        
            print(f"✅ Toutes les visualisations sauvegardées dans {output_dir}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la visualisation: {e}")
            
    
    def check_data_quality(self, df):
        """Vérifier la qualité des données"""
        print("\n📋 Vérification de la qualité des données:")
        print(f"   - Nombre de lignes: {len(df):,}")
        print(f"   - Nombre de colonnes: {len(df.columns)}")
        print(f"   - Valeurs manquantes: {df.isnull().sum().sum():,}")
        print(f"   - Doublons: {df.duplicated().sum():,}")
        
        # Colonnes avec beaucoup de valeurs manquantes
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 20]
        if not high_missing.empty:
            print("   - Colonnes avec >20% de valeurs manquantes:")
            for col, pct in high_missing.items():
                print(f"     • {col}: {pct:.1f}%")
    
    def create_data_overview(self, df, output_dir):
        """Vue d'ensemble des données"""
        print("📋 Création de la vue d'ensemble...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Distribution des montants de réclamation (adapté à CLM_PMT_AMT)
        if 'CLM_PMT_AMT' in df.columns:
            amounts = df['CLM_PMT_AMT']
            q95 = amounts.quantile(0.95)
            filtered_amounts = amounts[amounts <= q95]

            axes[0, 0].hist(filtered_amounts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Distribution des Montants de Réclamation (95e percentile)')
            axes[0, 0].set_xlabel('Montant ($)')
            axes[0, 0].set_ylabel('Fréquence')
            axes[0, 0].ticklabel_format(style='plain', axis='x')

        # 2. Distribution par genre - non présent dans les données, on saute cette partie
        axes[0, 1].axis('off')  # Vide

        # 3. Distribution des âges - non présent dans les données, on saute cette partie
        axes[1, 0].axis('off')  # Vide

        # 4. Top États - non présent dans les données, on saute cette partie
        axes[1, 1].axis('off')  # Vide

        plt.tight_layout()
        plt.savefig(f"{output_dir}/data_overview.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Statistiques descriptives sur les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            desc_stats = df[numeric_cols].describe()
            desc_stats.to_csv(f"{output_dir}/descriptive_statistics.csv")
            print(f"✅ Statistiques descriptives sauvegardées")
    

    def analyze_claims_patterns(self, df, output_dir):
        """Analyser les patterns de réclamations"""

        print("🏥 Analyse des patterns de réclamations...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Montants par provider (top 20)
        if 'PRVDR_NUM' in df.columns and 'CLM_PMT_AMT' in df.columns:
            provider_amounts = df.groupby('PRVDR_NUM')['CLM_PMT_AMT'].agg(['sum', 'mean', 'count']).reset_index()
            provider_amounts = provider_amounts.sort_values('sum', ascending=False).head(20)

            axes[0, 0].bar(range(len(provider_amounts)), provider_amounts['sum'], color='steelblue')
            axes[0, 0].set_title('Top 20 Providers par Montant Total')
            axes[0, 0].set_xlabel('Providers (Index)')
            axes[0, 0].set_ylabel('Montant Total ($)')
            axes[0, 0].ticklabel_format(style='plain', axis='y')

        # 2. Distribution de la durée d'hospitalisation (CLM_UTLZTN_DAY_CNT)
        if 'CLM_UTLZTN_DAY_CNT' in df.columns:
            los_filtered = df['CLM_UTLZTN_DAY_CNT'][df['CLM_UTLZTN_DAY_CNT'] <= df['CLM_UTLZTN_DAY_CNT'].quantile(0.95)]
            axes[0, 1].hist(los_filtered, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[0, 1].set_title("Distribution de la Durée d'Hospitalisation (95e percentile)")
            axes[0, 1].set_xlabel('Jours')
            axes[0, 1].set_ylabel('Fréquence')

        # 3. Relation montant vs durée de séjour
        if 'CLM_PMT_AMT' in df.columns and 'CLM_UTLZTN_DAY_CNT' in df.columns:
            sample_df = df.sample(min(1000, len(df)), random_state=42)
            q95_amount = sample_df['CLM_PMT_AMT'].quantile(0.95)
            q95_los = sample_df['CLM_UTLZTN_DAY_CNT'].quantile(0.95)
            filtered_sample = sample_df[
                (sample_df['CLM_PMT_AMT'] <= q95_amount) & 
                (sample_df['CLM_UTLZTN_DAY_CNT'] <= q95_los)
            ]

            axes[1, 0].scatter(filtered_sample['CLM_UTLZTN_DAY_CNT'], filtered_sample['CLM_PMT_AMT'], 
                            alpha=0.6, color='purple')
            axes[1, 0].set_title('Montant vs Durée d’Hospitalisation')
            axes[1, 0].set_xlabel('Durée d’Hospitalisation (jours)')
            axes[1, 0].set_ylabel('Montant de Réclamation ($)')
            axes[1, 0].ticklabel_format(style='plain', axis='y')

        # 4. Top diagnostics ICD9 (premier code diagnostic)
        if 'ICD9_DGNS_CD_1' in df.columns:
            diag_counts = df['ICD9_DGNS_CD_1'].value_counts().head(20)
            axes[1, 1].bar(range(len(diag_counts)), diag_counts.values, color='red', alpha=0.7)
            axes[1, 1].set_title('Top 20 Diagnostics ICD9 (1er code)')
            axes[1, 1].set_xlabel('Code ICD9')
            axes[1, 1].set_ylabel('Nombre de Cas')
            axes[1, 1].set_xticks(range(len(diag_counts)))
            axes[1, 1].set_xticklabels(diag_counts.index, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/claims_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_beneficiaries(self, df, output_dir):
                                
        """Analyser les bénéficiaires"""

        print("👥 Analyse des bénéficiaires...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Répartition par sexe
        if 'BENE_SEX_IDENT_CD' in df.columns:
            gender_map = {1: 'Homme', 2: 'Femme'}
            genders = df['BENE_SEX_IDENT_CD'].map(gender_map).value_counts()
            axes[0, 0].pie(genders.values, labels=genders.index,
                        autopct='%1.1f%%', colors=['lightblue', 'lightpink'])
            axes[0, 0].set_title('Répartition par Genre')

        # 2. Répartition par race
        if 'BENE_RACE_CD' in df.columns:
            race_map = {
                1: 'Blanc',
                2: 'Noir',
                3: 'Autre',
                4: 'Asiatique',
                5: 'Hispanique'
            }
            races = df['BENE_RACE_CD'].map(race_map).value_counts()
            axes[0, 1].bar(races.index, races.values, color='orchid')
            axes[0, 1].set_title('Répartition par Race')
            axes[0, 1].set_xlabel('Race')
            axes[0, 1].set_ylabel('Nombre de Bénéficiaires')
            axes[0, 1].tick_params(axis='x', rotation=30)

        # 3. Prévalence des conditions chroniques (SP_*)
        chronic_cols = [col for col in df.columns if col.startswith('SP_') and col not in ['SP_STATE_CODE']]
        if chronic_cols:
            # Calcul du taux de présence de la condition (1 = présent, 2 = absent)
            prevalence = ((df[chronic_cols] == 1).sum() / len(df)) * 100
            prevalence = prevalence.sort_values(ascending=False)

            sns.heatmap(prevalence.to_frame().T, ax=axes[1, 0], cmap='Reds',
                        annot=True, fmt=".1f", cbar_kws={'label': 'Prévalence (%)'})
            axes[1, 0].set_title('Prévalence des Conditions Chroniques')
            axes[1, 0].set_yticklabels([''])
            axes[1, 0].set_xticklabels([col.replace('SP_', '') for col in prevalence.index], rotation=45)

        # 4. Distribution du nombre de mois de couverture
        if 'Total_mons' in df.columns:
            axes[1, 1].hist(df['Total_mons'], bins=range(0, 49, 6), color='green', edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Distribution du Nombre de Mois de Couverture')
            axes[1, 1].set_xlabel('Nombre de Mois')
            axes[1, 1].set_ylabel('Nombre de Bénéficiaires')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/beneficiaries_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    
    def correlation_analysis(self, df, output_dir):
        """Analyse des corrélations"""
        print("📊 Analyse des corrélations...")

        # Sélectionner les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            print("⚠️ Pas assez de colonnes numériques pour l'analyse de corrélation")
            return

        # Limiter à 15 colonnes pour éviter des graphiques illisibles
        numeric_cols = numeric_cols[:15]

        # Matrice de corrélation
        plt.figure(figsize=(14, 12))
        corr_matrix = df[numeric_cols].corr()

        # Heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                    cmap='coolwarm', center=0, square=True,
                    linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('🔍 Matrice de Corrélation des Variables Numériques')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Pairplot (scatter matrix) pour les variables les plus importantes
        important_vars = numeric_cols[:6]  # Sélectionne les 6 premières
        if len(important_vars) >= 2:
            sample_size = min(1000, len(df))  # Échantillonner si trop de données
            sample_df = df[important_vars].sample(sample_size, random_state=42)

            sns.pairplot(sample_df, corner=True)
            plt.suptitle("🔗 Pairplot des Variables Clés (échantillon)", y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "pairplot_variables.png"), dpi=300)
            plt.close()
    
    def geographic_analysis(self, df, output_dir):
        print("🌍 Analyse géographique basée sur SP_STATE_CODE...")

        if 'SP_STATE_CODE' not in df.columns:
            print("⚠️ Colonne 'SP_STATE_CODE' non trouvée")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Nombre de réclamations par état (top 15)
        state_counts = df['SP_STATE_CODE'].value_counts().head(15)
        axes[0, 0].bar(state_counts.index, state_counts.values, color='steelblue')
        axes[0, 0].set_title("Top 15 États par Nombre de Réclamations")
        axes[0, 0].set_xlabel("État")
        axes[0, 0].set_ylabel("Nombre de Réclamations")
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Montant moyen des paiements par état (top 15)
        if 'CLM_PMT_AMT' in df.columns:
            avg_payment_by_state = df.groupby('SP_STATE_CODE')['CLM_PMT_AMT'].mean().sort_values(ascending=False).head(15)
            axes[0, 1].bar(avg_payment_by_state.index, avg_payment_by_state.values, color='orange')
            axes[0, 1].set_title("Montant Moyen Paiements par État")
            axes[0, 1].set_xlabel("État")
            axes[0, 1].set_ylabel("Montant Moyen ($)")
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, "Pas de colonne 'CLM_PMT_AMT'", ha='center', va='center')
            axes[0, 1].axis('off')

        # 3. Répartition par genre par état (top 10 états)
        if 'BENE_SEX_IDENT_CD' in df.columns:
            top_states = state_counts.head(10).index
            gender_counts = df[df['SP_STATE_CODE'].isin(top_states)].groupby(['SP_STATE_CODE', 'BENE_SEX_IDENT_CD']).size().unstack(fill_value=0)
            gender_counts.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='Pastel1')
            axes[1, 0].set_title("Répartition par Genre dans Top 10 États")
            axes[1, 0].set_xlabel("État")
            axes[1, 0].set_ylabel("Nombre de Réclamations")
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, "Pas de colonne 'BENE_SEX_IDENT_CD'", ha='center', va='center')
            axes[1, 0].axis('off')

        # 4. Présence de comorbidités : % avec SP_DIABETES par état (top 15)
        if 'SP_DIABETES' in df.columns:
            diabetes_rate = df.groupby('SP_STATE_CODE')['SP_DIABETES'].mean().sort_values(ascending=False).head(15) * 100
            axes[1, 1].bar(diabetes_rate.index, diabetes_rate.values, color='purple')
            axes[1, 1].set_title("Taux de Diabète (%) par État (Top 15)")
            axes[1, 1].set_xlabel("État")
            axes[1, 1].set_ylabel("Pourcentage (%)")
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, "Pas de colonne 'SP_DIABETES'", ha='center', va='center')
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/geographic_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Analyse géographique sauvegardée")

if __name__ == "__main__":
    # Définir les chemins
    processed_data_path = "data/processed/processed_data.csv"
    output_dir = "reports/figures"

    # Construire les chemins absolus
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    base_path = os.path.join(BASE_DIR,"02-insurance-claims-analysis" , "insurance-claims-analysis")
       

    if not os.path.isabs(processed_data_path):
        processed_data_path = os.path.join(base_path, processed_data_path)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_path, output_dir)

    print(f"📊 Chemin des données: {processed_data_path}")
    print(f"📁 Dossier de sortie: {output_dir}")

    # Créer le dossier de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données
    df = pd.read_csv(processed_data_path)
    print(f"✅ Données chargées: {df.shape}")

    # Initialiser le visualiseur
    visualizer = InsuranceDataVisualizer()
    visualizer.visualize_insurance_data(processed_data_path, output_dir)
    # Appel individuel 
    # visualizer.check_data_quality(df)
    # visualizer.create_data_overview(df, output_dir)
    # visualizer.analyze_claims_patterns(df, output_dir)
    # visualizer.analyze_beneficiaries(df, output_dir)
    # visualizer.temporal_analysis(df, output_dir)
    # visualizer.geographic_analysis(df, output_dir)
    # visualizer.correlation_analysis(df, output_dir)
