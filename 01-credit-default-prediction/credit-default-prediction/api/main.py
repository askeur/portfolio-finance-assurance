from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse

app = FastAPI(title="Model Report Analyzer")

class ConfusionMatrix(BaseModel):
    matrix: List[List[int]]  # [[tn, fp], [fn, tp]]

class ReportInput(BaseModel):
    report_text: str
    confusion_matrix: ConfusionMatrix
    roc_auc: float

@app.post("/analyze")
def analyze_report(input: ReportInput):
    analysis = generate_analysis(input.report_text, input.confusion_matrix.matrix, input.roc_auc)
    return JSONResponse(content={"analysis": analysis})


def generate_analysis(report_text: str, confusion_matrix: List[List[int]], roc_auc: float) -> List[str]:
    lines = report_text.strip().split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    metrics = {}

    for line in lines[1:3]:  # classes 0 et 1
        parts = line.split()
        label = parts[0]
        precision, recall, f1, support = map(float, parts[1:])
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

    acc_line = lines[3]
    accuracy = float(acc_line.split()[1])
    tn, fp, fn, tp = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]

    output = []

    output.append(f"✅ Accuracy globale : {accuracy:.2%}")
    output.append(f"✅ ROC-AUC : {roc_auc:.3f}")
    
    output.append("\n🔍 Classe majoritaire (0) :")
    output.append(f"- Précision : {metrics['0']['precision']:.2f}")
    output.append(f"- Rappel    : {metrics['0']['recall']:.2f}")
    output.append(f"- F1-score  : {metrics['0']['f1']:.2f}")

    output.append("\n⚠️ Classe minoritaire (1 - défaut) :")
    output.append(f"- Précision : {metrics['1']['precision']:.2f}")
    output.append(f"- Rappel    : {metrics['1']['recall']:.2f}")
    output.append(f"- F1-score  : {metrics['1']['f1']:.2f}")


    if metrics['1']['recall'] < 0.4:
        output.append("❗ Le modèle détecte mal les défauts (rappel faible).")
        output.append("🔧 ➤ Envisagez un équilibrage ou ajustez le seuil.")

    if metrics['1']['precision'] < 0.5:
        output.append("❗ Beaucoup de faux positifs (précision faible).")

    if roc_auc > 0.8:
        output.append("✅ Bonne séparation globale entre classes.")
    elif roc_auc > 0.7:
        output.append("⚠️ Séparation correcte, mais améliorable.")
    else:
        output.append("❌ Mauvaise séparation : à améliorer.")

    output.append("\n📦 Matrice de confusion :")
    output.append(f" - Vrais négatifs  : {tn}")
    output.append(f" - Faux positifs   : {fp}")
    output.append(f" - Faux négatifs   : {fn}")
    output.append(f" - Vrais positifs  : {tp}")

    if fn > tp:
        output.append("❗ Beaucoup de défauts non détectés (FN > TP).")
    
    return output
