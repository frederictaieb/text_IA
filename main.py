import spacy
import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

folder_path = "txt"

def text_segmentation(path=f"{folder_path}/input.txt"):
    nlp = spacy.load("en_core_web_md")
    with open(path, "r", encoding="utf-8") as f:
        texte = f.read()
    doc = nlp(texte)
    phrases = [sent.text.strip() for sent in doc.sents]
    with open(f"{folder_path}/output.txt", "w", encoding="utf-8") as out:
        for i, phrase in enumerate(phrases, 1):
            out.write(f"{phrase}\n")
                      
def text_emotionnal_traduction(path=f"{folder_path}/output.txt"):
    # 1. Pipeline d'analyse émotionnelle
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=7)

    json_lines = []

    # 2. Lecture ligne par ligne
    with open(path, "r", encoding="utf-8") as infile, open(f"{folder_path}/output_eat.jsonl", "w", encoding="utf-8") as outfile_jsonl:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            results = classifier(line)[0]  # Liste des 7 émotions

            json_line = {
                "text": line,
                "emotions": [{"label": res["label"], "score": round(res["score"], 4)} for res in results]
            }

            # Écriture ligne JSON
            outfile_jsonl.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            json_lines.append(json_line)

    # 3. Écriture du fichier JSON complet (tableau)
    with open(f"{folder_path}/output_eat_full.json", "w", encoding="utf-8") as outfile_full:
        json.dump(json_lines, outfile_full, ensure_ascii=False, indent=2)


def plot_emotion_graph(json_path=f"{folder_path}/output_eat_full.json"):
    # Charger les données JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convertir en DataFrame
    records = []
    for i, item in enumerate(data):
        for emo in item['emotions']:
            records.append({
                "index": i,
                "text": item["text"],
                "emotion": emo["label"],
                "score": emo["score"]
            })

    df = pd.DataFrame(records)
    pivot_df = df.pivot(index="index", columns="emotion", values="score")

    # Tracer le graphique
    plt.figure(figsize=(14, 6))
    for emotion in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[emotion], label=emotion)
    
    plt.title("Évolution des émotions au fil des phrases")
    plt.xlabel("Phrase n°")
    plt.ylabel("Score d'émotion")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    text_segmentation()
    text_emotionnal_traduction()
    plot_emotion_graph()

