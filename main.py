import spacy
import json
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from pprint import pprint
import logging
import numpy as np
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO)

# Emotion color and order definitions (from report.py)
EMOTION_COLORS = {
    'disgust': '#00FF00',   # vert
    'surprise': '#4B0082', # indigo
    'joy': '#FFFF00',      # jaune
    'sadness': '#0000FF',  # bleu
    'fear': '#FFA500',     # orange
    'anger': '#FF0000',    # rouge
    'neutral': '#800080'   # violet
}

EMOTION_ORDER = ['neutral', 'surprise', 'sadness', 'disgust', 'joy', 'fear', 'anger']

def text_analysis(path=f"txt/input.txt"):

    nlp = spacy.load("en_core_web_md")
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=7)
    json_lines = []

    logging.info(f"Processing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        texte = f.read()
    doc = nlp(texte)
    phrases = [sent.text.strip() for sent in doc.sents]
    logging.info(f"Found {len(phrases)} phrases")

    logging.info(f"Processing phrases")
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue

        results = classifier(phrase)[0]
        json_line = {
            "text": phrase,
            "emotions": [{"label": res["label"], "score": res["score"]} for res in results]
        }
        json_lines.append(json_line)
    
    logging.info(f"Writing output.json")
    with open(f"txt/output.json", "w", encoding="utf-8") as outfile:     
        json.dump(json_lines, outfile, ensure_ascii=False, indent=2)

    logging.info(f"Done")

def process_data_json(data):
    rows = []
    for i, entry in enumerate(data):
        text = entry["text"]
        for emo in entry["emotions"]:
            rows.append({
                "index": i,
                "text": text,
                "emotion": emo["label"],
                "score": emo["score"]
            })
    df = pd.DataFrame(rows)
    return df

def analyze_emotions(df):
    pivot = df.pivot(index="index", columns="emotion", values="score").fillna(0)
    pivot["dominant"] = pivot.idxmax(axis=1)
    return pivot

def interpolate_color(color_hex, score, pale_factor=0.8):
    color_rgb = np.array(mcolors.to_rgb(color_hex))
    white_rgb = np.array([1, 1, 1])
    pale_rgb = white_rgb * pale_factor + color_rgb * (1 - pale_factor)
    result_rgb = pale_rgb * (1 - score) + color_rgb * score
    return result_rgb

def save_heatmap(df, output_path):
    if "dominant" in df.columns:
        data = df.drop(columns="dominant")
    else:
        data = df
    emotions = [e for e in EMOTION_ORDER if e in data.columns]
    data = data[emotions]
    n_emotions = len(emotions)
    n_phrases = data.shape[0]
    img = np.ones((n_emotions, n_phrases, 3))
    for i, emo in enumerate(emotions):
        color = EMOTION_COLORS.get(emo, '#000000')
        for j in range(n_phrases):
            score = data.iloc[j][emo]
            img[i, j, :] = interpolate_color(color, score)
    plt.imsave(output_path, img)
    logging.info(f"Heatmap saved")

def make_heatmap_from_output_json(path="txt/output.json", output_path="heatmap.png"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df_long = process_data_json(data)
    df_wide = analyze_emotions(df_long)
    save_heatmap(df_wide, output_path)


if __name__ == "__main__":
    text_analysis()
    make_heatmap_from_output_json()


