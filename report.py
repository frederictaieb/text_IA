import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from pprint import pprint

# Dictionnaire des couleurs HEX pour chaque émotion
EMOTION_COLORS = {
    'disgust': '#00FF00',   # vert
    'surprise': '#4B0082', # indigo
    'joy': '#FFFF00',      # jaune
    'sadness': '#0000FF',  # bleu
    'fear': '#FFA500',     # orange
    'anger': '#FF0000',    # rouge
    'neutral': '#800080'   # violet
}

EMOTION_ORDER = ['neutral', 'surprise', 'sadness', 'disgust', 'joy', 'fear', 'anger']  # du haut vers le bas

# Charger les données depuis un fichier JSONL
def load_emotion_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

# Convertir en DataFrame structuré
def process_data(data):
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

# Calculs principaux
def analyze_emotions(df):
    pivot = df.pivot(index="index", columns="emotion", values="score").fillna(0)
    pivot["dominant"] = pivot.idxmax(axis=1)
    return pivot

# Visualisation 1 : Distribution globale
def plot_global_distribution(df):
    dominant_counts = df["dominant"].value_counts()
    dominant_counts.plot(kind="bar", title="Répartition des émotions dominantes")
    plt.ylabel("Nombre de phrases")
    plt.xlabel("Émotions dominantes")
    plt.tight_layout()
    plt.show()

# Visualisation 2 : Courbe temporelle
def plot_emotion_curves(df):
    df.drop(columns="dominant").plot(
        figsize=(14, 6), title="Évolution temporelle des émotions", alpha=0.8
    )
    plt.xlabel("Index (ordre du texte)")
    plt.ylabel("Score d'émotion")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def interpolate_color(color_hex, score, pale_factor=0.8):
    color_rgb = np.array(mcolors.to_rgb(color_hex))
    white_rgb = np.array([1, 1, 1])
    pale_rgb = white_rgb * pale_factor + color_rgb * (1 - pale_factor)
    # Interpolation entre pale_rgb (score=0) et color_rgb (score=1)
    result_rgb = pale_rgb * (1 - score) + color_rgb * score
    return result_rgb

# Visualisation 3 : Heatmap améliorée
def plot_emotion_heatmap(df):
    # On retire la colonne "dominant" si elle existe
    if "dominant" in df.columns:
        data = df.drop(columns="dominant")
    else:
        data = df
    emotions = list(data.columns)
    n_emotions = len(emotions)
    n_phrases = data.shape[0]
    # Créer une matrice RGB (émotions, phrases, 3)
    img = np.ones((n_emotions, n_phrases, 3))  # initialisé à blanc
    for i, emo in enumerate(emotions):
        color = EMOTION_COLORS.get(emo, '#000000')
        for j in range(n_phrases):
            score = data.iloc[j][emo]
            img[i, j, :] = interpolate_color(color, score)
    plt.figure(figsize=(14, 6))
    plt.imshow(img, aspect='auto')
    plt.title("Heatmap des émotions par phrase (couleurs personnalisées)")
    plt.xlabel("Index (ordre du texte)")
    plt.ylabel("Émotions")
    plt.yticks(np.arange(n_emotions), emotions)
    plt.tight_layout()
    plt.show()

def save_emotion_heatmap(df, output_path="emotion_heatmap.png"):
    plt.figure(figsize=(16, 6))
    # On retire la colonne "dominant" si elle existe
    if "dominant" in df.columns:
        data = df.drop(columns="dominant")
    else:
        data = df
    emotions = [e for e in EMOTION_ORDER if e in data.columns]
    data = data[emotions]
    n_emotions = len(emotions)
    n_phrases = data.shape[0]
    # Créer une matrice RGB (émotions, phrases, 3)
    img = np.ones((n_emotions, n_phrases, 3))  # initialisé à blanc
    for i, emo in enumerate(emotions):
        color = EMOTION_COLORS.get(emo, '#000000')
        for j in range(n_phrases):
            score = data.iloc[j][emo]
            img[i, j, :] = interpolate_color(color, score)
    plt.imshow(img, aspect='auto')
    plt.title("Heatmap des émotions par segment (couleurs personnalisées)")
    plt.xlabel("Index (ordre du texte)")
    plt.ylabel("Émotions")
    plt.yticks(np.arange(n_emotions), emotions)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Heatmap sauvegardée sous {output_path}")

def save_raw_heatmap(df, output_path="emotion_heatmap_raw.png"):
    # On retire la colonne "dominant" si elle existe
    if "dominant" in df.columns:
        data = df.drop(columns="dominant")
    else:
        data = df
    emotions = [e for e in EMOTION_ORDER if e in data.columns]
    data = data[emotions]
    n_emotions = len(emotions)
    n_phrases = data.shape[0]
    # Créer une matrice RGB (émotions, phrases, 3)
    img = np.ones((n_emotions, n_phrases, 3))  # initialisé à blanc
    for i, emo in enumerate(emotions):
        color = EMOTION_COLORS.get(emo, '#000000')
        for j in range(n_phrases):
            score = data.iloc[j][emo]
            img[i, j, :] = interpolate_color(color, score)
    plt.imsave(output_path, img)
    print(f"Image brute sauvegardée sous {output_path}")

# Exécution complète
def main():
    filename = "txt/output_eat.jsonl"  # ou .json si liste d'objets JSON
    data = load_emotion_data(filename)
    pprint(data)
    df_long = process_data(data)
    pprint(df_long)
    df_wide = analyze_emotions(df_long)
    pprint(df_wide)
    save_raw_heatmap(df_wide, output_path="emotion_heatmap_raw.png")
    #save_emotion_heatmap(df_wide, output_path="emotion_heatmap.png")
    #plot_global_distribution(df_wide)
    #plot_emotion_curves(df_wide)
    plot_emotion_heatmap(df_wide)

if __name__ == "__main__":
    main()
