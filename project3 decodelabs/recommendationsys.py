import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Item Dataset (Movies/Shows with descriptive metadata)
data = {
    "title": [
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Pulp Fiction",
        "The Matrix",
        "Forrest Gump",
        "Blade Runner 2049",
        "The Shawshank Redemption",
        "The Prestige",
        "Fight Club"
    ],
    "genres": [
        "Sci-Fi Action Thriller",
        "Sci-Fi Adventure Drama",
        "Action Crime Drama",
        "Crime Drama Dark-Comedy",
        "Sci-Fi Action Cyberpunk",
        "Drama Romance Comedy",
        "Sci-Fi Mystery Cyberpunk",
        "Drama Crime",
        "Mystery Sci-Fi Thriller Drama",
        "Drama Thriller Psychological"
    ],
    "keywords": [
        "subconscious heist dreams reality mind-bending",
        "space travel wormhole relativity black-hole time",
        "gotham vigilante justice joker crime-fighter",
        "gangster hitman nonlinear witty dialogue heist",
        "simulation virtual-reality rebellion ai machines",
        "life journey running love history innocence",
        "replicant future identity memory detective",
        "prison escape hope friendship redemption wrongful-conviction",
        "magicians rivalry illusion obsession secret",
        "underground rebellion consumerism alter-ego chaos"
    ]
}

df = pd.DataFrame(data)

# Combine relevant attributes into a single feature string
df["features"] = df["genres"] + " " + df["keywords"]

def recommend(user_query: str, top_n: int = 3):
    """
    Computes cosine similarity between user query and item features.
    """
    # 2. Vectorize text data
    tfidf = TfidfVectorizer(stop_words="english")
    
    # Fit TF-IDF on all items, then transform the item features
    item_vectors = tfidf.fit_transform(df["features"])
    
    # Transform the user input query into the same vector space
    user_vector = tfidf.transform([user_query])
    
    # 3. Compute Cosine Similarity
    similarity_scores = cosine_similarity(user_vector, item_vectors).flatten()
    
    # 4. Rank items based on similarity
    ranked_indices = similarity_scores.argsort()[::-1]
    
    results = []
    for idx in ranked_indices[:top_n]:
        score = similarity_scores[idx]
        if score > 0:  # Only include items with non-zero similarity match
            results.append({
                "Title": df.iloc[idx]["title"],
                "Genres": df.iloc[idx]["genres"],
                "Similarity Score": f"{score:.2%}"
            })
            
    return pd.DataFrame(results)


# --- Interactive CLI Demonstration ---
if __name__ == "__main__":
    print("=" * 60)
    print("DecodeLabs AI Project 3: Recommendation Engine")
    print("=" * 60)
    
    user_input = input("\nEnter your preferred genres or themes (e.g., 'space sci-fi time travel'):\n> ")
    
    recommendations = recommend(user_input, top_n=3)
    
    print("\nTop Recommended Items for You:")
    if not recommendations.empty:
        print(recommendations.to_string(index=False))
    else:
        print("No matching items found for your preferences. Try broader keywords.")