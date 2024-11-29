import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = {
    "Product": [1, 2, 3, 4, 5],
    "Name": ["Bun", "Noodle", "pasta", "chips", "chocolate"],
    "Category": ["Bakery", "Processed food", "Processed food", "chips", "sweet"],
    "Company": ["Nestle.Ltd.", "Marico Ltd.", "Varun Industries Ltd.", "Jubliant FoodWorks Ltd.", "Britania Industries Ltd."],
}

df = pd.DataFrame(data)


df["Features"] = df["Category"] + " " + df["Company"]


vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(df["Features"])

# Calculate cosine similarity
cosine_sim = cosine_similarity(feature_vectors)

# Function to recommend movies based on a movie title
def recommend_product(product_name, num_recommendations=3):
    if product_name not in df["Name"].values:
        return "product not found in the database."
    
    # Get the index of the movie
    product_idx = df[df["Name"] == product_name].index[0]
    
    # Get similarity scores for all movies
    similarity_scores = list(enumerate(cosine_sim[product_idx]))
    
    # Sort by similarity score
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    recommendations = []
    for i in range(1, num_recommendations + 1):  # Skip the first movie (itself)
        product_idx = sorted_scores[i][0]
        recommendations.append(df.iloc[product_idx]["Name"])
    
    return recommendations

# Test the recommendation system
user_input = "Bun"
recommendations = recommend_product(user_input)
print(f"Product similar to '{user_input}': {recommendations}")
