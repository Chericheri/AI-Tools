import spacy
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import sys

print("📝 Loading spaCy Model...")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)

# Sample Amazon reviews data
reviews = [
    "I love my new iPhone 14 Pro from Apple. The camera is amazing!",
    "Samsung Galaxy S23 has terrible battery life. Very disappointed.",
    "The Sony WH-1000XM4 headphones have excellent noise cancellation.",
    "Amazon Echo Dot is okay but the sound quality could be better.",
    "Microsoft Surface Laptop 5 is perfect for work and gaming.",
    "Google Pixel 7 Pro camera is fantastic but battery drains quickly.",
    "This HP laptop from Best Buy stopped working after 2 months.",
    "Dell XPS 13 is the best Windows laptop I've ever used.",
    "Nintendo Switch OLED is great for family gaming nights.",
    "Bose QuietComfort headphones are worth every penny!"
]

print(f"\n📊 Analyzing {len(reviews)} product reviews...")

# Perform NER and sentiment analysis
results = []

for i, review in enumerate(reviews):
    doc = nlp(review)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Simple rule-based sentiment
    positive_words = ['love', 'amazing', 'excellent', 'perfect', 'fantastic', 'great', 'best', 'worth']
    negative_words = ['terrible', 'disappointed', 'okay', 'could be better', 'drains', 'stopped working']
    
    positive_count = sum(1 for word in positive_words if word in review.lower())
    negative_count = sum(1 for word in negative_words if word in review.lower())
    
    if positive_count > negative_count:
        sentiment = "POSITIVE"
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    
    results.append({
        'review': review,
        'entities': entities,
        'sentiment': sentiment,
        'positive_words': positive_count,
        'negative_words': negative_count
    })

# Create DataFrame for results
df_results = pd.DataFrame(results)

print("\n🔍 Named Entity Recognition Results:")
for i, row in df_results.iterrows():
    print(f"\nReview {i+1}: {row['review']}")
    print(f"Sentiment: {row['sentiment']}")
    print(f"Entities: {row['entities']}")

# Analyze entity frequency
all_entities = []
for entities_list in df_results['entities']:
    all_entities.extend([ent[0] for ent in entities_list])

entity_counts = Counter(all_entities)
print(f"\n📈 Most Common Entities:")
for entity, count in entity_counts.most_common(5):
    print(f"  {entity}: {count} times")

# Visualization
plt.figure(figsize=(10, 6))

# Sentiment distribution
sentiment_counts = df_results['sentiment'].value_counts()
plt.subplot(1, 2, 1)
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
        colors=['#4CAF50', '#FF9800', '#F44336'])
plt.title('Sentiment Distribution')

# Top entities
plt.subplot(1, 2, 2)
top_entities = dict(entity_counts.most_common(5))
plt.bar(range(len(top_entities)), list(top_entities.values()))
plt.xticks(range(len(top_entities)), list(top_entities.keys()), rotation=45)
plt.title('Top 5 Named Entities')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('spacy_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Task 3 Complete! NLP analysis with spaCy finished!")