import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

class ProductSentimentAnalyzer:
    def __init__(self, data_path, text_column):
        self.data_path = data_path
        self.text_column = text_column
        self.df = None
        
        print("[*] Initializing Hugging Face Sentiment Pipeline...")
        # Using DistilBERT: Fast, accurate, and lightweight transformer
        self.nlp_pipeline = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1 # Set to 0 if running on a machine with a GPU
        )
        print("[+] Model loaded successfully.")

    def load_data(self):
        """Loads target CSV array and performs structural checks."""
        print(f"[*] Reading dataset from {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Target CSV dataset not found at: {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        print(f"[+] Loaded dataset containing {len(self.df)} rows.")
        
        if self.text_column not in self.df.columns:
            raise KeyError(f"Column '{self.text_column}' not found. Available: {list(self.df.columns)}")
            
        # Clean missing values out of target column
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=[self.text_column])
        self.df = self.df[self.df[self.text_column].str.strip() != ""]
        print(f"[!] Dropped {initial_len - len(self.df)} empty/null review rows.")

    def run_inference(self, batch_size=32):
        """Processes product reviews in batches to optimize RAM usage."""
        print(f"[*] Running sentiment inference in batches of {batch_size}...")
        reviews = self.df[self.text_column].astype(str).tolist()
        
        labels = []
        scores = []
        
        # Batching loop protects memory boundaries on large text arrays
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            # Truncation ensures strings exceeding 512 tokens don't crash the transformer
            results = self.nlp_pipeline(batch, truncation=True)
            
            for res in results:
                labels.append(res['label'])
                scores.append(res['score'])
                
        self.df['sentiment_prediction'] = labels
        self.df['confidence_score'] = scores
        print("[+] Inference engine processing complete.")

    def generate_analytics(self):
        """Prints statistical summaries and exports data plots."""
        print("\n=== SENTIMENT DISTRIBUTION SUMMARY ===")
        distribution = self.df['sentiment_prediction'].value_counts()
        percentages = self.df['sentiment_prediction'].value_counts(normalize=True) * 100
        
        for label in distribution.index:
            print(f"{label}: {distribution[label]} reviews ({percentages[label]:.2f}%)")
            
        # Plot Distribution Chart
        plt.figure(figsize=(7, 5))
        sns.countplot(data=self.df, x='sentiment_prediction', palette='coolwarm')
        plt.title('Product Review Sentiment Breakdown')
        plt.xlabel('Predicted Sentiment')
        plt.ylabel('Review Count')
        
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/sentiment_distribution.png', dpi=300)
        print("\n[+] Visualization saved to plots/sentiment_distribution.png")

    def save_results(self, output_path):
        """Exports the expanded dataset back to a clean CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print(f"[+] Annotated execution sheet exported to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-driven Product Review Sentiment Analyzer")
    parser.add_argument('--data_path', type=str, required=True, help='Path to your input review CSV file')
    parser.add_argument('--text_col', type=str, required=True, help='The column name containing the review text strings')
    parser.add_argument('--output_path', type=str, default='data/analyzed_reviews.csv', help='Export destination for final CSV file')
    
    args = parser.parse_args()
    
    analyzer = ProductSentimentAnalyzer(data_path=args.data_path, text_column=args.text_col)
    analyzer.load_data()
    analyzer.run_inference()
    analyzer.generate_analytics()
    analyzer.save_results(output_path=args.output_path)