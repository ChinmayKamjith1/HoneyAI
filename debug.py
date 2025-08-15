import pandas as pd
import numpy as np
import re
from collections import Counter

def analyze_data(csv_path):
    """Analyze the raw data to understand overfitting issues"""
    print("Loading and analyzing data...")
    df = pd.read_csv(csv_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check label distribution
    print(f"\nLabel distribution:")
    label_counts = df['label'].value_counts()
    print(label_counts)
    
    # Look at some samples
    print(f"\nSample data from each class:")
    for label in df['label'].unique():
        samples = df[df['label'] == label]['tty_contents'].head(3)
        print(f"\n--- {label} ---")
        for i, sample in enumerate(samples):
            print(f"{i+1}: {str(sample)[:200]}...")
    
    # Check for duplicates
    print(f"\nDuplicate analysis:")
    total_samples = len(df)
    unique_content = df['tty_contents'].nunique()
    duplicate_rate = 1 - (unique_content / total_samples)
    print(f"Total samples: {total_samples}")
    print(f"Unique content: {unique_content}")
    print(f"Duplicate rate: {duplicate_rate:.4f}")
    
    if duplicate_rate > 0.1:
        print("⚠️  HIGH DUPLICATE RATE - This can cause overfitting!")
        
        # Find most common duplicates
        duplicates = df['tty_contents'].value_counts().head(10)
        print(f"\nMost frequent duplicates:")
        for content, count in duplicates.items():
            if count > 1:
                labels_for_content = df[df['tty_contents'] == content]['label'].unique()
                print(f"Count: {count}, Labels: {labels_for_content}")
                print(f"Content: {str(content)[:100]}...")
                print("-" * 50)
    
    # Check for label consistency
    print(f"\nChecking for same content with different labels:")
    content_label_map = {}
    inconsistent_count = 0
    
    for _, row in df.iterrows():
        content = row['tty_contents']
        label = row['label']
        
        if content in content_label_map:
            if content_label_map[content] != label:
                inconsistent_count += 1
                if inconsistent_count <= 5:  # Show first 5 examples
                    print(f"Content: {str(content)[:100]}...")
                    print(f"Labels: {content_label_map[content]} vs {label}")
                    print("-" * 30)
        else:
            content_label_map[content] = label
    
    print(f"Inconsistent labeling instances: {inconsistent_count}")
    
    # Analyze sequence lengths after normalization
    def normalize_tty(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'^[\w\-\@\.]+[:\$#]\s*', '', text)
        text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'#\s*note.*?(?=\n|$)', '', text)
        text = re.sub(r';\s*sleep\s+\d+', '', text)
        return text.strip()
    
    def word_tokenize(text):
        return re.findall(r'\S+', text)
    
    print(f"\nAnalyzing normalized content lengths:")
    df['normalized'] = df['tty_contents'].apply(normalize_tty)
    df['tokens'] = df['normalized'].apply(word_tokenize)
    df['token_count'] = df['tokens'].apply(len)
    
    print(f"Token count stats:")
    print(f"Mean: {df['token_count'].mean():.2f}")
    print(f"Median: {df['token_count'].median():.2f}")
    print(f"Min: {df['token_count'].min()}")
    print(f"Max: {df['token_count'].max()}")
    print(f"Std: {df['token_count'].std():.2f}")
    
    # Check for empty sequences after normalization
    empty_after_norm = df[df['token_count'] == 0]
    print(f"Empty sequences after normalization: {len(empty_after_norm)}")
    
    if len(empty_after_norm) > 0:
        print("Examples of sequences that became empty:")
        for i, row in empty_after_norm.head(3).iterrows():
            print(f"Original: {str(row['tty_contents'])[:100]}...")
            print(f"Normalized: '{row['normalized']}'")
            print("-" * 30)
    
    # Analyze vocabulary
    all_tokens = []
    for tokens in df['tokens']:
        all_tokens.extend(tokens)
    
    token_counter = Counter(all_tokens)
    print(f"\nVocabulary analysis:")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Unique tokens: {len(token_counter)}")
    print(f"Most common tokens: {token_counter.most_common(20)}")
    
    # Check for potential data leakage patterns
    print(f"\nChecking for potential data leakage patterns:")
    
    # Look for class-specific tokens
    class_tokens = {}
    for label in df['label'].unique():
        class_data = df[df['label'] == label]['tokens']
        class_all_tokens = []
        for tokens in class_data:
            class_all_tokens.extend(tokens)
        class_tokens[label] = Counter(class_all_tokens)
    
    # Find tokens that appear almost exclusively in one class
    for label, tokens in class_tokens.items():
        other_labels = [l for l in class_tokens.keys() if l != label]
        exclusive_tokens = []
        
        for token, count in tokens.most_common(50):
            total_other = sum(class_tokens[other_label].get(token, 0) for other_label in other_labels)
            if total_other == 0 and count > 10:  # Token appears only in this class
                exclusive_tokens.append((token, count))
        
        if exclusive_tokens:
            print(f"\nTokens exclusive to '{label}' class:")
            for token, count in exclusive_tokens[:10]:
                print(f"  '{token}': {count} times")

if __name__ == "__main__":
    csv_path = r"C:\Users\Faster\Downloads\Main_TTYs.csv"
    analyze_data(csv_path)
