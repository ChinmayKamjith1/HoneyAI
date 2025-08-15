import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Config for ultra-clean data (expect MUCH lower accuracy!)
MAX_SEQ_LEN = 50  # Very short sequences
BATCH_SIZE = 80   # Smaller batch size
EPOCHS = 80       # More epochs for harder learning
EMBED_DIM = 64    # Smaller embeddings
HIDDEN_DIM = 64   # Smaller hidden size
LEARNING_RATE = 0.001  # Higher learning rate
DROPOUT = 0.3     # Lower dropout (less overfitting needed)
SAVE_PATH = r"C:\Users\Faster\Downloads\Autonomous\honeyai_lstm_realistic.pt"

# Expected accuracy ranges for ultra-clean data
EXPECTED_ACCURACY = {
    'excellent': 0.4,   # 40% would be excellent for 4-class with minimal features
    'good': 0.3,        # 30% is good (much better than 25% random)
    'acceptable': 0.25, # 25% is random baseline for 4 classes
}

class CommandDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    
    seq_lengths = [min(len(seq), MAX_SEQ_LEN) for seq in sequences]
    max_len = max(seq_lengths) if seq_lengths else 1
    
    padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded_seqs[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
    
    labels = torch.stack(labels)
    return padded_seqs, labels

class RealisticLSTMClassifier(nn.Module):
    """Simpler LSTM for realistic performance on ultra-clean data"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        
        # Simpler architecture to prevent overfitting on minimal features
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(dropout)
        
        # Single LSTM layer (not bidirectional to reduce overfitting)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Simple classification head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Simple weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.normal_(param, 0, 0.1)  # Small random weights
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
    def forward(self, x):
        # Simple forward pass
        embedded = self.embedding(x)
        embedded = self.dropout1(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state (simple approach)
        last_hidden = hidden[-1]  # Take last layer's hidden state
        
        output = self.dropout2(last_hidden)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout3(output)
        output = self.fc2(output)
        
        return output

def evaluate_model(model, val_loader, criterion, device, label_encoder, verbose=True):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for seqs, labs in val_loader:
            seqs, labs = seqs.to(device), labs.to(device)
            outputs = model(seqs)
            loss = criterion(outputs, labs)
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    
    if verbose:
        target_names = label_encoder.classes_
        print("\nValidation Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print("Predicted ->")
        print(f"{'':>12}", end="")
        for class_name in target_names:
            print(f"{class_name:>12}", end="")
        print()
        
        for i, class_name in enumerate(target_names):
            print(f"{class_name:>12}", end="")
            for j in range(len(target_names)):
                print(f"{cm[i][j]:>12}", end="")
            print()
    
    return avg_loss, accuracy, all_preds, all_labels

def assess_performance(accuracy, num_classes=4):
    """Assess if performance is realistic given the ultra-clean data"""
    random_baseline = 1.0 / num_classes
    
    if accuracy > 0.8:
        return "🚨 SUSPICIOUS - Too high! Likely data leakage still present"
    elif accuracy > 0.6:
        return "⚠️  HIGH - Good performance, but double-check for leakage"
    elif accuracy > random_baseline + 0.1:
        return "✅ REALISTIC - Better than random, achievable with clean data"
    elif accuracy > random_baseline:
        return "😐 MARGINAL - Slightly better than random"
    else:
        return "❌ POOR - At or below random baseline"

def cross_validate(sequences, labels, vocab_size, num_classes, device, label_encoder):
    """Cross-validation to get more reliable performance estimates"""
    print("\n" + "="*60)
    print("RUNNING 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    cv_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\nFold {fold + 1}/5:")
        
        # Split data
        X_train = [sequences[i] for i in train_idx]
        X_val = [sequences[i] for i in val_idx]
        y_train = [labels[i] for i in train_idx]
        y_val = [labels[i] for i in val_idx]
        
        # Create datasets
        train_dataset = CommandDataset(X_train, y_train)
        val_dataset = CommandDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        # Create model
        model = RealisticLSTMClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes, DROPOUT).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Quick training (fewer epochs for CV)
        best_val_acc = 0
        for epoch in range(20):
            model.train()
            for seqs, labs in train_loader:
                seqs, labs = seqs.to(device), labs.to(device)
                optimizer.zero_grad()
                outputs = model(seqs)
                loss = criterion(outputs, labs)
                loss.backward()
                optimizer.step()
            
            # Validate
            val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device, label_encoder, verbose=False)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        cv_accuracies.append(best_val_acc)
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
        assessment = assess_performance(best_val_acc, num_classes)
        print(f"  Assessment: {assessment}")
    
    mean_acc = np.mean(cv_accuracies)
    std_acc = np.std(cv_accuracies)
    
    print(f"\nCross-validation results:")
    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Individual folds: {[f'{acc:.4f}' for acc in cv_accuracies]}")
    
    overall_assessment = assess_performance(mean_acc, num_classes)
    print(f"Overall assessment: {overall_assessment}")
    
    return mean_acc, std_acc

def main():
    # Load ultra-clean data
    data_path = r"C:\Users\Faster\Downloads\Autonomous\preprocessed_data_ultra_clean.npz"
    label_enc_path = r"C:\Users\Faster\Downloads\Autonomous\preprocessed_data_ultra_clean_label_encoder.pkl"
    vocab_path = r"C:\Users\Faster\Downloads\Autonomous\preprocessed_data_ultra_clean_vocab.pkl"
    debug_path = r"C:\Users\Faster\Downloads\Autonomous\preprocessed_data_ultra_clean_debug.pkl"
    
    print("Loading ultra-clean data...")
    data_npz = np.load(data_path, allow_pickle=True)
    sequences_padded = data_npz['sequences']
    labels = data_npz['labels']
    sequence_lengths = data_npz.get('sequence_lengths', None)
    
    # Convert back to list of lists
    if sequence_lengths is not None:
        sequences = []
        for i, length in enumerate(sequence_lengths):
            sequences.append(sequences_padded[i][:length].tolist())
    else:
        sequences = []
        for seq in sequences_padded:
            seq_list = seq.tolist()
            last_nonzero = len(seq_list) - 1
            while last_nonzero >= 0 and seq_list[last_nonzero] == 0:
                last_nonzero -= 1
            sequences.append(seq_list[:last_nonzero + 1])
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Load vocabulary and label encoder
    with open(label_enc_path, 'rb') as f:
        label_encoder = pickle.load(f)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Load and display debug info
    try:
        with open(debug_path, 'rb') as f:
            debug_info = pickle.load(f)
        
        print(f"\nDebug info:")
        stats = debug_info.get('stats', {})
        print(f"  Uniqueness ratio: {stats.get('uniqueness_ratio', 'N/A'):.4f}")
        print(f"  Unknown token rate: {stats.get('unk_rate', 'N/A'):.4f}")
        print(f"  Max sequence length: {stats.get('max_seq_len', 'N/A')}")
        print(f"  Has potential leakage: {stats.get('has_potential_leakage', 'N/A')}")
        
        if stats.get('has_potential_leakage', False):
            print("🚨 WARNING: Preprocessing detected potential data leakage!")
            print("   The model may still achieve unrealistic accuracy.")
        
        print(f"\nTop vocabulary tokens: {list(vocab.keys())[2:12]}")  # Skip PAD and UNK
        
    except FileNotFoundError:
        print("Debug file not found - continuing without debug info")
    
    # Check data quality
    unique_sequences = len(set(str(seq) for seq in sequences))
    uniqueness_ratio = unique_sequences / len(sequences)
    print(f"\nSequence statistics:")
    print(f"Unique sequences: {unique_sequences} out of {len(sequences)} total")
    print(f"Uniqueness ratio: {uniqueness_ratio:.4f}")
    
    if uniqueness_ratio < 0.5:
        print("🚨 CRITICAL: Very low sequence uniqueness - high risk of overfitting!")
    elif uniqueness_ratio < 0.7:
        print("⚠️  Warning: Low sequence uniqueness - may still overfit")
    else:
        print("✅ Good sequence diversity")
    
    # Sequence length analysis
    seq_lens = [len(seq) for seq in sequences]
    print(f"Sequence lengths - Mean: {np.mean(seq_lens):.2f}, Max: {max(seq_lens)}, Min: {min(seq_lens)}")
    
    # Set up model parameters
    vocab_size = len(vocab)
    num_classes = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run cross-validation first to get realistic expectations
    print(f"\n{'='*80}")
    print("PHASE 1: CROSS-VALIDATION FOR REALISTIC PERFORMANCE ASSESSMENT")
    print(f"{'='*80}")
    
    cv_mean, cv_std = cross_validate(sequences, labels, vocab_size, num_classes, device, label_encoder)
    
    # Set realistic targets based on CV results
    if cv_mean > 0.8:
        print(f"\n🚨 CRITICAL WARNING: CV accuracy {cv_mean:.4f} is too high!")
        print("This suggests data leakage is still present. Results may not be meaningful.")
        realistic_target = 0.4  # Still train, but expect lower
    elif cv_mean > 0.6:
        print(f"\n⚠️  CV accuracy {cv_mean:.4f} is quite high - double check for leakage")
        realistic_target = cv_mean * 0.9  # Slightly lower target
    else:
        print(f"\n✅ CV accuracy {cv_mean:.4f} looks realistic for ultra-clean data")
        realistic_target = cv_mean * 1.1  # Slightly higher target
    
    print(f"Setting target accuracy for full training: {realistic_target:.4f}")
    
    # Split data for final training
    print(f"\n{'='*80}")
    print("PHASE 2: FULL TRAINING ON ULTRA-CLEAN DATA")
    print(f"{'='*80}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.3, stratify=labels, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Create datasets and loaders
    train_dataset = CommandDataset(X_train, y_train)
    val_dataset = CommandDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = RealisticLSTMClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, num_classes, DROPOUT).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Simple loss function (no class weighting for realistic evaluation)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training loop with realistic expectations
    best_val_acc = 0
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"\nStarting training on ultra-clean data...")
    print(f"Expected accuracy range: {EXPECTED_ACCURACY['acceptable']:.1%} - {EXPECTED_ACCURACY['excellent']:.1%}")
    print(f"Target accuracy: {realistic_target:.1%}")
    print("-" * 80)
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for seqs, labs in train_pbar:
            seqs, labs = seqs.to(device), labs.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labs)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * seqs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        train_acc = correct / total
        avg_train_loss = total_loss / total
        
        # Validation
        val_loss, val_acc, val_preds, val_labels = evaluate_model(
            model, val_loader, criterion, device, label_encoder, verbose=(epoch % 10 == 0 or epoch == EPOCHS-1)
        )
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Performance assessment
        performance_msg = assess_performance(val_acc, num_classes)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Assessment: {performance_msg}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Check for overfitting (less strict for ultra-clean data)
        overfitting_gap = train_acc - val_acc
        if overfitting_gap > 0.20:
            print(f"⚠️  Significant overfitting detected (gap: {overfitting_gap:.4f})")
        elif overfitting_gap > 0.10:
            print(f"⚠️  Moderate overfitting (gap: {overfitting_gap:.4f})")
        else:
            print("✅ Good generalization")
        
        # Model saving and early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_classes': num_classes,
                'dropout': DROPOUT,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'realistic_target': realistic_target
            }
            
            torch.save(checkpoint, SAVE_PATH)
            print(f"✅ New best model saved! Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏳ Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Final assessment
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    
    print(f"Cross-validation accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Final validation accuracy: {best_val_acc:.4f}")
    print(f"Target accuracy: {realistic_target:.4f}")
    
    final_assessment = assess_performance(best_val_acc, num_classes)
    print(f"Final assessment: {final_assessment}")
    
    if best_val_acc > 0.8:
        print(f"\n🚨 CONCLUSION: Accuracy {best_val_acc:.1%} is unrealistically high!")
        print("This strongly suggests that data leakage is still present.")
        print("Consider even more aggressive preprocessing or manual data inspection.")
    elif best_val_acc > realistic_target + 0.1:
        print(f"\n✅ CONCLUSION: Good performance ({best_val_acc:.1%}) above target!")
        print("This suggests the model learned meaningful patterns from clean data.")
    elif best_val_acc >= EXPECTED_ACCURACY['acceptable']:
        print(f"\n😐 CONCLUSION: Acceptable performance ({best_val_acc:.1%}) for ultra-clean data.")
        print("Better than random baseline, which is realistic given minimal features.")
    else:
        print(f"\n❌ CONCLUSION: Poor performance ({best_val_acc:.1%}) - at or below random.")
        print("Model failed to learn meaningful patterns. Consider:")
        print("- Less aggressive preprocessing")
        print("- Different model architecture")
        print("- More training data")
    
    # Final detailed evaluation
    print(f"\nDetailed final evaluation:")
    evaluate_model(model, val_loader, criterion, device, label_encoder, verbose=True)

if __name__ == '__main__':
    main()
