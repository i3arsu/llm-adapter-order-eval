import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Paths
correct_answers_path = 'shopping_cart_final_normalized.csv'
models_dir = 'results'  # folder with model CSV files
model_files = [f for f in os.listdir(models_dir) if f.endswith('.csv')]

# Columns to compare (correct answers vs model predictions)
correct_answer_columns = ['action', 'product', 'quantity']  # from correct_answers.csv
model_answer_columns = ['llm_action', 'llm_product', 'llm_quantity']  # corresponding columns in model CSVs

# Load ground truth
df_truth = pd.read_csv(correct_answers_path)

results = []

for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    try:
        df_model = pd.read_csv(model_path)
    except Exception as e:
        print(f"Error loading model CSV '{model_file}': {e}")
        continue

    # Check if model CSV has all required columns
    missing_cols = [col for col in model_answer_columns if col not in df_model.columns]
    if missing_cols:
        print(f"Model CSV '{model_file}' missing columns: {missing_cols}")
        continue

    # Ensure both dataframes have the same number of rows
    min_rows = min(len(df_truth), len(df_model))
    if len(df_truth) != len(df_model):
        print(f"Warning: Row count mismatch for '{model_file}'. Using first {min_rows} rows.")
        df_truth_subset = df_truth.iloc[:min_rows]
        df_model_subset = df_model.iloc[:min_rows]
    else:
        df_truth_subset = df_truth
        df_model_subset = df_model

    # Calculate per-column accuracy and overall exact match
    column_accuracies = {}
    matches_per_column = []
    
    for corr_col, mod_col in zip(correct_answer_columns, model_answer_columns):
        # Make sure both columns exist in their respective DataFrames
        if corr_col not in df_truth_subset.columns:
            print(f"Correct answers missing column '{corr_col}', skipping model '{model_file}'.")
            continue
        
        # Compare strings after stripping whitespace and converting to string
        y_true_col = df_truth_subset[corr_col].astype(str).str.strip().str.lower()
        y_pred_col = df_model_subset[mod_col].astype(str).str.strip().str.lower()

        # Create binary matches for this column
        matches = (y_true_col == y_pred_col).astype(int)
        matches_per_column.append(matches)
        
        # Calculate accuracy for this individual column
        column_accuracy = matches.mean()
        column_accuracies[f'{corr_col}_accuracy'] = column_accuracy

    # Aggregate matches across all columns (all columns must match for exact match)
    if matches_per_column:
        # Exact match: all columns must be correct
        exact_matches = np.logical_and.reduce(matches_per_column).astype(int)
        
        # For F1 calculation: 1 = exact match, 0 = not exact match
        y_true_binary = np.ones(len(exact_matches))  # Perfect prediction would be all 1s
        y_pred_binary = exact_matches  # Actual predictions (1 if exact match, 0 otherwise)
        
        # Calculate metrics for exact match
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        
        # Alternative: Calculate macro-average F1 across individual columns
        column_f1_scores = []
        for i, (corr_col, mod_col) in enumerate(zip(correct_answer_columns, model_answer_columns)):
            y_true_col = df_truth_subset[corr_col].astype(str).str.strip().str.lower()
            y_pred_col = df_model_subset[mod_col].astype(str).str.strip().str.lower()
            
            # Get unique labels for this column
            unique_labels = sorted(set(y_true_col.unique()) | set(y_pred_col.unique()))
            
            if len(unique_labels) > 1:
                col_f1 = f1_score(y_true_col, y_pred_col, average='weighted', zero_division=0)
                column_f1_scores.append(col_f1)
        
        macro_f1 = np.mean(column_f1_scores) if column_f1_scores else 0
        
    else:
        print(f"No columns compared for model '{model_file}', skipping.")
        continue

    # Store results
    result = {
        'model_name': model_file.replace('.csv', ''),
        'exact_match_f1': f1,
        'exact_match_precision': precision,
        'exact_match_recall': recall,
        'exact_match_accuracy': accuracy,
        'macro_avg_f1': macro_f1,
        'total_samples': len(exact_matches),
        'exact_matches': exact_matches.sum()
    }
    
    # Add individual column accuracies
    result.update(column_accuracies)
    results.append(result)

# Build results DataFrame and sort by F1 score descending
results_df = pd.DataFrame(results).sort_values(by='exact_match_f1', ascending=False)

# Display results
print("Model Performance Results:")
print("=" * 80)
display_columns = ['model_name', 'exact_match_f1', 'exact_match_precision', 
                  'exact_match_recall', 'exact_match_accuracy', 'macro_avg_f1', 
                  'exact_matches', 'total_samples']
print(results_df[display_columns].to_string(index=False, float_format='%.3f'))

# Plot F1 scores
plt.figure(figsize=(20, 10))

# Plot 1: Exact Match F1 Scores
plt.subplot(2, 1, 1)
sns.barplot(data=results_df, x='model_name', y='exact_match_f1', palette='viridis')
plt.title("F1 Scores per Model", fontsize=20, fontweight='bold')
plt.ylabel("F1 Score", fontsize=18, fontweight='bold')
plt.xlabel("")
for i, v in enumerate(results_df['exact_match_f1']):
    # plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    height = plt.gca().patches[i].get_height()
    plt.gca().text(plt.gca().patches[i].get_x() + plt.gca().patches[i].get_width() / 2,
        height / 2,
        f'{height:.3f}',
        ha='center',
        va='center', 
        fontsize=18,
        fontweight='bold')

plt.xticks(rotation=45, ha='right', fontsize=16, fontweight='bold')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)

# Plot 2: Comparison of different metrics
# plt.subplot(2, 1, 2)
# metrics_to_plot = ['exact_match_f1', 'exact_match_precision', 'exact_match_recall', 'macro_avg_f1']
# x = np.arange(len(results_df))
# width = 0.2

# for i, metric in enumerate(metrics_to_plot):
#     offset = (i - len(metrics_to_plot)/2 + 0.5) * width
#     plt.bar(x + offset, results_df[metric], width, 
#             label=metric.replace('exact_match_', '').replace('_', ' ').title(), alpha=0.8)

# plt.xlabel('Model')
# plt.ylabel('Score')
# plt.title('Comparison of Different Metrics', fontsize=14, fontweight='bold')
# plt.xticks(x, results_df['model_name'], rotation=45, ha='right')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.ylim(0, 1)
# plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("chart.png", dpi=800, bbox_inches='tight')
plt.show()


# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Best performing model (Exact Match F1): {results_df.iloc[0]['model_name']} ({results_df.iloc[0]['exact_match_f1']:.3f})")
print(f"Average F1 score across all models: {results_df['exact_match_f1'].mean():.3f}")
print(f"Standard deviation of F1 scores: {results_df['exact_match_f1'].std():.3f}")

# Show individual column performance for best model
if len(results_df) > 0:
    best_model = results_df.iloc[0]
    print(f"\nIndividual column performance for best model '{best_model['model_name']}':")
    for col in correct_answer_columns:
        col_key = f'{col}_accuracy'
        if col_key in best_model:
            print(f"  {col}: {best_model[col_key]:.3f}")