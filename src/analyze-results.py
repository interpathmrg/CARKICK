import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt # Matplotlib should now pick a GUI backend
import os

# --- Create an output directory for plots if you still want to save them ---
plots_dir = "prediction_analysis_plots"
os.makedirs(plots_dir, exist_ok=True)

# --- CSV Path (ensure this points to the correct, single CSV file) ---
try:
    base_output_dir = "/home/mrgonzalez/PYTHON/CARKICK/output/predicciones_rf"
    csv_files_in_output = [f for f in os.listdir(base_output_dir) if f.startswith("part-") and f.endswith(".csv")]
    if not csv_files_in_output:
        raise FileNotFoundError(f"No CSV part-files found in {base_output_dir}")
    output_csv_path = os.path.join(base_output_dir, csv_files_in_output[0])
    print(f"Reading predictions from: {output_csv_path}")
    predictions_df = pd.read_csv(output_csv_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except pd.errors.EmptyDataError:
    print(f"Error: The CSV file at {output_csv_path} is empty.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the CSV: {e}")
    exit()

# --- Distribution of Predictions ---
if 'prediction' in predictions_df.columns:
    plt.figure(figsize=(6, 4)) # Create a new figure
    sns.countplot(x='prediction', data=predictions_df)
    plt.title('Distribution of Predicted BadBuys in Test Set')
    plt.xlabel('Predicted Label (0: GoodBuy, 1: BadBuy)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "prediction_distribution.png")) # Still good to save
    print(f"Saved: {os.path.join(plots_dir, 'prediction_distribution.png')}")
    plt.show() # This should now work
    plt.close() # Close after showing if you open multiple plots
else:
    print("Warning: 'prediction' column not found in CSV. Skipping count plot.")

# --- Distribution of Probabilities ---
if 'probability_class_1' in predictions_df.columns:
    plt.figure(figsize=(8, 5)) # Create a new figure
    sns.histplot(predictions_df['probability_class_1'], kde=True, bins=20)
    plt.title('Distribution of Predicted Probabilities for BadBuy (Class 1)')
    plt.xlabel('Predicted Probability of Being a BadBuy')
    plt.ylabel('Frequency')
    plt.axvline(0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "probability_distribution.png")) # Still good to save
    print(f"Saved: {os.path.join(plots_dir, 'probability_distribution.png')}")
    plt.show() # This should now work
    plt.close() # Close after showing
else:
    print("Warning: 'probability_class_1' column not found in CSV. Skipping histogram.")

print(f"\nPlots saved in '{plots_dir}/' directory and shown interactively.")