# neural-languange-model
# STF-IDF Model Evaluation

This project explores sentiment analysis on Sesotho and Setswana tweets using various feature extraction and classification techniques, including TF-IDF, subword tokenization (BPE, WordPiece), and neural models. The experiments are implemented in the `stf_idf_evalutions.ipynb` notebook.

## Dataset

- **Sesotho Tweets:** sesotho_tweets.csv
- **Setswana Tweets:** setswana_tweets.csv
- **Transformed News Dataset:** Transformed_NewsSA_Dataset.csv

## Feature Extraction

- **TF-IDF Vectorization:** Both character and word n-gram TF-IDF features are extracted using `TfidfVectorizer`.
- **Subword Tokenization:** Experiments include BPE and WordPiece tokenization, followed by TF-IDF vectorization.

## Model Architectures

- **Logistic Regression:** Baseline classifier using TF-IDF features.
- **Neural Network Classifier:** Custom PyTorch model (`STFIDFClassifier`) with configurable hidden layers.
- **Transformer Model:** AfroXLMR transformer for comparison.

## Training & Evaluation

- Data is split into training and validation sets using a custom `train_val_dataloader`.
- Models are trained with cross-entropy loss and Adam optimizer.
- Evaluation metrics include weighted F1-score and classification reports.

## Results

- The notebook visualizes model performance using bar charts comparing F1-scores across different models and feature sets.
- Example models compared:
  - TF-IDF + Logistic Regression
  - BPE STF-IDF + Logistic Regression
  - AfroXLMR Transformer
  - STF-IDF + ngrams
  - STF-IDF + BPE
  - STF-IDF + Word Piece

## Checkpoints

Model checkpoints and training artifacts are saved in the results directory, with subfolders for each checkpoint (e.g., `checkpoint-1042/`, `checkpoint-2084/`, etc.).

## Usage

1. **Install dependencies:**  
   Make sure you have Python, PyTorch, scikit-learn, and Jupyter installed.
2. **Open the notebook:**  
   Open `stf_idf_evalutions.ipynb` in Jupyter or VS Code.
3. **Run cells:**  
   Execute the notebook cells to preprocess data, train models, and visualize results.

## File Structure

- `stf_idf_evalutions.ipynb`: Main experiment notebook.
- sesotho_tweets.csv, setswana_tweets.csv: Tweet datasets.
- Transformed_NewsSA_Dataset.csv: Additional news dataset.
- results: Model checkpoints and training artifacts.

## Citation

If you use this work, please cite the notebook and datasets accordingly.

---

For details, see the full notebook: `stf_idf_evalutions.ipynb
