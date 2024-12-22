#  Spoken Digits Classification using ANN

## Project Objective

The objective of this project is to build a multiclass classification model using an Artificial Neural Network (ANN) to classify spoken commands representing digits (0 to 9). The input audio data is processed into Mel Frequency Cepstral Coefficients (MFCC) features, which are then used to train the ANN model. The model aims to achieve high accuracy on the test set by utilizing robust preprocessing and regularization techniques.

---

## Dataset Information

This project uses the [Google Speech Commands dataset](https://www.kaggle.com/datasets/neehakurelli/google-speech-commands), which contains audio files of spoken words. The dataset includes multiple pronunciations of various words, including digits `0` to `9`.

### Dataset Preprocessing Steps:

1. **Filtering:** Extract only the audio files corresponding to digits (`0` to `9`) from the dataset.
2. **Feature Extraction:** Compute Mel Frequency Cepstral Coefficients (MFCC) features for each audio file. These features are widely used in audio and speech processing.
3. **Normalization:** Scale the extracted MFCC features to have values between 0 and 1.
4. **Data Splitting:** Split the data into training (80%) and validation (20%) sets.
5. **One-Hot Encoding:** Convert the class labels (digits) into one-hot encoded format for compatibility with the ANN model.

---

## Instructions for Running the Code

1. Clone this repository to your local machine:

   ```bash
   git clone <repository-url>
   ```

2. Download the dataset from Kaggle:
   Ensure you have the `kaggle` API configured. Use the following command to download the dataset:

   ```bash
   kaggle datasets download -d neehakurelli/google-speech-commands
   ```

3. Extract the dataset into a directory (e.g., `data/`).

4. Run the preprocessing and training script:

   ```bash
   python main.py
   ```

5. The script will:

   - Preprocess the data to extract MFCC features and split it into training and validation sets.
   - Train the ANN model using the training set and evaluate it on the validation set.
   - Save the trained model and display evaluation metrics.

6. Evaluate the model:

   - The script provides accuracy, precision, recall, and F1-score metrics for model performance.
   - It also plots training/validation loss and accuracy curves.

---

## Dependencies and Installation Instructions

### Dependencies:

- Python 3.8+
- TensorFlow 2.0+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Librosa

### Installation Instructions:

1. Create and activate a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # For Linux/Mac
   env\Scripts\activate   # For Windows
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure the `requirements.txt` file includes the following:

   ```
   tensorflow
   numpy
   pandas
   matplotlib
   scikit-learn
   librosa
   kaggle
   ```

---

## Notes:

- Ensure the dataset path is correctly configured in the script.
- Adjust hyperparameters (e.g., batch size, learning rate) in the script for fine-tuning the model performance.
- For any issues, feel free to open a ticket or contact the repository maintainer.

