# train_lstm.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def create_sequences(data, labels, sequence_length=10):
    """
    Converts time-series data into sequences for LSTM training.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # The sequence of features (e.g., 10 time steps)
        X.append(data[i:(i + sequence_length)])
        # The label (RUL) at the *end* of that sequence
        y.append(labels[i + sequence_length - 1])
        
    return np.array(X), np.array(y)

def build_model(input_shape):
    """Builds a simple LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1)) # Output layer: 1 node for RUL
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # --- 1. Load the Processed Dataset ---
    try:
        data = pd.read_csv("results/ml_training_dataset.csv")
    except FileNotFoundError:
        print("Error: 'results/ml_training_dataset.csv' not found.")
        print("Please run 'run_preprocessing.py' first.")
        return
        
    print(f"Loaded {len(data)} total data points.")
    
    # Define features (X) and label (y)
    FEATURES = ['rho_smooth', 'D_smooth']
    LABEL = 'RUL'
    
    # --- 2. Scale Features ---
    # Scaling is critical for neural networks
    scaler = StandardScaler()
    data[FEATURES] = scaler.fit_transform(data[FEATURES])
    
    # --- 3. Create Sequences ---
    # We will create sequences *within* each coupon's data
    
    SEQUENCE_LENGTH = 10 # Look at 10 time steps to predict the 10th
    
    all_X_seq = []
    all_y_seq = []
    
    # Group by coupon_id to avoid creating sequences that span
    # across two different coupons
    for coupon_id, group in data.groupby('coupon_id'):
        if len(group) < SEQUENCE_LENGTH:
            continue # Skip coupons with too little data
            
        group_data = group[FEATURES].values
        group_labels = group[LABEL].values
        
        X_seq, y_seq = create_sequences(group_data, group_labels, SEQUENCE_LENGTH)
        
        all_X_seq.append(X_seq)
        all_y_seq.append(y_seq)
        
    if not all_X_seq:
        print(f"Error: No sequences created. Is SEQUENCE_LENGTH ({SEQUENCE_LENGTH}) larger than your smallest coupon data?")
        return
        
    # Combine all sequences into one big training set
    X = np.concatenate(all_X_seq)
    y = np.concatenate(all_y_seq)
    
    print(f"Created {len(X)} sequences of length {SEQUENCE_LENGTH}.")
    
    # --- 4. Split Data ---
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # --- 5. Build & Train Model ---
    # Input shape is (SEQUENCE_LENGTH, num_features)
    input_shape = (X_train.shape[1], X_train.shape[2]) 
    model = build_model(input_shape)
    model.summary()
    
    # Stop training if the validation loss doesn't improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2, # Use part of training data for validation
        callbacks=[early_stopping]
    )
    
    # --- 6. Evaluate Model ---
    test_loss = model.evaluate(X_test, y_test)
    print(f"\nTest Loss (MSE): {test_loss}")
    
    # --- 7. Plot Predictions ---
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title('RUL Prediction: True vs. Predicted')
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
    plt.legend()
    plt.savefig('results/lstm_prediction_plot.png')
    plt.show()
    
    print("Saved prediction plot to 'results/lstm_prediction_plot.png'")

if __name__ == "__main__":
    main()