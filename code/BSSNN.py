import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

data = pd.read_csv("binary_data.csv")

# ------------------
# Dataset Preparation
# ------------------
class BayesianDataset(Dataset):
    def __init__(self, df, input_cols, target_col):
        self.inputs = df[input_cols].values
        self.targets = df[target_col].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )

# ------------------
# Neural Network Model
# ------------------
class BayesianInspiredNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BayesianInspiredNNModel, self).__init__()
        # Compute P(y, x1, x2, ...)
        self.fc1_joint = nn.Linear(input_size, hidden_size)
        self.relu_joint = nn.ReLU()
        self.fc2_joint = nn.Linear(hidden_size, 1)
        
        # Compute P(x1, x2, ...)
        self.fc1_marginal = nn.Linear(input_size, hidden_size)
        self.relu_marginal = nn.ReLU()
        self.fc2_marginal = nn.Linear(hidden_size, 1)
        
        # Final sigmoid activation for probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Joint probability computation
        joint = self.relu_joint(self.fc1_joint(x))
        joint = self.fc2_joint(joint)  # Output: logit for P(y, x1, x2, ...)
        
        # Marginal probability computation
        marginal = self.relu_marginal(self.fc1_marginal(x))
        marginal = self.fc2_marginal(marginal)  # Output: logit for P(x1, x2, ...)
        
        # Bayesian division: P(y|x1, x2, ...) = P(y, x1, x2, ...) / P(x1, x2, ...)
        conditional = joint - marginal  # Log-space division
        return self.sigmoid(conditional)  # Probability score

# ------------------
# Training and Evaluation Functions
# ------------------
def train_model(model, loader, optimizer, criterion, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(x_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            pred = model(x_batch).squeeze()
            preds.append(pred.numpy())
            targets.append(y_batch.numpy())
    preds = np.hstack(preds)
    targets = np.hstack(targets)
    return preds, targets

def calculate_auc_ks(preds, targets):
    auc = roc_auc_score(targets, preds)
    fpr, tpr, _ = roc_curve(targets, preds)
    ks = max(tpr - fpr)
    return auc, ks

# ------------------
# Main Experiment
# ------------------
input_cols = ["x1", "x2", "x3"]
target_col = "y"

# Split data into train and validation sets
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = BayesianDataset(train_df, input_cols, target_col)
val_dataset = BayesianDataset(val_df, input_cols, target_col)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize and train the model
input_size = len(input_cols)
hidden_size = 64

model = BayesianNNModel(input_size, hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()

print("Training Bayesian Neural Network...")
train_model(model, train_loader, optimizer, criterion, epochs=400)

print("Evaluating Model...")
nn_preds, nn_targets = evaluate_model(model, val_loader)
nn_auc, nn_ks = calculate_auc_ks(nn_preds, nn_targets)

print(f"\nPerformance Metrics:\n  AUC: {nn_auc:.4f}\n  KS: {nn_ks:.4f}")

def logistic_regression_benchmark(train_df, val_df, input_cols, target_col):
    X_train = train_df[input_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[input_cols].values
    y_val = val_df[target_col].values

    # Train Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Predict probabilities
    preds = log_reg.predict_proba(X_val)[:, 1]  # Get probability for the positive class

    # Calculate AUC and KS
    auc, ks = calculate_auc_ks(preds, y_val)
    return auc, ks

# Evaluate Logistic Regression
print("Evaluating Logistic Regression...")
lr_auc, lr_ks = logistic_regression_benchmark(train_df, val_df, input_cols, target_col)

print(f"\nLogistic Regression Performance:\n  AUC: {lr_auc:.4f}\n  KS: {lr_ks:.4f}")


######################predict x|Y#######################

# ------------------
# Dataset Preparation
# ------------------
class BayesianDataset(Dataset):
    def __init__(self, df, input_cols, target_col):
        self.inputs = df[target_col].values  # y as input
        self.targets = df[input_cols].values  # X as target

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(0),  # y as input
            torch.tensor(self.targets[idx], dtype=torch.float32),  # X as target
        )

# ------------------
# Neural Network Model
# ------------------
class BayesianPredictXGivenY(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianPredictXGivenY, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)  # Predict multiple X values

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Linear output for regression
        return x

# ------------------
# Training and Evaluation Functions
# ------------------
def train_model(model, loader, optimizer, criterion, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for y_batch, x_batch in loader:
            optimizer.zero_grad()
            preds = model(y_batch)  # Forward pass
            loss = criterion(preds, x_batch)  # Compute loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

def evaluate_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for y_batch, x_batch in loader:
            pred = model(y_batch)
            preds.append(pred.numpy())
            targets.append(x_batch.numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    mse = mean_squared_error(targets, preds)
    return preds, targets, mse

# ------------------
# Main Experiment
# ------------------
input_cols = ["x1", "x2", "x3", "x4"]
target_col = "y"

# Split data into train and validation sets
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = BayesianDataset(train_df, input_cols, target_col)
val_dataset = BayesianDataset(val_df, input_cols, target_col)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize and train the model
input_size = 1  # Single input: y
hidden_size = 64
output_size = len(input_cols)  # Predict all X variables

model = BayesianPredictXGivenY(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

print("Training Bayesian Predict X|Y Model...")
train_model(model, train_loader, optimizer, criterion, epochs=100)

print("Evaluating Model...")
nn_preds, nn_targets, nn_mse = evaluate_model(model, val_loader)

print(f"\nEvaluation Metric:\n  Mean Squared Error (MSE): {nn_mse:.4f}")

# Print some sample predictions
print("\nSample Predictions (P(X|Y)):")
for i in range(5):
    print(f"True X: {nn_targets[i]}, Predicted X: {nn_preds[i]}")
    


