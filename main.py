# main.py

from src.data_gen import generate_data
from src.preprocess import load_data, preprocess, split_features_target
from src.model import train_model, save_model
from src.evaluate import evaluate_model, print_evaluation

# Step 1: Generate data (run once)
generate_data()

# Step 2: Load + preprocess
df = load_data()
df = preprocess(df)

# Step 3: Split
X, y = split_features_target(df)

# Step 4: Train model
model, X_test, y_test = train_model(X, y)

# Step 5: Evaluate
results = evaluate_model(model, X_test, y_test)
print_evaluation(results)

# Step 6: Save model
save_model(model)