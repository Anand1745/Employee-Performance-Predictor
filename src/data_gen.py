# src/data_gen.py

import pandas as pd
import numpy as np
import os

def generate_data(n=500, save_path="data/employees.csv"):
    np.random.seed(42)

    data = pd.DataFrame({
        'Age': np.random.randint(22, 60, n),
        'Experience': np.random.randint(1, 20, n),
        'Salary': np.random.randint(20000, 150000, n),
        'Training_Hours': np.random.randint(5, 100, n),
        'Department': np.random.choice(['HR', 'IT', 'Sales'], n)
    })

    # Performance calculation logic
    data['Performance_Score'] = (
        data['Experience'] * 0.3 +
        data['Training_Hours'] * 0.2 +
        data['Salary'] * 0.0001
    )

    # Convert score to categories
    data['Performance'] = pd.cut(
        data['Performance_Score'],
        bins=[0, 15, 25, 100],
        labels=['Low', 'Medium', 'High']
    )

    data.drop(columns=['Performance_Score'], inplace=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data.to_csv(save_path, index=False)
    print(f"✅ Dataset saved to {save_path}")

if __name__ == "__main__":
    generate_data()