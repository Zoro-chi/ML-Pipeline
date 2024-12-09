import pandas as pd
import sys
import os
import yaml

# Load parameters from params.yaml
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(root_dir, "params.yaml")
params = yaml.safe_load(open(params_path))["preprocess"]


def preprocess(input_path, output_path):
    # Load data
    data = pd.read_csv(input_path)

    # Feature Engineering done here if needed:
    # # Drop rows with missing values
    # data = data.dropna()

    # Create Output Directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save data
    data.to_csv(output_path, header=None, index=False)
    print(f"Preprocessed Data saved to {output_path}")


# if __name__ == "__main__":
#     preprocess(params["input"], params["output"])

if __name__ == "__main__":
    input_path = os.path.join(root_dir, params["input"])
    output_path = os.path.join(root_dir, params["output"])

    preprocess(input_path, output_path)
