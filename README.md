# gradient-descent-comparison-ml

# Gradient Descent Comparison (BGD vs SGD vs Mini Batch GD)

This project demonstrates the implementation and comparison of three gradient descent optimization techniques used in machine learning.

Algorithms Implemented
- Batch Gradient Descent (BGD)
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent

Dataset
The dataset used is `homeprices_banglore.csv` which contains:

Features:
- Area
- Bedrooms

Target:
- Price

Project Workflow
1. Load dataset
2. Data preprocessing
3. Feature scaling using MinMaxScaler
4. Implement Gradient Descent algorithms
5. Compare cost vs epoch performance

Gradient Descent Types

Batch Gradient Descent  
Uses the entire dataset to compute gradients before updating weights.

Stochastic Gradient Descent  
Updates weights after every single training sample.

Mini Batch Gradient Descent  
Updates weights after a small batch of samples.

Example:

Total samples = 50  
Batch size = 10  

Weight updates happen after:
10 samples  
20 samples  
30 samples  
40 samples  
50 samples

Visualization

The notebook plots a **Cost vs Epoch graph** comparing all three methods.

Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

How to Run

1. Install dependencies

pip install -r requirements.txt

2. Run the notebook

jupyter notebook gradient_descent_comparison.ipynb
