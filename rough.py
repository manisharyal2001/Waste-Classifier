import numpy as np

def main():
    # Input number of data points and features
    n = int(input("Enter number of data points: "))
    m = int(input("Enter number of features: "))

    # Input feature matrix
    print("Enter feature matrix (each row space-separated):")
    X = [list(map(float, input().split())) for _ in range(n)]

    # Input target vector
    y = list(map(float, input("Enter target values: ").split()))

    X = np.array(X)
    y = np.array(y)

    # Add column of ones for intercept
    X_b = np.c_[np.ones((n, 1)), X]

    # Normal equation: θ = (XᵀX)⁻¹ Xᵀy
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    intercept = theta[0]
    slopes = theta[1:]

    print("\nIntercept:", intercept)
    print("Slopes:", slopes)

    # Predictions and evaluation
    y_pred = X_b @ theta
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    print("\nMSE:", mse)
    print("R² Score:", r2)

if __name__ == "__main__":
    main()
