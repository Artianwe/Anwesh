# KNN Classifier for Iris Dataset
# Developed for basic ML implementation and understanding

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data        # Feature data
    y = iris.target      # Target labels

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create the KNN classifier with k=3
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model
    knn.fit(X_train, y_train)

    # Predict using the model
    y_pred = knn.predict(X_test)

    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("KNN Iris Classifier Results")
    print("---------------------------")
    print("Predicted labels:", y_pred)
    print("Actual labels:   ", y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()