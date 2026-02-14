# Import necessary libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load the Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    # Train a Gradient Boosting classifier
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=10)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Print classification report (precision, recall, f1-score)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine.target_names)
    disp.plot(cmap='Blues')
    plt.title("Wine Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as confusion_matrix.png")

    # Save the model to a file
    joblib.dump(model, 'wine_model.pkl')

    print("The model training was successful")