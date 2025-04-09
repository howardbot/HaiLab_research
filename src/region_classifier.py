# src/Models.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

def train_classifier(X, y):
    print("Training classifier to decode slant...")

    # feature standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # linear SVM
    clf = SVC(kernel='linear')

    # 5-fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # calculate correctness
    scores = cross_val_score(clf, X_scaled, y, cv=cv)
    print(f"Cross-validated accuracy: {np.mean(scores)*100:.2f}%")

    # get predication to draw confusion matrix
    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Slant decoding")
    plt.tight_layout()
    plt.show()
