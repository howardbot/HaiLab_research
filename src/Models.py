import numpy as np
from  sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def train_classifier(X,Y):
    print("Training SVM classifier on slant labels...")

    # standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # support vector machines
    clf = SVC(kernel = 'linear')

    # cross validation
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scores = cross_val_score(clf,X_scaled,Y,cv=cv)
    print(f"Cross-validated accuracy: {np.mean(scores)*100:.2f}%")

    #Fitting the model and drawing the confusion matrix
    clf.fit(X_scaled,Y)
    y_pred = clf.predict(X_scaled)

    cm = confusion_matrix(Y,y_pred,labels=np.unique(Y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(Y))
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Area Classification (CIP vs V3A)")
    plt.tight_layout()
    plt.show()


