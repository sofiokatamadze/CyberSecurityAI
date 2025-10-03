from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=[0,1])
    return {"accuracy": acc}, cm

def save_confusion_matrix_plot(cm, out_path, class_names=("legit","spam")):
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0,1], class_names)
    plt.yticks([0,1], class_names)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
