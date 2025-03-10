from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
y = df_reduced['fraud']
# Perform cross-validated predictions
y_pred_cv = cross_val_predict(model, X, y, cv=10)  # Assuming 10-fold cross-validation

# Generate confusion matrix
cm = confusion_matrix(y, y_pred_cv)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Fraud', 'Fraud'],
            yticklabels=['No Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
