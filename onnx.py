import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# Don't forget to mount Google Drive if you need it:
# from google.colab import drive
# drive.mount('/content/drive')

# Setting the label here since it is used in multiple function calls
label = 'fraud'

# Import the data
df = import_data('/content/drive/MyDrive/IS 455/Data/orders.csv', messages=False)

# Clean/prepare the data
df = bin_groups(df, messages=False)
df = missing_drop(df, label)
df = impute_KNN(df, label)
print(df.columns)
# Select features and store a trained model
model = fit_cv_classification(df, 8, label, messages=False) # We have to begin with a trained model
df_reduced = select_features(df.copy(), label, model) # Use that model to select features
model = fit_cv_classification_expanded(df_reduced, label, k=8, r=4)  # Retrain the model with the smaller feature set

# Deployment pipeline
initial_type = [('float_input', FloatTensorType([None, df_reduced.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Saving the model
with open("decision_tree_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
