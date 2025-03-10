from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

# Extracting the first base estimator (decision tree) from the AdaBoost model
base_estimator = model.estimators_[0]

X = df_reduced.drop(columns='fraud')
dot_data = StringIO()
export_graphviz(base_estimator,
                out_file=dot_data,
                filled=True,
                rounded=True,
                feature_names=X.columns,
                class_names=['no', 'yes'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')  # Save the image file
Image(graph.create_png())
