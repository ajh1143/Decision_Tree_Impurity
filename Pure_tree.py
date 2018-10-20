from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def TestTrain(X, y, test_size):
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = test_size, stratify=y, random_state=1)
    return X_train, X_test, y_train, y_test


def entropy(X_train, y_train, y_test, X_test):
    dtc_entropy = DecisionTreeClassifier(criterion='entropy', random_state=1)
    dtc_entropy.fit(X_train,y_train)
    y_pred_entropy = dtc_entropy.predict(X_test)
    entropic_score = accuracy_score(y_test, y_pred_entropy)
    return entropic_score
    
    
def entropy_depth_test(depth, X_train, y_train, y_test, X_test):
    df = pd.DataFrame(columns=['Depth', 'Accuracy'])
    df = df.set_index('Depth')
    for cur_depth in range(depth):
      dtc_entropy = DecisionTreeClassifier(max_depth=cur_depth, criterion='entropy', random_state=1)
      dtc_entropy.fit(X_train,y_train)
      y_pred_entropy = dtc_entropy.predict(X_test)
      entropic_score = accuracy_score(y_test, y_pred_entropy)
      df.append([{'Depth': cur_depth}, {'Accuracy':entropic_score}], ignore_index=True)
      return df


def gini(X_train, y_train, y_test, X_test):
    dtc_gini = DecisionTreeClassifier(criterion='gini', random_state=1)
    dtc_gini.fit(X_train,y_train)
    y_pred_gini = dtc_gini.predict(X_test)
    gini_score = accuracy_score(y_test, y_pred_gini)
    return gini_score


 def gini_depth_test(depth, X_train, y_train, y_test, X_test):
    df = pd.DataFrame(columns=['Depth', 'Accuracy'])
    df = df.set_index('Depth')
    for cur_depth in range(depth):
      dtc_gini = DecisionTreeClassifier(max_depth=cur_depth, criterion='gini', random_state=1)
      dtc_gini.fit(X_train,y_train)
      y_pred_gini = dtc_gini.predict(X_test)
      gini_score = accuracy_score(y_test, y_pred_gini)
      df.append([{'Depth': cur_depth}, {'Accuracy':gini}], ignore_index=True)
      return df


def entropy_gini(X_train, y_train, y_test, X_test):
    entropy()
    gini()
    
    
def tree_viz(dtc, df, col_names, classes):
    classer = classes
    dot = tree.export_graphviz(dtc, out_file=None, feature_names=col_names, class_names=classer, filled=True, rounded=False,       
                               special_characters=True)
    graph = graphviz.Source(dot)
    graph.format = 'png'
    graph.render('Output', view=True)
