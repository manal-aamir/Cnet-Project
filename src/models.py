from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_classifiers(random_state=42):
	return {
		'DecisionTree': DecisionTreeClassifier(random_state=random_state),
		'RandomForest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
		'KNN': KNeighborsClassifier(n_neighbors=5),
		'SVM': SVC(kernel='rbf', gamma='scale', random_state=random_state)
	}

