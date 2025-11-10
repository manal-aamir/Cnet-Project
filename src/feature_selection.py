
# Feature Selection: CFS and GA (per base paper)
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import clone

def cfs_select(X, y, max_features=13, redundancy_threshold=0.7):
	"""
	Correlation-based Feature Selection (CFS):
	- Rank features by abs correlation with target
	- Greedily add features, skipping those highly correlated with already selected
	- Stop at max_features
	"""
	corrs = [(col, abs(np.corrcoef(X[col], y)[0,1])) for col in X.columns]
	corrs.sort(key=lambda x: x[1], reverse=True)
	selected = []
	for col, _ in corrs:
		if len(selected) >= max_features:
			break
		# Check redundancy
		redundant = False
		for sel in selected:
			if abs(np.corrcoef(X[col], X[sel])[0,1]) >= redundancy_threshold:
				redundant = True
				break
		if not redundant:
			selected.append(col)
	return selected

def ga_select(X, y, max_features=13, population_size=30, generations=25, mutation_rate=0.1, crossover_rate=0.8, cv_splits=3, random_state=42):
	"""
	Genetic Algorithm Feature Selection (GA):
	- Binary chromosome for feature inclusion
	- Fitness: mean CV accuracy (DecisionTree)
	- Returns best feature subset (length=max_features)
	"""
	rng = np.random.default_rng(random_state)
	n_features = X.shape[1]
	feature_names = list(X.columns)
	def random_chrom():
		chrom = rng.random(n_features) < (max_features / n_features)
		if not chrom.any():
			chrom[rng.integers(0, n_features)] = True
		return chrom
	population = np.array([random_chrom() for _ in range(population_size)], dtype=bool)
	best_mask, best_score = population[0], -np.inf
	skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
	for gen in range(generations):
		scores = np.zeros(population.shape[0])
		for i, mask in enumerate(population):
			if not mask.any():
				continue
			X_sub = X.iloc[:, mask]
			clf = DecisionTreeClassifier(random_state=random_state)
			try:
				cv_scores = cross_val_score(clf, X_sub, y, cv=skf, scoring='accuracy', n_jobs=1)
				score = np.mean(cv_scores)
			except Exception:
				score = 0.0
			# Penalty for too many features
			if mask.sum() > max_features:
				score -= 0.01 * (mask.sum() - max_features)
			scores[i] = score
		best_idx = int(np.argmax(scores))
		if scores[best_idx] > best_score:
			best_score, best_mask = scores[best_idx], population[best_idx].copy()
		# Selection (tournament)
		parents = []
		for _ in range(population_size):
			i1, i2 = rng.choice(population_size, 2, replace=False)
			parents.append(population[i1 if scores[i1] >= scores[i2] else i2])
		# Crossover and mutation
		offspring = []
		for i in range(0, population_size, 2):
			a, b = parents[i], parents[(i+1)%population_size]
			if rng.random() < crossover_rate:
				p = rng.integers(1, n_features-1)
				c1 = np.concatenate([a[:p], b[p:]])
				c2 = np.concatenate([b[:p], a[p:]])
			else:
				c1, c2 = a.copy(), b.copy()
			# Mutation
			for c in (c1, c2):
				mask = rng.random(n_features) < mutation_rate
				c[mask] = ~c[mask]
				if not c.any():
					c[rng.integers(0, n_features)] = True
			offspring.extend([c1, c2])
		population = np.array(offspring[:population_size], dtype=bool)
	# Final best subset
	final_mask = best_mask
	# If too many features, keep top max_features by importance
	if final_mask.sum() > max_features:
		importances = np.zeros(n_features)
		for i, use in enumerate(final_mask):
			if use:
				try:
					imp = abs(np.corrcoef(X.iloc[:,i], y)[0,1])
				except Exception:
					imp = 0.0
				importances[i] = imp
		top_idx = np.argsort(importances)[-max_features:]
		final_mask = np.zeros(n_features, dtype=bool)
		final_mask[top_idx] = True
	selected = [feature_names[i] for i, use in enumerate(final_mask) if use]
	return selected

