from sklearn.ensemble import RandomForestClassifier

# Active learning
DEFAULT_BATCH_SIZE = 10
DEFAULT_ESTIMATOR = RandomForestClassifier

# Redis
NOT_ANNOTATED = 'to_annotate'
ANNOTATED = 'annotated'
