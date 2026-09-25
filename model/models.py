from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from model.config import CONTAMINATION, RANDOM_STATE


def make_models(contamination=CONTAMINATION, seed=RANDOM_STATE):
    """
    Candidate unsupervised detectors. Distance/density/covariance based methods
    need scaled features; tree-based Isolation Forest does not.
    """
    return {
        "Isolation Forest": IsolationForest(
            n_estimators=200, contamination=contamination, random_state=seed
        ),
        "Local Outlier Factor": make_pipeline(
            StandardScaler(),
            LocalOutlierFactor(n_neighbors=35, novelty=True, contamination=contamination),
        ),
        "One-Class SVM": make_pipeline(
            StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)
        ),
        "Elliptic Envelope": make_pipeline(
            StandardScaler(), EllipticEnvelope(contamination=contamination, random_state=seed)
        ),
    }
