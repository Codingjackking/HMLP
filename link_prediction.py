"""
link_prediction.py
==================
Three classical link-prediction heuristics operating on attribute sets,
plus a lightweight relation-type classifier that answers the professor's
question: "Can you predict the relation?"

Heuristics
----------
- jaccard             : |A ∩ B| / |A ∪ B|
- adamic_adar         : Σ 1/log(freq(attr)) for shared attributes
                        (freq = number of anime that share this attribute)
- preferential_attachment : |A| * |B|

Relation-type prediction
------------------------
- RelationClassifier  : given a scored candidate pair, predict whether the
                        relation is "continuation" (SEQUEL/PREQUEL),
                        "side" (SIDE_STORY/SPIN_OFF), or "alternative"
                        (ALTERNATIVE/PARENT) using a simple feature vector
                        and a trained logistic-regression model.
"""

import math
from collections import defaultdict
from typing import Optional

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Attribute set builder

def build_attribute_set(anime: dict, include_source: bool = False) -> frozenset:
    """
    Build a frozenset of (type, value) tuples for a single anime dict.

    Attributes included by default:
      ('genre',  genre_name)
      ('studio', studio_name)

    Source is excluded by default to match the projection graph, which also
    drops source because it is too coarse (~42% of anime share "MANGA") and
    inflates pairwise similarity across unrelated titles.  Pass
    include_source=True to include it, e.g. when building features for the
    relation classifier where source as a same/different flag is still useful.

    Using frozenset ensures immutability and fast set operations.
    """
    attrs: set = set()
    for genre in anime.get("genres", []):
        attrs.add(("genre", genre))
    for studio in anime.get("studios", []):
        attrs.add(("studio", studio))
    if include_source:
        attrs.add(("source", anime.get("source", "UNKNOWN")))
    return frozenset(attrs)


def build_all_attribute_sets(anime_list: list[dict]) -> dict[int, frozenset]:
    """Return {anime_id -> frozenset of attributes} for the entire dataset."""
    return {a["id"]: build_attribute_set(a) for a in anime_list}


# Global attribute frequency index

def build_attr_frequency(anime_list: list[dict]) -> dict[tuple, int]:
    """
    Count how many anime share each attribute across the whole dataset.
    Used by Adamic-Adar to penalise common attributes (e.g. 'Action' genre).
    Source is excluded to match build_attribute_set (see its docstring).

    Returns
    -------
    dict  (type, value) -> count
    """
    freq: dict[tuple, int] = defaultdict(int)
    for anime in anime_list:
        for genre in anime.get("genres", []):
            freq[("genre",  genre)] += 1
        for studio in anime.get("studios", []):
            freq[("studio", studio)] += 1
        # source intentionally excluded – matches build_attribute_set default
    return dict(freq)


# Heuristic algorithms

def jaccard(a_attrs: frozenset, b_attrs: frozenset) -> float:
    """
    Jaccard similarity: ratio of shared attributes to total attributes.

    Score in [0, 1].  A score of 1 means identical attribute profiles.
    Returns 0 for empty sets to avoid division by zero.
    """
    union_size = len(a_attrs | b_attrs)
    if union_size == 0:
        return 0.0
    return len(a_attrs & b_attrs) / union_size


def adamic_adar(
    a_attrs: frozenset,
    b_attrs: frozenset,
    attr_freq: dict[tuple, int],
) -> float:
    """
    Adamic-Adar index: rewards rare shared attributes more than common ones.

    For each shared attribute, contribute 1 / log(frequency_in_dataset).
    Attributes appearing in only one anime are skipped (log(1) = 0).

    Parameters
    ----------
    attr_freq : global frequency dict from build_attr_frequency()
    """
    score = 0.0
    for attr in a_attrs & b_attrs:
        freq = attr_freq.get(attr, 1)
        if freq > 1:
            score += 1.0 / math.log(freq)
    return score


def preferential_attachment(a_attrs: frozenset, b_attrs: frozenset) -> float:
    """
    Preferential attachment: product of attribute-set sizes.

    Models the "rich-get-richer" intuition: anime with many attributes
    (genres, studios) are more likely to share connections with others.
    Returns an integer-valued float.
    """
    return float(len(a_attrs) * len(b_attrs))


# Batch scoring

def score_all_pairs(
    anime_list: list[dict],
    attr_sets: dict[int, frozenset],
    attr_freq: dict[tuple, int],
    known_pairs: Optional[set[frozenset]] = None,
) -> dict[str, list[tuple[int, int, float]]]:
    """
    Score every non-connected candidate pair with all three heuristics.

    Parameters
    ----------
    anime_list   : full anime dataset
    attr_sets    : precomputed attribute sets from build_all_attribute_sets()
    attr_freq    : global frequency index from build_attr_frequency()
    known_pairs  : set of frozensets {id_a, id_b} already connected in the
                   projection graph; these are skipped.

    Returns
    -------
    dict with keys "jaccard", "adamic_adar", "preferential_attachment",
    each mapping to a list of (id_a, id_b, score) sorted descending by score.
    """
    if known_pairs is None:
        known_pairs = set()

    ids = [a["id"] for a in anime_list]
    n   = len(ids)

    jac_scores: list[tuple[int, int, float]] = []
    aa_scores:  list[tuple[int, int, float]] = []
    pa_scores:  list[tuple[int, int, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            pair = frozenset([a, b])
            if pair in known_pairs:
                continue

            a_attrs = attr_sets[a]
            b_attrs = attr_sets[b]

            jac_scores.append((a, b, jaccard(a_attrs, b_attrs)))
            aa_scores.append( (a, b, adamic_adar(a_attrs, b_attrs, attr_freq)))
            pa_scores.append( (a, b, preferential_attachment(a_attrs, b_attrs)))

    return {
        "jaccard":                sorted(jac_scores, key=lambda x: x[2], reverse=True),
        "adamic_adar":            sorted(aa_scores,  key=lambda x: x[2], reverse=True),
        "preferential_attachment":sorted(pa_scores,  key=lambda x: x[2], reverse=True),
    }


# Relation-type classifier

class RelationClassifier:
    """
    Lightweight logistic regression classifier that predicts the semantic
    group of an Anime->Anime relation given attribute-level features.

    Target groups (from graph_builder.RELATION_GROUPS):
      "continuation" : SEQUEL, PREQUEL
      "side"         : SIDE_STORY, SPIN_OFF
      "alternative"  : ALTERNATIVE, PARENT

    Feature vector for pair (a, b)
    --------------------------------
    [0]  jaccard score
    [1]  adamic-adar score (normalised by max in dataset)
    [2]  preferential attachment (normalised)
    [3]  shared genre count
    [4]  shared studio count  (1 if same studio, else 0)
    [5]  same source flag     (1 if identical source, else 0)
    [6]  genre union size
    """

    def __init__(self):
        self._trained = False
        self._classes: list[str] = []
        self._weights = None   # shape (n_classes, n_features)
        self._bias    = None   # shape (n_classes,)

    def _featurize(
        self,
        a: dict,
        b: dict,
        attr_freq: dict[tuple, int],
    ) -> list[float]:
        # Use include_source=True here: source same/different is a useful
        # feature for the relation classifier even though it is excluded from
        # the projection graph similarity weights.
        a_attrs = build_attribute_set(a, include_source=True)
        b_attrs = build_attribute_set(b, include_source=True)

        shared     = a_attrs & b_attrs
        n_shared_g = sum(1 for t, _ in shared if t == "genre")
        n_shared_s = sum(1 for t, _ in shared if t == "studio")
        same_src   = 1.0 if a.get("source") == b.get("source") else 0.0

        jac = jaccard(a_attrs, b_attrs)
        aa  = adamic_adar(a_attrs, b_attrs, attr_freq)
        pa  = preferential_attachment(a_attrs, b_attrs)
        g_union = len({v for t, v in (a_attrs | b_attrs) if t == "genre"})

        return [jac, aa, pa, float(n_shared_g), float(n_shared_s), same_src, float(g_union)]

    def train(
        self,
        anime_list: list[dict],
        relation_triples: list[dict],
        attr_freq: dict[tuple, int],
    ) -> "RelationClassifier":
        """
        Train using labeled Anime->Anime triples.

        Parameters
        ----------
        anime_list        : full dataset (needed to look up each anime's attrs)
        relation_triples  : list of {"head": int, "relation": str,
                                     "tail": int, "group": str}
        attr_freq         : from build_attr_frequency()
        """
        if not HAS_SKLEARN:
            return self

        id_to_anime = {a["id"]: a for a in anime_list}
        X, y = [], []

        for triple in relation_triples:
            head_anime = id_to_anime.get(triple["head"])
            tail_anime = id_to_anime.get(triple["tail"])
            if head_anime is None or tail_anime is None:
                continue
            if triple["group"] == "other":
                continue   # skip low-signal groups
            features = self._featurize(head_anime, tail_anime, attr_freq)
            X.append(features)
            y.append(triple["group"])

        if len(set(y)) < 2:
            return self

        X_arr = np.array(X, dtype=float)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_arr)

        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        clf.fit(X_scaled, y)

        self._classes = list(clf.classes_)
        self._weights = clf.coef_
        self._bias    = clf.intercept_
        self._clf     = clf
        self._trained = True
        return self

    def predict(
        self,
        a: dict,
        b: dict,
        attr_freq: dict[tuple, int],
    ) -> dict:
        """
        Predict relation group for a candidate pair.

        Returns
        -------
        dict with keys:
          "predicted_group" : str  (e.g. "continuation")
          "probabilities"   : dict class -> probability
          "trained"         : bool
        """
        if not self._trained:
            return {"predicted_group": None, "probabilities": {}, "trained": False}

        features = np.array(
            self._featurize(a, b, attr_freq), dtype=float
        ).reshape(1, -1)
        features_scaled = self._scaler.transform(features)
        proba  = self._clf.predict_proba(features_scaled)[0]
        pred   = self._clf.predict(features_scaled)[0]

        return {
            "predicted_group": pred,
            "probabilities":   dict(zip(self._classes, proba.tolist())),
            "trained":         True,
        }

    def evaluate(
        self,
        anime_list: list[dict],
        test_triples: list[dict],
        attr_freq: dict[tuple, int],
    ) -> dict:
        """
        Compute per-class precision, recall, F1 on held-out triples.
        Skips "other" group triples.
        """
        if not self._trained:
            return {}

        id_to_anime = {a["id"]: a for a in anime_list}
        y_true, y_pred = [], []

        for triple in test_triples:
            if triple["group"] == "other":
                continue
            ha = id_to_anime.get(triple["head"])
            ta = id_to_anime.get(triple["tail"])
            if ha is None or ta is None:
                continue
            result = self.predict(ha, ta, attr_freq)
            y_true.append(triple["group"])
            y_pred.append(result["predicted_group"])

        if not y_true:
            return {}

        try:
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            return report
        except Exception:
            # Manual fallback if sklearn unavailable
            classes = list(set(y_true))
            results = {}
            for cls in classes:
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec  = tp / (tp + fn) if (tp + fn) else 0.0
                f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
                results[cls] = {"precision": prec, "recall": rec, "f1-score": f1}
            return results
