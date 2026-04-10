from typing import List, Dict, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FeatureExtractor:
    """
    A class for extracting TF-IDF features from text data.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df=1,
        max_df=1.0
    ):
        """
        Initialize the feature extractor.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=min_df,
            max_df=max_df
        )
        self.fitted = False

    def fit(self, documents: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on the given documents.
        """
        if not documents:
            raise ValueError("No documents provided for fitting the vectorizer")

        self.vectorizer.fit(documents)
        self.fitted = True

    def transform(self, documents: Union[str, List[str]]) -> np.ndarray:
        """
        Transform documents into TF-IDF features.
        """
        if not self.fitted:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")

        if isinstance(documents, str):
            documents = [documents]

        return self.vectorizer.transform(documents)

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform the documents in one step.
        """
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (terms) from the vectorizer.
        """
        if not self.fitted:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")

        return self.vectorizer.get_feature_names_out().tolist()

    def get_top_keywords(
        self,
        document: str,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get the top N keywords with highest TF-IDF scores for a document.
        """
        if not self.fitted:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")

        tfidf_vector = self.transform(document)

        feature_array = np.array(self.get_feature_names())
        tfidf_scores = tfidf_vector.toarray().flatten()
        top_indices = np.argsort(tfidf_scores)[::-1][:top_n]

        return [
            (feature_array[i], tfidf_scores[i])
            for i in top_indices
            if tfidf_scores[i] > 0
        ]

    def get_feature_matrix_info(self, feature_matrix) -> Dict:
        """
        Get information about the feature matrix.
        """
        if not self.fitted:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")

        if hasattr(feature_matrix, 'toarray'):
            feature_matrix = feature_matrix.toarray()

        nonzero = np.count_nonzero(feature_matrix)

        return {
            'shape': feature_matrix.shape,
            'num_nonzero_elements': nonzero,
            'sparsity': 1.0 - (nonzero / feature_matrix.size),
            'average_nonzero_per_doc': np.mean(
                np.count_nonzero(feature_matrix, axis=1)
            ),
            'max_tfidf': np.max(feature_matrix),
            'min_tfidf': np.min(feature_matrix[feature_matrix > 0])
        }


def calculate_similarity(doc1_tfidf, doc2_tfidf) -> float:
    """
    Calculate cosine similarity between two TF-IDF vectors.
    """
    if doc1_tfidf.shape[0] > 1 or doc2_tfidf.shape[0] > 1:
        raise ValueError("Input should be single document vectors")

    return cosine_similarity(doc1_tfidf, doc2_tfidf)[0][0]
