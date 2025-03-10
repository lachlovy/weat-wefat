from typing import Any, Dict, List, Optional

import numpy as np
import scipy.stats as stats


class WordEmbeddingTest:
    """
    A class that implements Word Embedding Association Tests (WEAT) and
    Word Embedding Factual Association Tests (WEFAT) as described in:
    "Semantics derived automatically from language corpora contain human-like biases"
    by Caliskan, Bryson, & Narayanan (2017).
    """

    def __init__(self, word_vectors: Dict[str, np.ndarray]):
        """
        Initialize with a dictionary mapping words to their vector representations.

        Parameters:
        -----------
        word_vectors : Dict[str, np.ndarray]
            Dictionary mapping words to their vector embeddings
        """
        self.word_vectors = word_vectors

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate the cosine similarity between two vectors.

        Parameters:
        -----------
        vec1, vec2 : np.ndarray
            The vectors to compare

        Returns:
        --------
        float
            Cosine similarity value
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0

        return dot_product / (norm_vec1 * norm_vec2)

    def s(self, t: np.ndarray, A: List[np.ndarray], B: List[np.ndarray]) -> float:
        """
        Measures the association of word vector t with attribute sets A and B.

        Parameters:
        -----------
        t : np.ndarray
            Target word vector
        A, B : List[np.ndarray]
            Lists of attribute word vectors

        Returns:
        --------
        float
            Association measure
        """
        return np.mean([self.cosine_similarity(t, a) for a in A]) - np.mean(
            [self.cosine_similarity(t, b) for b in B]
        )

    def test_statistic(
        self,
        X: List[np.ndarray],
        Y: List[np.ndarray],
        A: List[np.ndarray],
        B: List[np.ndarray],
    ) -> float:
        """
        Calculates the WEAT test statistic.

        Parameters:
        -----------
        X, Y : List[np.ndarray]
            Lists of target word vectors
        A, B : List[np.ndarray]
            Lists of attribute word vectors

        Returns:
        --------
        float
            WEAT test statistic
        """
        return np.sum([self.s(x, A, B) for x in X]) - np.sum(
            [self.s(y, A, B) for y in Y]
        )

    def effect_size(
        self,
        X: List[np.ndarray],
        Y: List[np.ndarray],
        A: List[np.ndarray],
        B: List[np.ndarray],
    ) -> float:
        """
        Calculates the WEAT effect size.

        Parameters:
        -----------
        X, Y : List[np.ndarray]
            Lists of target word vectors
        A, B : List[np.ndarray]
            Lists of attribute word vectors

        Returns:
        --------
        float
            Effect size (Cohen's d)
        """
        all_s = [self.s(w, A, B) for w in X + Y]
        s_X = [self.s(x, A, B) for x in X]
        s_Y = [self.s(y, A, B) for y in Y]

        return (np.mean(s_X) - np.mean(s_Y)) / np.std(all_s)

    def permutation_test(
        self,
        X: List[np.ndarray],
        Y: List[np.ndarray],
        A: List[np.ndarray],
        B: List[np.ndarray],
        n_iterations: int = 10000,
    ) -> float:
        """
        Implements the permutation test for WEAT.

        Parameters:
        -----------
        X, Y : List[np.ndarray]
            Lists of target word vectors
        A, B : List[np.ndarray]
            Lists of attribute word vectors
        n_iterations : int, optional
            Number of permutations for the test (default: 10000)

        Returns:
        --------
        float
            p-value
        """
        observed = self.test_statistic(X, Y, A, B)

        all_targets = X + Y
        size_X = len(X)
        count = 0

        for _ in range(n_iterations):
            indices = np.random.permutation(len(all_targets))
            Xi = [all_targets[i] for i in indices[:size_X]]
            Yi = [all_targets[i] for i in indices[size_X:]]
            test_stat_perm = self.test_statistic(Xi, Yi, A, B)

            if test_stat_perm > observed:
                count += 1

        return count / n_iterations

    def run_weat(
        self,
        target_X_words: List[str],
        target_Y_words: List[str],
        attr_A_words: List[str],
        attr_B_words: List[str],
        iterations: int = 10000,
    ) -> Optional[Dict[str, Any]]:
        """
        Runs a complete WEAT test with provided word lists.

        Parameters:
        -----------
        target_X_words, target_Y_words : List[str]
            Lists of words for target concepts
        attr_A_words, attr_B_words : List[str]
            Lists of words for attribute concepts
        iterations : int, optional
            Number of permutation iterations (default: 10000)

        Returns:
        --------
        Dict or None
            Dictionary with test results or None if words are missing
        """
        # Get vectors for words, skip if not in vocabulary
        X = [self.word_vectors[w] for w in target_X_words if w in self.word_vectors]
        Y = [self.word_vectors[w] for w in target_Y_words if w in self.word_vectors]
        A = [self.word_vectors[w] for w in attr_A_words if w in self.word_vectors]
        B = [self.word_vectors[w] for w in attr_B_words if w in self.word_vectors]

        # Check if we have vectors for all words
        if len(X) == 0 or len(Y) == 0 or len(A) == 0 or len(B) == 0:
            print(
                f"Words missing from vocabulary: {len(target_X_words) - len(X)} from X, "
                f"{len(target_Y_words) - len(Y)} from Y, {len(attr_A_words) - len(A)} from A, "
                f"{len(attr_B_words) - len(B)} from B"
            )
            return None

        # Run WEAT test
        stat = self.test_statistic(X, Y, A, B)
        es = self.effect_size(X, Y, A, B)
        p_val = self.permutation_test(X, Y, A, B, iterations)

        return {
            "test_statistic": stat,
            "effect_size": es,
            "p_value": p_val,
            "words_found": {"X": len(X), "Y": len(Y), "A": len(A), "B": len(B)},
            "words_tested": {
                "X": target_X_words,
                "Y": target_Y_words,
                "A": attr_A_words,
                "B": attr_B_words,
            },
        }

    def word_association_differential(
        self, w: np.ndarray, A: List[np.ndarray], B: List[np.ndarray]
    ) -> float:
        """
        Calculates normalized association differential for WEFAT.

        Parameters:
        -----------
        w : np.ndarray
            Word vector
        A, B : List[np.ndarray]
            Lists of attribute vectors

        Returns:
        --------
        float
            Normalized differential association score
        """
        all_attributes = A + B
        mean_a = np.mean([self.cosine_similarity(w, a) for a in A])
        mean_b = np.mean([self.cosine_similarity(w, b) for b in B])
        std_dev = np.std([self.cosine_similarity(w, x) for x in all_attributes])

        if std_dev == 0:
            return 0

        return (mean_a - mean_b) / std_dev

    def run_wefat(
        self,
        target_words: List[str],
        attr_A_words: List[str],
        attr_B_words: List[str],
        real_property: List[float],
    ) -> Optional[Dict[str, Any]]:
        """
        Runs a Word Embedding Factual Association Test (WEFAT).

        Parameters:
        -----------
        target_words : List[str]
            List of target concept words
        attr_A_words, attr_B_words : List[str]
            Lists of attribute concept words
        real_property : List[float]
            Real-world property values for each target word

        Returns:
        --------
        Dict or None
            Dictionary with test results or None if words are missing
        """
        # Get vectors for words, skip if not in vocabulary
        W = [(w, self.word_vectors[w]) for w in target_words if w in self.word_vectors]
        A = [self.word_vectors[w] for w in attr_A_words if w in self.word_vectors]
        B = [self.word_vectors[w] for w in attr_B_words if w in self.word_vectors]

        # Make sure we have enough valid words
        if len(W) == 0 or len(A) == 0 or len(B) == 0:
            print(
                f"Words missing from vocabulary: {len(target_words) - len(W)} from targets, "
                f"{len(attr_A_words) - len(A)} from A, {len(attr_B_words) - len(B)} from B"
            )
            return None

        # Check if real_property has the same length as valid target words
        if len(W) != len(real_property):
            # Filter real_property to match found words
            valid_indices = [
                i for i, w in enumerate(target_words) if w in self.word_vectors
            ]
            real_property = [real_property[i] for i in valid_indices]

        # Calculate association scores for each word
        associations = []
        words = []

        for word, vector in W:
            association = self.word_association_differential(vector, A, B)
            associations.append(association)
            words.append(word)

        # Calculate correlation between association scores and real-world property
        correlation, p_value = stats.pearsonr(associations, real_property)

        return {
            "correlation": correlation,
            "p_value": p_value,
            "associations": dict(zip(words, associations)),
            "real_values": dict(zip(words, real_property)),
            "words_found": {"target": len(W), "A": len(A), "B": len(B)},
        }


# Example usage
def main():
    # Load GloVe embeddings (this is just a demonstration)
    glove_path = "./data/glove-6B/glove.6B.100d.txt"
    glove_dict = load_glove(glove_path)

    # Initialize the test class
    weat_tester = WordEmbeddingTest(glove_dict)

    # Example WEAT: test for gender bias in occupations
    target_X = ["programmer", "engineer", "scientist", "mathematician", "developer"]
    target_Y = ["nurse", "teacher", "librarian", "secretary", "homemaker"]
    attr_A = ["man", "male", "he", "him", "his", "son", "father", "brother", "husband"]
    attr_B = [
        "woman",
        "female",
        "she",
        "her",
        "hers",
        "daughter",
        "mother",
        "sister",
        "wife",
    ]

    # Run WEAT
    weat_results = weat_tester.run_weat(target_X, target_Y, attr_A, attr_B)

    if weat_results:
        print(f"WEAT Test Statistic: {weat_results['test_statistic']:.4f}")
        print(f"WEAT Effect Size: {weat_results['effect_size']:.4f}")
        print(f"WEAT p-value: {weat_results['p_value']:.4f}")

    # Example WEFAT: test for correlation between gender association and occupation statistics
    occupations = ["doctor", "nurse", "programmer", "engineer", "teacher", "librarian"]
    # Example percentage of women in each occupation (hypothetical values)
    gender_percentages = [0.40, 0.85, 0.20, 0.15, 0.75, 0.80]

    # Run WEFAT
    wefat_results = weat_tester.run_wefat(
        occupations, attr_A, attr_B, gender_percentages
    )

    if wefat_results:
        print(f"WEFAT Correlation: {wefat_results['correlation']:.4f}")
        print(f"WEFAT p-value: {wefat_results['p_value']:.4f}")


def load_glove(glove_data_path: str) -> Dict[str, np.ndarray]:
    """
    Load GloVe word vectors from file.

    Parameters:
    -----------
    glove_data_path : str
        Path to the GloVe vector file

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping words to their vector embeddings
    """
    word_vectors = {}
    with open(glove_data_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors


if __name__ == "__main__":
    main()
