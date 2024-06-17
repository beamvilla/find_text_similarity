import numpy as np
from numpy.linalg import norm


def calculate_cosine(embedded_sentence_1: np.array, embedded_sentence_2: np.array) -> float:
    return np.dot(embedded_sentence_1, embedded_sentence_2) / (norm(embedded_sentence_1) * norm(embedded_sentence_2))