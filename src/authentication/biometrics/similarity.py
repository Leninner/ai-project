from typing import List, Union
import numpy as np


class SimilarityCalculator:
    @staticmethod
    def cosine_similarity(a: Union[List[float], np.ndarray], 
                         b: Union[List[float], np.ndarray]) -> float:
        a_array = np.array(a)
        b_array = np.array(b)
        
        dot_product = np.dot(a_array, b_array)
        norm_a = np.linalg.norm(a_array)
        norm_b = np.linalg.norm(b_array)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    @staticmethod
    def calculate_mean_embedding(embeddings_list: List[List[float]]) -> List[float]:
        if not embeddings_list:
            return []
        
        embeddings_array = np.array(embeddings_list)
        return np.mean(embeddings_array, axis=0).tolist()

