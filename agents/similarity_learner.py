"""
Context Similarity Learning System for TravelPlanner
This module implements novel similarity-based context discovery and transfer learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import json
import pickle
from collections import defaultdict

class ContextSimilarityLearner:
    """
    Novel context similarity learning system that can transfer knowledge
    between similar contexts without exact matches.
    """
    
    def __init__(self, embedding_dim: int = 100, similarity_threshold: float = 0.7):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Context embeddings
        self.context_embeddings = {}
        self.embedding_model = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Similarity patterns
        self.similarity_patterns = defaultdict(list)
        self.context_clusters = {}
        
        # Transfer learning data
        self.transfer_mappings = defaultdict(dict)
        
    def create_context_embedding(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Create numerical embedding of context for similarity comparison.
        This is a key innovation - converting complex contexts to comparable vectors.
        """
        # Convert context to text representation
        context_text = self._context_to_text(context)
        
        # Create embedding
        if not hasattr(self, 'embedding_model_fitted'):
            # First time - fit the model
            self.embedding_model_fitted = True
            embedding = self.embedding_model.fit_transform([context_text])
        else:
            embedding = self.embedding_model.transform([context_text])
        
        return embedding.toarray()[0]
    
    def _context_to_text(self, context: Dict[str, Any]) -> str:
        """Convert context dictionary to text representation"""
        text_parts = []
        
        # Add categorical features
        for key, value in context.items():
            if isinstance(value, str):
                text_parts.append(f"{key}_{value}")
            elif isinstance(value, (int, float)):
                # Convert numeric values to categories
                if key == 'budget':
                    if value < 50:
                        text_parts.append(f"{key}_low")
                    elif value < 100:
                        text_parts.append(f"{key}_medium")
                    else:
                        text_parts.append(f"{key}_high")
                elif key == 'group_size':
                    if value == 1:
                        text_parts.append(f"{key}_solo")
                    elif value <= 4:
                        text_parts.append(f"{key}_small")
                    else:
                        text_parts.append(f"{key}_large")
        
        return " ".join(text_parts)
    
    def learn_context_similarity(self, context1: Dict[str, Any], 
                               context2: Dict[str, Any], 
                               similarity_score: float,
                               user_feedback: str = ""):
        """
        Learn that certain contexts are similar based on user feedback or patterns.
        This enables transfer learning between contexts.
        """
        # Create embeddings
        emb1 = self.create_context_embedding(context1)
        emb2 = self.create_context_embedding(context2)
        
        # Store similarity pattern
        pattern = {
            'context1': context1,
            'context2': context2,
            'embedding1': emb1.tolist(),
            'embedding2': emb2.tolist(),
            'similarity': similarity_score,
            'user_feedback': user_feedback
        }
        
        # Store in similarity patterns
        context1_key = self._context_to_key(context1)
        context2_key = self._context_to_key(context2)
        
        self.similarity_patterns[context1_key].append(pattern)
        self.similarity_patterns[context2_key].append(pattern)
        
        # Update transfer mappings
        self._update_transfer_mappings(context1, context2, similarity_score)
    
    def _context_to_key(self, context: Dict[str, Any]) -> str:
        """Create a unique key for context"""
        return "|".join([f"{k}:{v}" for k, v in sorted(context.items())])
    
    def _update_transfer_mappings(self, context1: Dict[str, Any], 
                                context2: Dict[str, Any], similarity: float):
        """Update transfer learning mappings between contexts"""
        context1_key = self._context_to_key(context1)
        context2_key = self._context_to_key(context2)
        
        # Bidirectional mapping
        self.transfer_mappings[context1_key][context2_key] = similarity
        self.transfer_mappings[context2_key][context1_key] = similarity
    
    def find_similar_contexts(self, target_context: Dict[str, Any], 
                            available_contexts: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find contexts similar to target using learned similarity patterns.
        This enables knowledge transfer to new, unseen contexts.
        """
        target_embedding = self.create_context_embedding(target_context)
        similar_contexts = []
        
        for context in available_contexts:
            # Calculate similarity using learned patterns
            similarity = self._calculate_context_similarity(target_context, context)
            
            if similarity > self.similarity_threshold:
                similar_contexts.append((context, similarity))
        
        # Sort by similarity
        return sorted(similar_contexts, key=lambda x: x[1], reverse=True)
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                   context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts using multiple methods"""
        
        # Method 1: Direct embedding similarity
        emb1 = self.create_context_embedding(context1)
        emb2 = self.create_context_embedding(context2)
        embedding_similarity = cosine_similarity([emb1], [emb2])[0][0]
        
        # Method 2: Learned pattern similarity
        pattern_similarity = self._get_pattern_similarity(context1, context2)
        
        # Method 3: Feature-based similarity
        feature_similarity = self._calculate_feature_similarity(context1, context2)
        
        # Combine similarities with weights
        combined_similarity = (
            0.4 * embedding_similarity +
            0.4 * pattern_similarity +
            0.2 * feature_similarity
        )
        
        return combined_similarity
    
    def _get_pattern_similarity(self, context1: Dict[str, Any], 
                              context2: Dict[str, Any]) -> float:
        """Get similarity based on learned patterns"""
        context1_key = self._context_to_key(context1)
        context2_key = self._context_to_key(context2)
        
        # Check if we have direct transfer mapping
        if context2_key in self.transfer_mappings.get(context1_key, {}):
            return self.transfer_mappings[context1_key][context2_key]
        
        # Check for indirect patterns
        max_similarity = 0.0
        for pattern in self.similarity_patterns.get(context1_key, []):
            # Calculate similarity to pattern's other context
            if pattern['context1'] == context1:
                other_context = pattern['context2']
            else:
                other_context = pattern['context1']
            
            other_similarity = self._calculate_feature_similarity(context2, other_context)
            pattern_similarity = pattern['similarity'] * other_similarity
            max_similarity = max(max_similarity, pattern_similarity)
        
        return max_similarity
    
    def _calculate_feature_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """Calculate similarity based on individual features"""
        common_features = set(context1.keys()) & set(context2.keys())
        if not common_features:
            return 0.0
        
        similarities = []
        for feature in common_features:
            val1, val2 = context1[feature], context2[feature]
            
            if val1 == val2:
                similarities.append(1.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                max_val = max(val1, val2)
                min_val = min(val1, val2)
                if max_val > 0:
                    similarities.append(min_val / max_val)
                else:
                    similarities.append(1.0)
            else:
                # String similarity (simple)
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def discover_context_clusters(self, contexts: List[Dict[str, Any]], 
                                n_clusters: int = 5) -> Dict[int, List[Dict[str, Any]]]:
        """
        Discover clusters of similar contexts using unsupervised learning.
        This enables automatic pattern discovery.
        """
        if len(contexts) < n_clusters:
            return {0: contexts}
        
        # Create embeddings for all contexts
        embeddings = []
        context_mapping = {}
        
        for i, context in enumerate(contexts):
            embedding = self.create_context_embedding(context)
            embeddings.append(embedding)
            context_mapping[i] = context
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group contexts by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(context_mapping[i])
        
        # Store cluster information
        self.context_clusters = dict(clusters)
        
        return dict(clusters)
    
    def get_context_recommendations(self, target_context: Dict[str, Any], 
                                  context_database: List[Dict[str, Any]], 
                                  top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get context recommendations based on similarity learning.
        This demonstrates the novel transfer learning capability.
        """
        # Find similar contexts
        similar_contexts = self.find_similar_contexts(target_context, context_database)
        
        # Apply context-specific adjustments
        adjusted_contexts = []
        for context, similarity in similar_contexts:
            adjusted_similarity = self._apply_context_adjustments(
                target_context, context, similarity
            )
            adjusted_contexts.append((context, adjusted_similarity))
        
        # Sort and return top-k
        return sorted(adjusted_contexts, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _apply_context_adjustments(self, target_context: Dict[str, Any], 
                                 candidate_context: Dict[str, Any], 
                                 base_similarity: float) -> float:
        """Apply context-specific adjustments to similarity scores"""
        adjusted_similarity = base_similarity
        
        # Time-based adjustments
        if (target_context.get('meal_time') != candidate_context.get('meal_time') and
            'meal_time' in target_context and 'meal_time' in candidate_context):
            adjusted_similarity *= 0.8  # Reduce similarity for different meal times
        
        # Occasion-based adjustments
        if (target_context.get('occasion') != candidate_context.get('occasion') and
            'occasion' in target_context and 'occasion' in candidate_context):
            adjusted_similarity *= 0.7  # Reduce similarity for different occasions
        
        # Budget-based adjustments
        if 'budget' in target_context and 'budget' in candidate_context:
            budget_diff = abs(target_context['budget'] - candidate_context['budget'])
            if budget_diff > 50:  # Large budget difference
                adjusted_similarity *= 0.6
        
        # Location-based adjustments (if available)
        if ('location' in target_context and 'location' in candidate_context and
            target_context['location'] != candidate_context['location']):
            adjusted_similarity *= 0.9  # Slight reduction for different locations
        
        return adjusted_similarity
    
    def learn_from_user_feedback(self, user_feedback: str, 
                               original_context: Dict[str, Any],
                               recommended_context: Dict[str, Any],
                               satisfaction_score: float):
        """
        Learn from user feedback to improve similarity patterns.
        This enables continuous learning and improvement.
        """
        # Extract feedback sentiment and context preferences
        feedback_sentiment = self._analyze_feedback_sentiment(user_feedback)
        
        # Update similarity based on feedback
        if satisfaction_score > 0.7:  # Positive feedback
            similarity_adjustment = 0.1
        elif satisfaction_score < 0.3:  # Negative feedback
            similarity_adjustment = -0.1
        else:  # Neutral feedback
            similarity_adjustment = 0.0
        
        # Learn new similarity pattern
        self.learn_context_similarity(
            original_context,
            recommended_context,
            max(0.0, min(1.0, 0.5 + similarity_adjustment)),
            user_feedback
        )
    
    def _analyze_feedback_sentiment(self, feedback: str) -> float:
        """Simple sentiment analysis of user feedback"""
        positive_words = ['good', 'great', 'excellent', 'perfect', 'love', 'like', 'enjoy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'wrong', 'poor']
        
        feedback_lower = feedback.lower()
        
        positive_count = sum(1 for word in positive_words if word in feedback_lower)
        negative_count = sum(1 for word in negative_words if word in feedback_lower)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        return positive_count / (positive_count + negative_count)
    
    def save_similarity_data(self, filepath: str):
        """Save similarity learning data"""
        data = {
            'similarity_patterns': dict(self.similarity_patterns),
            'transfer_mappings': dict(self.transfer_mappings),
            'context_clusters': self.context_clusters
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_similarity_data(self, filepath: str):
        """Load similarity learning data"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.similarity_patterns = defaultdict(list, data['similarity_patterns'])
            self.transfer_mappings = defaultdict(dict, data['transfer_mappings'])
            self.context_clusters = data['context_clusters']
            
        except FileNotFoundError:
            print(f"Similarity data file {filepath} not found. Starting with empty data.")
        except Exception as e:
            print(f"Error loading similarity data: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize similarity learner
    similarity_learner = ContextSimilarityLearner()
    
    # Example contexts
    context1 = {"meal_time": "lunch", "occasion": "casual", "budget": 50}
    context2 = {"meal_time": "lunch", "occasion": "casual", "budget": 60}
    context3 = {"meal_time": "dinner", "occasion": "business", "budget": 100}
    
    # Learn similarity patterns
    similarity_learner.learn_context_similarity(context1, context2, 0.8, "Similar contexts")
    similarity_learner.learn_context_similarity(context1, context3, 0.3, "Different contexts")
    
    # Find similar contexts
    target_context = {"meal_time": "lunch", "occasion": "casual", "budget": 55}
    similar_contexts = similarity_learner.find_similar_contexts(
        target_context, [context1, context2, context3]
    )
    
    print("Similar contexts:", similar_contexts)
