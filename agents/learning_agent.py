"""
Enhanced Learning Agent for TravelPlanner
This module integrates all learning components to create a comprehensive learning system.
"""

import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np

from .context_learner import HierarchicalContextLearner, LocationContext
from .similarity_learner import ContextSimilarityLearner
from .pattern_discovery import DynamicPatternDiscovery, InteractionPattern

@dataclass
class LearningRecommendation:
    """Represents a learning-enhanced recommendation"""
    item: str
    confidence: float
    learning_source: str  # 'hierarchical', 'similarity', 'pattern', 'hybrid'
    context_reasoning: str
    personalization_factor: float

class EnhancedLearningAgent:
    """
    Enhanced Learning Agent that integrates all learning components.
    This is the main research contribution - a comprehensive learning system.
    """
    
    def __init__(self, learning_rate: float = 0.1, decay_rate: float = 0.05):
        # Initialize all learning components
        self.context_learner = HierarchicalContextLearner(learning_rate, decay_rate)
        self.similarity_learner = ContextSimilarityLearner()
        self.pattern_discovery = DynamicPatternDiscovery()
        
        # Learning configuration
        self.learning_enabled = True
        self.adaptation_threshold = 0.7
        self.personalization_weight = 0.6
        
        # Learning history
        self.learning_history = []
        self.performance_metrics = {
            'total_recommendations': 0,
            'successful_recommendations': 0,
            'user_satisfaction_avg': 0.0,
            'learning_improvements': 0
        }
    
    def learn_from_interaction(self, user_id: str, interaction: Dict[str, Any]):
        """
        Learn from a user interaction.
        This is the main learning entry point.
        """
        if not self.learning_enabled:
            return
        
        # Extract learning data
        context = interaction.get('context', {})
        items = interaction.get('items', [])
        location = interaction.get('location', 'unknown')
        feedback = interaction.get('feedback', '')
        satisfaction = interaction.get('satisfaction', 0.5)
        
        # Learn in all components
        self._learn_context_preferences(user_id, context, items, location, satisfaction)
        self._learn_similarity_patterns(user_id, context, feedback, satisfaction)
        self._learn_interaction_patterns(user_id, interaction)
        
        # Update performance metrics
        self._update_performance_metrics(satisfaction)
        
        # Store learning event
        self.learning_history.append({
            'timestamp': time.time(),
            'user_id': user_id,
            'interaction': interaction,
            'learning_components': ['context', 'similarity', 'pattern']
        })
    
    def _learn_context_preferences(self, user_id: str, context: Dict[str, Any], 
                                 items: List[str], location: str, satisfaction: float):
        """Learn context preferences using hierarchical learning"""
        for item in items:
            # Convert satisfaction to strength
            strength = self._satisfaction_to_strength(satisfaction)
            
            # Learn preference
            self.context_learner.learn_preference(
                user_id=user_id,
                item=item,
                context=context,
                location=location,
                feedback_strength=strength
            )
    
    def _learn_similarity_patterns(self, user_id: str, context: Dict[str, Any], 
                                 feedback: str, satisfaction: float):
        """Learn similarity patterns from user feedback"""
        if not feedback:
            return
        
        # Find similar contexts from user history
        user_contexts = self._get_user_context_history(user_id)
        
        for similar_context in user_contexts:
            # Learn similarity based on feedback
            similarity_score = self._extract_similarity_from_feedback(feedback, satisfaction)
            
            self.similarity_learner.learn_context_similarity(
                context1=context,
                context2=similar_context,
                similarity_score=similarity_score,
                user_feedback=feedback
            )
    
    def _learn_interaction_patterns(self, user_id: str, interaction: Dict[str, Any]):
        """Learn interaction patterns using pattern discovery"""
        self.pattern_discovery.add_interaction(user_id, interaction)
    
    def get_learning_recommendations(self, user_id: str, context: Dict[str, Any], 
                                   location: str, top_k: int = 5) -> List[LearningRecommendation]:
        """
        Get recommendations using all learning components.
        This demonstrates the novel hybrid learning approach.
        """
        recommendations = []
        
        # Get recommendations from each learning component
        hierarchical_recs = self._get_hierarchical_recommendations(user_id, context, location, top_k)
        similarity_recs = self._get_similarity_recommendations(user_id, context, top_k)
        pattern_recs = self._get_pattern_recommendations(user_id, context, top_k)
        
        # Combine recommendations using ensemble learning
        combined_recs = self._combine_recommendations(
            hierarchical_recs, similarity_recs, pattern_recs, top_k
        )
        
        # Convert to learning recommendations
        for item, confidence, source, reasoning in combined_recs:
            recommendation = LearningRecommendation(
                item=item,
                confidence=confidence,
                learning_source=source,
                context_reasoning=reasoning,
                personalization_factor=self._calculate_personalization_factor(user_id, item)
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_hierarchical_recommendations(self, user_id: str, context: Dict[str, Any], 
                                        location: str, top_k: int) -> List[Tuple[str, float, str, str]]:
        """Get recommendations from hierarchical context learning"""
        try:
            recs = self.context_learner.get_recommendations(user_id, context, location, top_k)
            return [(item, score, 'hierarchical', f'Learned from {location} context') 
                   for item, score in recs]
        except Exception as e:
            print(f"Error in hierarchical recommendations: {e}")
            return []
    
    def _get_similarity_recommendations(self, user_id: str, context: Dict[str, Any], 
                                      top_k: int) -> List[Tuple[str, float, str, str]]:
        """Get recommendations from similarity learning"""
        try:
            # Get user's context history
            user_contexts = self._get_user_context_history(user_id)
            
            if not user_contexts:
                return []
            
            # Find similar contexts
            similar_contexts = self.similarity_learner.find_similar_contexts(
                context, user_contexts
            )
            
            # Extract items from similar contexts
            recommendations = []
            for similar_context, similarity in similar_contexts[:top_k]:
                # Get items from this context
                items = self._get_items_from_context(user_id, similar_context)
                for item in items:
                    recommendations.append((
                        item, similarity, 'similarity', 
                        f'Similar to past context with {similarity:.2f} similarity'
                    ))
            
            return recommendations[:top_k]
        except Exception as e:
            print(f"Error in similarity recommendations: {e}")
            return []
    
    def _get_pattern_recommendations(self, user_id: str, context: Dict[str, Any], 
                                   top_k: int) -> List[Tuple[str, float, str, str]]:
        """Get recommendations from pattern discovery"""
        try:
            pattern_recs = self.pattern_discovery.get_pattern_recommendations(
                user_id, context, top_k
            )
            
            recommendations = []
            for pattern, score in pattern_recs:
                for item in pattern.items:
                    recommendations.append((
                        item, score, 'pattern',
                        f'From discovered pattern: {pattern.pattern_id}'
                    ))
            
            return recommendations[:top_k]
        except Exception as e:
            print(f"Error in pattern recommendations: {e}")
            return []
    
    def _combine_recommendations(self, hierarchical_recs: List[Tuple[str, float, str, str]], 
                               similarity_recs: List[Tuple[str, float, str, str]],
                               pattern_recs: List[Tuple[str, float, str, str]], 
                               top_k: int) -> List[Tuple[str, float, str, str]]:
        """Combine recommendations from all learning components using ensemble learning"""
        # Collect all recommendations
        all_recs = hierarchical_recs + similarity_recs + pattern_recs
        
        # Group by item
        item_scores = {}
        item_sources = {}
        item_reasonings = {}
        
        for item, score, source, reasoning in all_recs:
            if item not in item_scores:
                item_scores[item] = []
                item_sources[item] = []
                item_reasonings[item] = []
            
            item_scores[item].append(score)
            item_sources[item].append(source)
            item_reasonings[item].append(reasoning)
        
        # Calculate ensemble scores
        ensemble_recs = []
        for item in item_scores:
            scores = item_scores[item]
            sources = item_sources[item]
            reasonings = item_reasonings[item]
            
            # Weighted ensemble (could be learned)
            weights = {'hierarchical': 0.4, 'similarity': 0.3, 'pattern': 0.3}
            
            ensemble_score = 0.0
            total_weight = 0.0
            combined_reasoning = []
            
            for i, (score, source) in enumerate(zip(scores, sources)):
                weight = weights.get(source, 0.1)
                ensemble_score += score * weight
                total_weight += weight
                combined_reasoning.append(f"{source}: {reasonings[i]}")
            
            if total_weight > 0:
                ensemble_score /= total_weight
                ensemble_recs.append((
                    item, ensemble_score, 'hybrid',
                    '; '.join(combined_reasoning)
                ))
        
        # Sort by ensemble score and return top-k
        return sorted(ensemble_recs, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _satisfaction_to_strength(self, satisfaction: float) -> float:
        """Convert satisfaction score to learning strength"""
        # Map satisfaction [0, 1] to strength [0, 1]
        if satisfaction >= 0.8:
            return 1.0
        elif satisfaction >= 0.6:
            return 0.7
        elif satisfaction >= 0.4:
            return 0.4
        else:
            return 0.1
    
    def _extract_similarity_from_feedback(self, feedback: str, satisfaction: float) -> float:
        """Extract similarity score from user feedback"""
        # Simple heuristic - could be improved with NLP
        if satisfaction >= 0.8:
            return 0.8
        elif satisfaction >= 0.6:
            return 0.6
        elif satisfaction >= 0.4:
            return 0.4
        else:
            return 0.2
    
    def _get_user_context_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's context history for similarity learning"""
        # This would typically come from a database
        # For now, return empty list
        return []
    
    def _get_items_from_context(self, user_id: str, context: Dict[str, Any]) -> List[str]:
        """Get items associated with a context"""
        # This would typically come from a database
        # For now, return empty list
        return []
    
    def _calculate_personalization_factor(self, user_id: str, item: str) -> float:
        """Calculate how personalized a recommendation is"""
        # This could be based on user's interaction history with the item
        # For now, return a random factor
        return np.random.uniform(0.5, 1.0)
    
    def _update_performance_metrics(self, satisfaction: float):
        """Update learning performance metrics"""
        self.performance_metrics['total_recommendations'] += 1
        
        if satisfaction >= 0.6:
            self.performance_metrics['successful_recommendations'] += 1
        
        # Update average satisfaction
        total = self.performance_metrics['total_recommendations']
        current_avg = self.performance_metrics['user_satisfaction_avg']
        self.performance_metrics['user_satisfaction_avg'] = (
            (current_avg * (total - 1) + satisfaction) / total
        )
    
    def get_learning_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive learning insights for a user"""
        insights = {
            'user_id': user_id,
            'learning_status': 'active' if self.learning_enabled else 'inactive',
            'performance_metrics': self.performance_metrics,
            'pattern_insights': self.pattern_discovery.get_pattern_insights(user_id),
            'learning_history_count': len(self.learning_history),
            'recommendation_sources': self._get_recommendation_sources(user_id)
        }
        
        return insights
    
    def _get_recommendation_sources(self, user_id: str) -> Dict[str, int]:
        """Get count of recommendations by source for a user"""
        sources = {'hierarchical': 0, 'similarity': 0, 'pattern': 0, 'hybrid': 0}
        
        # This would typically come from a database
        # For now, return empty counts
        return sources
    
    def adapt_learning_parameters(self, user_id: str, performance_threshold: float = 0.7):
        """Adapt learning parameters based on user performance"""
        if not self.learning_enabled:
            return
        
        # Get user's performance
        user_performance = self._get_user_performance(user_id)
        
        if user_performance < performance_threshold:
            # Increase learning rate for better adaptation
            self.context_learner.learning_rate = min(0.2, self.context_learner.learning_rate * 1.1)
            self.performance_metrics['learning_improvements'] += 1
        else:
            # Decrease learning rate for stability
            self.context_learner.learning_rate = max(0.05, self.context_learner.learning_rate * 0.95)
    
    def _get_user_performance(self, user_id: str) -> float:
        """Get user's learning performance score"""
        # This would typically calculate based on user's satisfaction history
        # For now, return a random score
        return np.random.uniform(0.5, 1.0)
    
    def save_learning_data(self, filepath: str):
        """Save all learning data"""
        # Save context learning data
        self.context_learner.save_learning_data(f"{filepath}_context.json")
        
        # Save similarity learning data
        self.similarity_learner.save_similarity_data(f"{filepath}_similarity.json")
        
        # Save pattern discovery data
        self.pattern_discovery.save_patterns(f"{filepath}_patterns.json")
        
        # Save learning agent data
        agent_data = {
            'learning_enabled': self.learning_enabled,
            'performance_metrics': self.performance_metrics,
            'learning_history': self.learning_history[-100:]  # Keep last 100 entries
        }
        
        with open(f"{filepath}_agent.json", 'w') as f:
            json.dump(agent_data, f, indent=2)
    
    def load_learning_data(self, filepath: str):
        """Load all learning data"""
        try:
            # Load context learning data
            self.context_learner.load_learning_data(f"{filepath}_context.json")
            
            # Load similarity learning data
            self.similarity_learner.load_similarity_data(f"{filepath}_similarity.json")
            
            # Load pattern discovery data
            self.pattern_discovery.load_patterns(f"{filepath}_patterns.json")
            
            # Load learning agent data
            with open(f"{filepath}_agent.json", 'r') as f:
                agent_data = json.load(f)
            
            self.learning_enabled = agent_data.get('learning_enabled', True)
            self.performance_metrics = agent_data.get('performance_metrics', self.performance_metrics)
            self.learning_history = agent_data.get('learning_history', [])
            
        except FileNotFoundError:
            print(f"Learning data files not found. Starting with empty data.")
        except Exception as e:
            print(f"Error loading learning data: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize enhanced learning agent
    learning_agent = EnhancedLearningAgent()
    
    # Example interaction
    interaction = {
        'user_id': 'user123',
        'context': {'meal_time': 'lunch', 'occasion': 'casual', 'budget': 50},
        'items': ['pizza', 'salad'],
        'location': 'New York, USA',
        'feedback': 'Great recommendation!',
        'satisfaction': 0.9
    }
    
    # Learn from interaction
    learning_agent.learn_from_interaction(interaction['user_id'], interaction)
    
    # Get recommendations
    recommendations = learning_agent.get_learning_recommendations(
        user_id='user123',
        context={'meal_time': 'dinner', 'occasion': 'casual', 'budget': 60},
        location='San Francisco, USA',
        top_k=3
    )
    
    print("Learning recommendations:")
    for rec in recommendations:
        print(f"- {rec.item}: {rec.confidence:.2f} ({rec.learning_source})")
        print(f"  Reasoning: {rec.context_reasoning}")
        print(f"  Personalization: {rec.personalization_factor:.2f}")
        print()
