"""
Dynamic Pattern Discovery System for TravelPlanner
This module implements novel unsupervised pattern discovery from user interactions.
"""

import json
import time
from typing import Dict, List, Tuple, Any, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx

@dataclass
class InteractionPattern:
    """Represents a discovered interaction pattern"""
    pattern_id: str
    context_combination: Dict[str, Any]
    items: List[str]
    frequency: int
    confidence: float
    locations: List[str]
    time_patterns: Dict[str, Any]
    user_demographics: Dict[str, Any]

class DynamicPatternDiscovery:
    """
    Novel dynamic pattern discovery system that automatically finds
    meaningful patterns in user interactions without supervision.
    """
    
    def __init__(self, min_support: int = 3, confidence_threshold: float = 0.6):
        self.min_support = min_support
        self.confidence_threshold = confidence_threshold
        
        # Pattern storage
        self.discovered_patterns = {}
        self.pattern_graph = nx.Graph()
        self.pattern_evolution = defaultdict(list)
        
        # User interaction history
        self.user_interactions = defaultdict(list)
        self.interaction_features = defaultdict(list)
        
        # Pattern mining parameters
        self.temporal_window = 7 * 24 * 3600  # 7 days
        self.spatial_clustering_eps = 0.1
        self.temporal_clustering_eps = 3600  # 1 hour
        
    def add_interaction(self, user_id: str, interaction: Dict[str, Any]):
        """
        Add a new user interaction for pattern discovery.
        This is the input to the pattern discovery system.
        """
        interaction['timestamp'] = time.time()
        interaction['user_id'] = user_id
        
        self.user_interactions[user_id].append(interaction)
        
        # Extract features for pattern mining
        features = self._extract_interaction_features(interaction)
        self.interaction_features[user_id].append(features)
        
        # Trigger pattern discovery more frequently for demo purposes
        if len(self.user_interactions[user_id]) >= self.min_support:
            self.discover_patterns_for_user(user_id)
    
    def _extract_interaction_features(self, interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from interaction for pattern mining"""
        features = {
            'context': interaction.get('context', {}),
            'items': interaction.get('items', []),
            'location': interaction.get('location', 'unknown'),
            'timestamp': interaction.get('timestamp', time.time()),
            'satisfaction': interaction.get('satisfaction', 0.5),
            'budget': interaction.get('budget', 0),
            'group_size': interaction.get('group_size', 1)
        }
        
        # Add derived features
        features['hour_of_day'] = time.localtime(features['timestamp']).tm_hour
        features['day_of_week'] = time.localtime(features['timestamp']).tm_wday
        features['is_weekend'] = features['day_of_week'] >= 5
        
        return features
    
    def discover_patterns_for_user(self, user_id: str):
        """Discover patterns for a specific user"""
        if user_id not in self.user_interactions:
            return
        
        interactions = self.user_interactions[user_id]
        features = self.interaction_features[user_id]
        
        # Discover different types of patterns
        context_patterns = self._discover_context_patterns(interactions)
        temporal_patterns = self._discover_temporal_patterns(features)
        spatial_patterns = self._discover_spatial_patterns(features)
        item_patterns = self._discover_item_patterns(interactions)
        
        # Combine patterns
        combined_patterns = self._combine_patterns(
            context_patterns, temporal_patterns, spatial_patterns, item_patterns
        )
        
        # Store discovered patterns
        for pattern in combined_patterns:
            self._store_pattern(user_id, pattern)
        
        # If no patterns were discovered, create a basic pattern from interactions
        if not combined_patterns and len(interactions) >= self.min_support:
            basic_pattern = self._create_basic_pattern(user_id, interactions)
            if basic_pattern:
                self._store_pattern(user_id, basic_pattern)
    
    def _discover_context_patterns(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover patterns in context combinations"""
        context_combinations = defaultdict(int)
        context_items = defaultdict(set)
        
        for interaction in interactions:
            context = interaction.get('context', {})
            items = interaction.get('items', [])
            
            # Create context signature
            context_sig = self._create_context_signature(context)
            context_combinations[context_sig] += 1
            
            for item in items:
                context_items[context_sig].add(item)
        
        # Find frequent context patterns
        patterns = []
        for context_sig, frequency in context_combinations.items():
            if frequency >= self.min_support:
                pattern = {
                    'type': 'context',
                    'context_signature': context_sig,
                    'frequency': frequency,
                    'items': list(context_items[context_sig]),
                    'confidence': frequency / len(interactions)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _discover_temporal_patterns(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover temporal patterns in user behavior"""
        if len(features) < self.min_support:
            return []
        
        # Extract temporal features
        temporal_data = []
        for feature in features:
            temporal_data.append([
                feature['hour_of_day'],
                feature['day_of_week'],
                feature['is_weekend']
            ])
        
        # Cluster temporal patterns
        if len(temporal_data) >= self.min_support:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(temporal_data)
            
            clustering = DBSCAN(eps=self.temporal_clustering_eps, min_samples=self.min_support)
            cluster_labels = clustering.fit_predict(scaled_data)
            
            # Extract patterns from clusters
            patterns = []
            for cluster_id in set(cluster_labels):
                if cluster_id == -1:  # Noise
                    continue
                
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_features = [features[i] for i in cluster_indices]
                
                pattern = self._create_temporal_pattern(cluster_features, cluster_id)
                patterns.append(pattern)
            
            return patterns
        
        return []
    
    def _discover_spatial_patterns(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover spatial patterns in user behavior"""
        location_counts = Counter(feature['location'] for feature in features)
        
        patterns = []
        for location, count in location_counts.items():
            if count >= self.min_support:
                location_features = [f for f in features if f['location'] == location]
                
                pattern = {
                    'type': 'spatial',
                    'location': location,
                    'frequency': count,
                    'common_items': self._find_common_items(location_features),
                    'time_patterns': self._extract_time_patterns(location_features),
                    'confidence': count / len(features)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _discover_item_patterns(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover patterns in item combinations"""
        item_combinations = defaultdict(int)
        item_contexts = defaultdict(set)
        
        for interaction in interactions:
            items = interaction.get('items', [])
            context = interaction.get('context', {})
            
            if len(items) > 1:
                # Find all item pairs
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pair = tuple(sorted([items[i], items[j]]))
                        item_combinations[pair] += 1
                        item_contexts[pair].add(self._create_context_signature(context))
            
            # Also track individual items
            for item in items:
                item_combinations[(item,)] += 1
                item_contexts[(item,)].add(self._create_context_signature(context))
        
        # Find frequent item patterns
        patterns = []
        for items, frequency in item_combinations.items():
            if frequency >= self.min_support:
                pattern = {
                    'type': 'item',
                    'items': list(items),
                    'frequency': frequency,
                    'contexts': list(item_contexts[items]),
                    'confidence': frequency / len(interactions)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _create_basic_pattern(self, user_id: str, interactions: List[Dict[str, Any]]) -> Optional[InteractionPattern]:
        """Create a basic pattern from user interactions when no complex patterns are found"""
        if len(interactions) < self.min_support:
            return None
        
        # Extract common items
        all_items = []
        for interaction in interactions:
            all_items.extend(interaction.get('items', []))
        
        item_counts = Counter(all_items)
        common_items = [item for item, count in item_counts.most_common(3)]
        
        # Extract common context
        contexts = [interaction.get('context', {}) for interaction in interactions]
        context_counts = Counter(self._create_context_signature(ctx) for ctx in contexts)
        most_common_context = context_counts.most_common(1)[0][0] if context_counts else "unknown"
        
        # Extract locations
        locations = list(set(interaction.get('location', 'unknown') for interaction in interactions))
        
        # Create basic pattern
        pattern = InteractionPattern(
            pattern_id=f"basic_pattern_{user_id}_{int(time.time())}",
            context_combination=most_common_context,
            items=common_items,
            frequency=len(interactions),
            confidence=0.5,  # Basic confidence
            locations=locations,
            time_patterns={},
            user_demographics={}
        )
        
        return pattern
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Create a signature for context combination"""
        return "|".join([f"{k}:{v}" for k, v in sorted(context.items())])
    
    def _create_temporal_pattern(self, cluster_features: List[Dict[str, Any]], 
                               cluster_id: int) -> Dict[str, Any]:
        """Create a temporal pattern from clustered features"""
        hours = [f['hour_of_day'] for f in cluster_features]
        days = [f['day_of_week'] for f in cluster_features]
        weekends = [f['is_weekend'] for f in cluster_features]
        
        return {
            'type': 'temporal',
            'cluster_id': cluster_id,
            'frequency': len(cluster_features),
            'avg_hour': np.mean(hours),
            'common_days': Counter(days).most_common(3),
            'weekend_ratio': sum(weekends) / len(weekends),
            'confidence': len(cluster_features) / len(self.user_interactions[list(self.user_interactions.keys())[0]])
        }
    
    def _find_common_items(self, features: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Find common items in a set of features"""
        all_items = []
        for feature in features:
            all_items.extend(feature.get('items', []))
        
        return Counter(all_items).most_common(5)
    
    def _extract_time_patterns(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract time patterns from features"""
        hours = [f['hour_of_day'] for f in features]
        days = [f['day_of_week'] for f in features]
        
        return {
            'peak_hour': Counter(hours).most_common(1)[0][0] if hours else 12,
            'common_days': Counter(days).most_common(3),
            'time_variance': np.var(hours) if len(hours) > 1 else 0
        }
    
    def _combine_patterns(self, context_patterns: List[Dict[str, Any]], 
                         temporal_patterns: List[Dict[str, Any]],
                         spatial_patterns: List[Dict[str, Any]],
                         item_patterns: List[Dict[str, Any]]) -> List[InteractionPattern]:
        """Combine different types of patterns into comprehensive patterns"""
        combined_patterns = []
        
        # Create comprehensive patterns by combining different pattern types
        for context_pattern in context_patterns:
            pattern_id = f"pattern_{len(combined_patterns)}"
            
            # Find related temporal and spatial patterns
            related_temporal = self._find_related_patterns(context_pattern, temporal_patterns)
            related_spatial = self._find_related_patterns(context_pattern, spatial_patterns)
            related_items = self._find_related_patterns(context_pattern, item_patterns)
            
            # Create combined pattern
            combined_pattern = InteractionPattern(
                pattern_id=pattern_id,
                context_combination=context_pattern.get('context_signature', {}),
                items=context_pattern.get('items', []),
                frequency=context_pattern.get('frequency', 0),
                confidence=context_pattern.get('confidence', 0.0),
                locations=[p.get('location', 'unknown') for p in related_spatial],
                time_patterns=related_temporal[0] if related_temporal else {},
                user_demographics={}  # Could be extended with user demographics
            )
            
            combined_patterns.append(combined_pattern)
        
        return combined_patterns
    
    def _find_related_patterns(self, base_pattern: Dict[str, Any], 
                             other_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find patterns related to the base pattern"""
        related = []
        
        for pattern in other_patterns:
            # Simple relatedness check - could be made more sophisticated
            if pattern.get('frequency', 0) >= self.min_support:
                related.append(pattern)
        
        return related
    
    def _store_pattern(self, user_id: str, pattern: InteractionPattern):
        """Store a discovered pattern"""
        pattern_key = f"{user_id}_{pattern.pattern_id}"
        self.discovered_patterns[pattern_key] = pattern
        
        # Add to pattern graph for relationship discovery
        self.pattern_graph.add_node(pattern_key, pattern=pattern)
        
        # Track pattern evolution
        self.pattern_evolution[pattern_key].append({
            'timestamp': time.time(),
            'frequency': pattern.frequency,
            'confidence': pattern.confidence
        })
    
    def get_pattern_recommendations(self, user_id: str, current_context: Dict[str, Any], 
                                  top_k: int = 5) -> List[Tuple[InteractionPattern, float]]:
        """
        Get pattern-based recommendations for a user.
        This demonstrates the novel pattern discovery capability.
        """
        user_patterns = [p for key, p in self.discovered_patterns.items() 
                        if key.startswith(f"{user_id}_")]
        
        if not user_patterns:
            return []
        
        # Score patterns based on context similarity
        scored_patterns = []
        for pattern in user_patterns:
            score = self._calculate_pattern_score(pattern, current_context)
            if score > 0.3:  # Minimum relevance threshold
                scored_patterns.append((pattern, score))
        
        # Sort by score and return top-k
        return sorted(scored_patterns, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _calculate_pattern_score(self, pattern: InteractionPattern, 
                               current_context: Dict[str, Any]) -> float:
        """Calculate relevance score for a pattern given current context"""
        score = 0.0
        
        # Context similarity
        context_similarity = self._calculate_context_similarity(
            pattern.context_combination, current_context
        )
        score += 0.4 * context_similarity
        
        # Pattern confidence
        score += 0.3 * pattern.confidence
        
        # Frequency (normalized)
        normalized_frequency = min(1.0, pattern.frequency / 10.0)
        score += 0.2 * normalized_frequency
        
        # Time relevance (if temporal patterns exist)
        if pattern.time_patterns:
            time_relevance = self._calculate_time_relevance(pattern.time_patterns)
            score += 0.1 * time_relevance
        
        return score
    
    def _calculate_context_similarity(self, pattern_context: str, 
                                    current_context: Dict[str, Any]) -> float:
        """Calculate similarity between pattern context and current context"""
        current_sig = self._create_context_signature(current_context)
        
        # Simple string similarity - could be improved with embeddings
        if pattern_context == current_sig:
            return 1.0
        
        # Calculate Jaccard similarity
        pattern_set = set(pattern_context.split('|'))
        current_set = set(current_sig.split('|'))
        
        intersection = len(pattern_set & current_set)
        union = len(pattern_set | current_set)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_time_relevance(self, time_patterns: Dict[str, Any]) -> float:
        """Calculate time relevance of a pattern"""
        current_hour = time.localtime().tm_hour
        current_day = time.localtime().tm_wday
        
        # Check hour relevance
        peak_hour = time_patterns.get('peak_hour', 12)
        hour_diff = abs(current_hour - peak_hour)
        hour_relevance = max(0, 1 - hour_diff / 12.0)
        
        # Check day relevance
        common_days = [day for day, _ in time_patterns.get('common_days', [])]
        day_relevance = 1.0 if current_day in common_days else 0.0
        
        return 0.7 * hour_relevance + 0.3 * day_relevance
    
    def get_pattern_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about discovered patterns for a user"""
        user_patterns = [p for key, p in self.discovered_patterns.items() 
                        if key.startswith(f"{user_id}_")]
        
        if not user_patterns:
            return {
                "message": "No patterns discovered yet",
                "total_patterns": 0,
                "most_frequent_items": [],
                "common_contexts": [],
                "temporal_insights": {},
                "spatial_insights": {}
            }
        
        insights = {
            'total_patterns': len(user_patterns),
            'most_frequent_items': self._get_most_frequent_items(user_patterns),
            'common_contexts': self._get_common_contexts(user_patterns),
            'temporal_insights': self._get_temporal_insights(user_patterns),
            'spatial_insights': self._get_spatial_insights(user_patterns)
        }
        
        return insights
    
    def _get_most_frequent_items(self, patterns: List[InteractionPattern]) -> List[Tuple[str, int]]:
        """Get most frequent items across patterns"""
        item_counts = Counter()
        for pattern in patterns:
            for item in pattern.items:
                item_counts[item] += pattern.frequency
        
        return item_counts.most_common(10)
    
    def _get_common_contexts(self, patterns: List[InteractionPattern]) -> List[str]:
        """Get most common context combinations"""
        context_counts = Counter()
        for pattern in patterns:
            context_counts[pattern.context_combination] += pattern.frequency
        
        return [context for context, _ in context_counts.most_common(5)]
    
    def _get_temporal_insights(self, patterns: List[InteractionPattern]) -> Dict[str, Any]:
        """Get temporal insights from patterns"""
        all_hours = []
        all_days = []
        
        for pattern in patterns:
            if pattern.time_patterns:
                all_hours.append(pattern.time_patterns.get('peak_hour', 12))
                for day, _ in pattern.time_patterns.get('common_days', []):
                    all_days.append(day)
        
        return {
            'peak_hour': Counter(all_hours).most_common(1)[0][0] if all_hours else 12,
            'common_days': Counter(all_days).most_common(3),
            'activity_variance': np.var(all_hours) if len(all_hours) > 1 else 0
        }
    
    def _get_spatial_insights(self, patterns: List[InteractionPattern]) -> Dict[str, Any]:
        """Get spatial insights from patterns"""
        location_counts = Counter()
        for pattern in patterns:
            for location in pattern.locations:
                location_counts[location] += pattern.frequency
        
        return {
            'most_common_locations': location_counts.most_common(5),
            'location_diversity': len(location_counts)
        }
    
    def save_patterns(self, filepath: str):
        """Save discovered patterns to file"""
        data = {
            'discovered_patterns': {
                key: {
                    'pattern_id': pattern.pattern_id,
                    'context_combination': pattern.context_combination,
                    'items': pattern.items,
                    'frequency': pattern.frequency,
                    'confidence': pattern.confidence,
                    'locations': pattern.locations,
                    'time_patterns': pattern.time_patterns,
                    'user_demographics': pattern.user_demographics
                } for key, pattern in self.discovered_patterns.items()
            },
            'pattern_evolution': dict(self.pattern_evolution)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_patterns(self, filepath: str):
        """Load discovered patterns from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore patterns
            for key, pattern_data in data['discovered_patterns'].items():
                pattern = InteractionPattern(
                    pattern_id=pattern_data['pattern_id'],
                    context_combination=pattern_data['context_combination'],
                    items=pattern_data['items'],
                    frequency=pattern_data['frequency'],
                    confidence=pattern_data['confidence'],
                    locations=pattern_data['locations'],
                    time_patterns=pattern_data['time_patterns'],
                    user_demographics=pattern_data['user_demographics']
                )
                self.discovered_patterns[key] = pattern
            
            # Restore pattern evolution
            self.pattern_evolution = defaultdict(list, data['pattern_evolution'])
            
        except FileNotFoundError:
            print(f"Pattern data file {filepath} not found. Starting with empty data.")
        except Exception as e:
            print(f"Error loading pattern data: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize pattern discovery
    pattern_discovery = DynamicPatternDiscovery()
    
    # Example interactions
    interactions = [
        {
            'user_id': 'user123',
            'context': {'meal_time': 'lunch', 'occasion': 'casual'},
            'items': ['pizza', 'salad'],
            'location': 'New York',
            'satisfaction': 0.8
        },
        {
            'user_id': 'user123',
            'context': {'meal_time': 'dinner', 'occasion': 'casual'},
            'items': ['pizza', 'beer'],
            'location': 'New York',
            'satisfaction': 0.9
        }
    ]
    
    # Add interactions
    for interaction in interactions:
        pattern_discovery.add_interaction(interaction['user_id'], interaction)
    
    # Get pattern insights
    insights = pattern_discovery.get_pattern_insights('user123')
    print("Pattern insights:", insights)
