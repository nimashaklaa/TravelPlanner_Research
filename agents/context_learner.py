"""
Novel Hierarchical Context Learning System for TravelPlanner
This module implements scalable context-aware learning that can handle any location globally.
"""

import json
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

@dataclass
class LocationContext:
    """Represents a location with hierarchical information"""
    neighborhood: str = "unknown"
    city: str = "unknown"
    region: str = "unknown"
    country: str = "unknown"
    continent: str = "unknown"
    coordinates: Tuple[float, float] = (0.0, 0.0)

@dataclass
class UserPreference:
    """Represents a user preference with context"""
    item: str
    context: Dict[str, Any]
    strength: float
    timestamp: float
    location_context: LocationContext
    frequency: int = 1

class HierarchicalContextLearner:
    """
    Novel hierarchical context learning system that can scale to any global location.
    This is a key research contribution - learning preferences at multiple hierarchical levels.
    """
    
    def __init__(self, learning_rate: float = 0.1, decay_rate: float = 0.05):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # Hierarchical preference storage
        self.context_hierarchy = {
            'global': defaultdict(dict),
            'continent': defaultdict(dict),
            'country': defaultdict(dict),
            'region': defaultdict(dict),
            'city': defaultdict(dict),
            'neighborhood': defaultdict(dict)
        }
        
        # User-specific preferences
        self.user_preferences = defaultdict(list)
        
        # Context similarity model
        self.context_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.context_embeddings = {}
        self.similarity_threshold = 0.7
        
        # Pattern discovery
        self.pattern_clusters = defaultdict(list)
        self.min_pattern_support = 3
        
    def extract_location_hierarchy(self, location: str) -> LocationContext:
        """
        Extract hierarchical location information using geocoding.
        This handles any location globally without hard-coding.
        """
        try:
            # Use a free geocoding service (you can replace with Google Maps API)
            response = requests.get(
                f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1",
                timeout=5  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    place = data[0]
                    return LocationContext(
                        neighborhood=place.get('name', 'unknown'),
                        city=place.get('address', {}).get('city', 'unknown'),
                        region=place.get('address', {}).get('state', 'unknown'),
                        country=place.get('address', {}).get('country', 'unknown'),
                        continent=self._get_continent(place.get('address', {}).get('country', 'unknown')),
                        coordinates=(float(place.get('lat', 0)), float(place.get('lon', 0)))
                    )
        except Exception as e:
            print(f"Geocoding error for {location}: {e}")
        
        # Fallback to basic parsing
        parts = location.split(',')
        return LocationContext(
            city=parts[0].strip() if len(parts) > 0 else location,
            country=parts[-1].strip() if len(parts) > 1 else 'unknown',
            continent=self._get_continent(parts[-1].strip() if len(parts) > 1 else 'unknown')
        )
    
    def _get_continent(self, country: str) -> str:
        """Map country to continent"""
        continent_map = {
            'USA': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
            'Brazil': 'South America', 'Argentina': 'South America', 'Chile': 'South America',
            'China': 'Asia', 'Japan': 'Asia', 'India': 'Asia', 'Thailand': 'Asia',
            'Germany': 'Europe', 'France': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe',
            'Australia': 'Oceania', 'New Zealand': 'Oceania',
            'Egypt': 'Africa', 'South Africa': 'Africa', 'Nigeria': 'Africa'
        }
        return continent_map.get(country, 'Unknown')
    
    def learn_preference(self, user_id: str, item: str, context: Dict[str, Any], 
                        location: str, feedback_strength: float = 1.0):
        """
        Learn user preferences with hierarchical context awareness.
        This is the core learning mechanism.
        """
        location_context = self.extract_location_hierarchy(location)
        
        # Create preference object
        preference = UserPreference(
            item=item,
            context=context,
            strength=feedback_strength,
            timestamp=time.time(),
            location_context=location_context
        )
        
        # Store user-specific preference
        self.user_preferences[user_id].append(preference)
        
        # Learn at each hierarchical level
        self._learn_at_level('global', 'earth', preference)
        self._learn_at_level('continent', location_context.continent, preference)
        self._learn_at_level('country', location_context.country, preference)
        self._learn_at_level('region', location_context.region, preference)
        self._learn_at_level('city', location_context.city, preference)
        self._learn_at_level('neighborhood', location_context.neighborhood, preference)
        
        # Update context embeddings for similarity matching
        self._update_context_embeddings(context)
        
        # Discover patterns
        self._discover_patterns(user_id, preference)
    
    def _learn_at_level(self, level: str, place: str, preference: UserPreference):
        """Learn preference at a specific hierarchical level"""
        if place == 'unknown':
            return
            
        context_key = self._create_context_key(preference.context)
        
        if context_key not in self.context_hierarchy[level][place]:
            self.context_hierarchy[level][place][context_key] = {
                'items': {},
                'total_interactions': 0,
                'last_updated': time.time()
            }
        
        level_data = self.context_hierarchy[level][place][context_key]
        
        # Update item preference
        if preference.item not in level_data['items']:
            level_data['items'][preference.item] = {
                'strength': 0,
                'frequency': 0,
                'last_seen': time.time()
            }
        
        item_data = level_data['items'][preference.item]
        
        # Apply learning with decay
        time_decay = self._calculate_time_decay(item_data['last_seen'])
        item_data['strength'] = (item_data['strength'] * time_decay + 
                               preference.strength * self.learning_rate) / (1 + self.learning_rate)
        item_data['frequency'] += 1
        item_data['last_seen'] = preference.timestamp
        
        level_data['total_interactions'] += 1
        level_data['last_updated'] = time.time()
    
    def _create_context_key(self, context: Dict[str, Any]) -> str:
        """Create a string key for context"""
        key_parts = []
        for key in sorted(context.keys()):
            key_parts.append(f"{key}:{context[key]}")
        return "|".join(key_parts)
    
    def _calculate_time_decay(self, last_seen: float) -> float:
        """Calculate time decay factor"""
        days_since = (time.time() - last_seen) / (24 * 3600)
        return math.exp(-self.decay_rate * days_since)
    
    def _update_context_embeddings(self, context: Dict[str, Any]):
        """Update context embeddings for similarity matching"""
        context_text = " ".join([f"{k}:{v}" for k, v in context.items()])
        
        if not hasattr(self, 'context_embeddings_fitted'):
            self.context_embeddings_fitted = True
            self.context_embeddings['dummy'] = context_text
        else:
            self.context_embeddings[context_text] = context_text
    
    def _discover_patterns(self, user_id: str, preference: UserPreference):
        """Discover patterns from user interactions"""
        # Group recent interactions by location similarity
        recent_preferences = [p for p in self.user_preferences[user_id] 
                            if time.time() - p.timestamp < 7 * 24 * 3600]  # Last 7 days
        
        if len(recent_preferences) >= self.min_pattern_support:
            # Find common context patterns
            context_patterns = self._extract_context_patterns(recent_preferences)
            
            for pattern in context_patterns:
                if pattern['frequency'] >= self.min_pattern_support:
                    self.pattern_clusters[user_id].append(pattern)
    
    def _extract_context_patterns(self, preferences: List[UserPreference]) -> List[Dict]:
        """Extract common context patterns from preferences"""
        context_combinations = defaultdict(int)
        
        for pref in preferences:
            context_key = self._create_context_key(pref.context)
            context_combinations[context_key] += 1
        
        patterns = []
        for context_key, frequency in context_combinations.items():
            if frequency >= self.min_pattern_support:
                patterns.append({
                    'context_key': context_key,
                    'frequency': frequency,
                    'confidence': frequency / len(preferences)
                })
        
        return patterns
    
    def get_recommendations(self, user_id: str, context: Dict[str, Any], 
                          location: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get recommendations using hierarchical context learning.
        This demonstrates the novel scalable approach.
        """
        location_context = self.extract_location_hierarchy(location)
        context_key = self._create_context_key(context)
        
        # Try to find recommendations at each level, starting from most specific
        for level in ['neighborhood', 'city', 'region', 'country', 'continent', 'global']:
            if level == 'global':
                place = 'earth'
            else:
                place = getattr(location_context, level)
            
            if place != 'unknown' and place in self.context_hierarchy[level]:
                recommendations = self._get_recommendations_at_level(
                    level, place, context_key, context
                )
                
                if recommendations:
                    # Apply diversity and freshness filters
                    filtered_recs = self._apply_filters(recommendations, user_id, context)
                    return filtered_recs[:top_k]
        
        # Fallback to user's personal preferences
        return self._get_personal_recommendations(user_id, context, top_k)
    
    def _get_recommendations_at_level(self, level: str, place: str, 
                                    context_key: str, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get recommendations at a specific hierarchical level"""
        if context_key in self.context_hierarchy[level][place]:
            # Exact context match
            items = self.context_hierarchy[level][place][context_key]['items']
            return [(item, data['strength']) for item, data in items.items()]
        
        # Find similar contexts
        similar_contexts = self._find_similar_contexts(
            context, self.context_hierarchy[level][place]
        )
        
        if similar_contexts:
            # Weight recommendations by similarity
            weighted_recommendations = defaultdict(float)
            for similar_context, similarity in similar_contexts:
                items = self.context_hierarchy[level][place][similar_context]['items']
                for item, data in items.items():
                    weighted_recommendations[item] += data['strength'] * similarity
            
            return list(weighted_recommendations.items())
        
        return []
    
    def _find_similar_contexts(self, target_context: Dict[str, Any], 
                             available_contexts: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Find contexts similar to target using embedding similarity"""
        if not self.context_embeddings:
            return []
        
        target_text = " ".join([f"{k}:{v}" for k, v in target_context.items()])
        
        # Create embeddings if not already done
        if not hasattr(self, 'context_embeddings_matrix'):
            all_contexts = list(self.context_embeddings.keys())
            if len(all_contexts) > 1:
                self.context_embeddings_matrix = self.context_vectorizer.fit_transform(all_contexts)
            else:
                return []
        
        # Calculate similarity
        target_embedding = self.context_vectorizer.transform([target_text])
        similarities = cosine_similarity(target_embedding, self.context_embeddings_matrix)[0]
        
        similar_contexts = []
        for i, (context_key, _) in enumerate(available_contexts.items()):
            if i < len(similarities) and similarities[i] > self.similarity_threshold:
                similar_contexts.append((context_key, similarities[i]))
        
        return similar_contexts
    
    def _apply_filters(self, recommendations: List[Tuple[str, float]], 
                      user_id: str, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Apply diversity and freshness filters to recommendations"""
        # Apply context-specific adjustments
        filtered_recs = []
        
        for item, score in recommendations:
            adjusted_score = score
            
            # Apply context-specific adjustments
            if context.get('meal_time') == 'breakfast' and 'pizza' in item.lower():
                adjusted_score *= 0.1  # Pizza less likely for breakfast
            elif context.get('occasion') == 'business' and 'pizza' in item.lower():
                adjusted_score *= 0.3  # Pizza less appropriate for business
            elif context.get('budget') == 'high' and 'fast food' in item.lower():
                adjusted_score *= 0.2  # Fast food less likely for high budget
            
            # Apply diversity bonus for less frequently recommended items
            diversity_bonus = self._calculate_diversity_bonus(item, user_id)
            adjusted_score *= (1 + diversity_bonus)
            
            filtered_recs.append((item, adjusted_score))
        
        # Sort by adjusted score
        return sorted(filtered_recs, key=lambda x: x[1], reverse=True)
    
    def _calculate_diversity_bonus(self, item: str, user_id: str) -> float:
        """Calculate diversity bonus to avoid over-recommending popular items"""
        user_items = [p.item for p in self.user_preferences[user_id]]
        item_frequency = user_items.count(item)
        
        if item_frequency == 0:
            return 0.2  # Bonus for new items
        elif item_frequency > 5:
            return -0.1  # Penalty for over-recommended items
        else:
            return 0.0
    
    def _get_personal_recommendations(self, user_id: str, context: Dict[str, Any], 
                                    top_k: int) -> List[Tuple[str, float]]:
        """Get recommendations based on user's personal preferences"""
        user_prefs = self.user_preferences[user_id]
        
        if not user_prefs:
            return []
        
        # Find similar contexts in user's history
        similar_prefs = []
        for pref in user_prefs:
            similarity = self._calculate_context_similarity(context, pref.context)
            if similarity > 0.5:
                similar_prefs.append((pref.item, pref.strength * similarity))
        
        # Group by item and sum scores
        item_scores = defaultdict(float)
        for item, score in similar_prefs:
            item_scores[item] += score
        
        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                   context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for key in common_keys if context1[key] == context2[key])
        return matches / len(common_keys)
    
    def save_learning_data(self, filepath: str):
        """Save learned data to file"""
        data = {
            'context_hierarchy': dict(self.context_hierarchy),
            'user_preferences': {k: [
                {
                    'item': p.item,
                    'context': p.context,
                    'strength': p.strength,
                    'timestamp': p.timestamp,
                    'location_context': {
                        'neighborhood': p.location_context.neighborhood,
                        'city': p.location_context.city,
                        'region': p.location_context.region,
                        'country': p.location_context.country,
                        'continent': p.location_context.continent,
                        'coordinates': p.location_context.coordinates
                    },
                    'frequency': p.frequency
                } for p in v
            ] for k, v in self.user_preferences.items()},
            'pattern_clusters': dict(self.pattern_clusters)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_learning_data(self, filepath: str):
        """Load learned data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore context hierarchy
            for level, places in data['context_hierarchy'].items():
                self.context_hierarchy[level] = defaultdict(dict, places)
            
            # Restore user preferences
            for user_id, prefs in data['user_preferences'].items():
                self.user_preferences[user_id] = [
                    UserPreference(
                        item=p['item'],
                        context=p['context'],
                        strength=p['strength'],
                        timestamp=p['timestamp'],
                        location_context=LocationContext(**p['location_context']),
                        frequency=p['frequency']
                    ) for p in prefs
                ]
            
            # Restore pattern clusters
            self.pattern_clusters = defaultdict(list, data['pattern_clusters'])
            
        except FileNotFoundError:
            print(f"Learning data file {filepath} not found. Starting with empty data.")
        except Exception as e:
            print(f"Error loading learning data: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the learning system
    learner = HierarchicalContextLearner()
    
    # Example: Learn from user feedback
    learner.learn_preference(
        user_id="user123",
        item="pizza",
        context={"meal_time": "lunch", "occasion": "casual", "budget": "medium"},
        location="New York, USA",
        feedback_strength=0.8
    )
    
    learner.learn_preference(
        user_id="user123",
        item="sushi",
        context={"meal_time": "dinner", "occasion": "casual", "budget": "high"},
        location="Tokyo, Japan",
        feedback_strength=0.9
    )
    
    # Get recommendations for a new location
    recommendations = learner.get_recommendations(
        user_id="user123",
        context={"meal_time": "lunch", "occasion": "casual", "budget": "medium"},
        location="San Francisco, USA"
    )
    
    print("Recommendations:", recommendations)
