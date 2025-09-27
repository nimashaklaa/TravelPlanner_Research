from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd

class suggestion_agent_output(TypedDict):
    suggestions: str
    category: str
    location: str
    count: int


def analyze_query(query):
    """
    Analyze the user query to determine what type of suggestions they want
    """
    # Handle empty or None query
    if not query or not isinstance(query, str):
        return 'restaurant', 'New York'
    
    query_lower = query.lower()
    
    # Determine suggestion type
    if any(word in query_lower for word in ['dinner', 'restaurant', 'food', 'eat', 'dining', 'lunch', 'breakfast']):
        suggestion_type = 'restaurant'
    elif any(word in query_lower for word in ['hotel', 'accommodation', 'stay', 'lodging']):
        suggestion_type = 'hotel'
    elif any(word in query_lower for word in ['attraction', 'sightseeing', 'tourist', 'visit', 'place', 'activity']):
        suggestion_type = 'attraction'
    elif any(word in query_lower for word in ['city', 'cities', 'town']):
        suggestion_type = 'city'
    else:
        suggestion_type = 'restaurant'  # Default to restaurant if unclear
    
    # Extract location - look for common city patterns
    words = query.split()
    potential_cities = []
    
    # Look for capitalized words that could be city names
    for word in words:
        if not word:  # Skip empty words
            continue
        clean_word = word.strip('.,!?;:')
        # Check if clean_word is not empty and has at least one character
        if clean_word and len(clean_word) > 2 and clean_word[0].isupper() and clean_word.isalpha():
            potential_cities.append(clean_word)
    
    # If no city found, try to extract from common patterns
    if not potential_cities:
        # Look for "in [City]" pattern
        import re
        in_pattern = r'in\s+([A-Z][a-z]+)'
        match = re.search(in_pattern, query, re.IGNORECASE)
        if match:
            potential_cities.append(match.group(1))
    
    location = potential_cities[0] if potential_cities else "New York"  # Default location
    
    return suggestion_type, location

def suggestion_agent(query):
    """
    Suggestion agent that provides recommendations for attractions, hotels, restaurants, etc.
    in a specific area without creating full travel plans.
    """
    try:
        # Analyze the query to understand what the user wants
        suggestion_type, location = analyze_query(query)
    except Exception as e:
        print(f"Error analyzing query: {e}")
        suggestion_type, location = 'restaurant', 'New York'  # Default fallback
    
    # Loading Data Sets
    restaurants_data_path = "./database/restaurants/clean_restaurant_2022.csv"
    restaurants_data = pd.read_csv(restaurants_data_path).dropna()[['Name','Average Cost','Cuisines','Aggregate Rating','City']]
    print("Restaurants data loaded.")

    accommodations_data_path = "./database/accommodations/clean_accommodations_2022.csv"
    accommodations_data = pd.read_csv(accommodations_data_path).dropna()[['NAME','price','room type', 'house_rules', 'minimum nights', 'maximum occupancy', 'review rate number', 'city']]
    print("Accommodation data loaded.")

    attractions_data_path = "./database/attractions/attractions.csv"
    attractions_data = pd.read_csv(attractions_data_path).dropna()[['Name',"City"]]
    print("Attractions data loaded.")

    city_data_path = "./database/background/citySet_with_states.txt"
    cityStateMapping = open(city_data_path, "r").read().strip().split("\n")
    city_data = {}
    for unit in cityStateMapping:
        city, state = unit.split("\t")
        if state not in city_data:
            city_data[state] = [city]
        else:
            city_data[state].append(city)
    print("City data loaded.")

    # Define Tools for Suggestions
    @tool
    def GetAttractionSuggestions(city: str, count: int = 5) -> str:
        """
        Get attraction suggestions for a specific city.
        Parameters:
            city: The name of the city
            count: Number of suggestions to return (default: 5)
        """
        results = attractions_data[attractions_data["City"] == city]
        if len(results) == 0:
            return f"No attractions found in {city}."
        
        # Get top suggestions based on count
        top_attractions = results.head(count)
        suggestions = []
        for _, attraction in top_attractions.iterrows():
            suggestions.append(f"• {attraction['Name']}")
        
        return f"🏛️ **Attractions in {city}:**\n" + "\n".join(suggestions)

    @tool
    def GetHotelSuggestions(city: str, count: int = 5, budget_range: str = "any") -> str:
        """
        Get hotel suggestions for a specific city.
        Parameters:
            city: The name of the city
            count: Number of suggestions to return (default: 5)
            budget_range: Budget preference - "budget", "mid", "luxury", or "any"
        """
        results = accommodations_data[accommodations_data["city"] == city]
        if len(results) == 0:
            return f"No hotels found in {city}."
        
        # Filter by budget if specified
        if budget_range != "any":
            if budget_range == "budget":
                results = results[results['price'] < 100]
            elif budget_range == "mid":
                results = results[(results['price'] >= 100) & (results['price'] < 300)]
            elif budget_range == "luxury":
                results = results[results['price'] >= 300]
        
        # Sort by review rating and name for consistent results
        results = results.sort_values(['review rate number', 'NAME'], ascending=[False, True])
        top_hotels = results.head(count)
        
        suggestions = []
        for _, hotel in top_hotels.iterrows():
            price_info = f" (${hotel['price']}/night)" if pd.notna(hotel['price']) else ""
            rating_info = f" - Rating: {hotel['review rate number']}" if pd.notna(hotel['review rate number']) else ""
            suggestions.append(f"• {hotel['NAME']}{price_info}{rating_info}")
        
        return f"🏨 **Hotels in {city}:**\n" + "\n".join(suggestions)

    @tool
    def GetRestaurantSuggestions(city: str, count: int = 5, cuisine_type: str = "any") -> str:
        """
        Get restaurant suggestions for a specific city.
        Parameters:
            city: The name of the city
            count: Number of suggestions to return (default: 5)
            cuisine_type: Type of cuisine - "any", "italian", "chinese", "mexican", etc.
        """
        results = restaurants_data[restaurants_data["City"] == city]
        if len(results) == 0:
            return f"No restaurants found in {city}."
        
        # Filter by cuisine if specified
        if cuisine_type != "any":
            results = results[results['Cuisines'].str.contains(cuisine_type, case=False, na=False)]
        
        # Sort by rating and name for consistent results
        results = results.sort_values(['Aggregate Rating', 'Name'], ascending=[False, True])
        top_restaurants = results.head(count)
        
        suggestions = []
        for _, restaurant in top_restaurants.iterrows():
            cost_info = f" (${restaurant['Average Cost']})" if pd.notna(restaurant['Average Cost']) else ""
            rating_info = f" - Rating: {restaurant['Aggregate Rating']}" if pd.notna(restaurant['Aggregate Rating']) else ""
            cuisine_info = f" - {restaurant['Cuisines']}" if pd.notna(restaurant['Cuisines']) else ""
            suggestions.append(f"• {restaurant['Name']}{cost_info}{rating_info}{cuisine_info}")
        
        return f"🍽️ **Dinner/Restaurant Suggestions in {city}:**\n" + "\n".join(suggestions)

    @tool
    def GetCitySuggestions(state: str) -> str:
        """
        Get city suggestions within a state.
        Parameters:
            state: The name of the state
        """
        if state not in city_data:
            return f"No cities found in {state}."
        
        cities = city_data[state]
        suggestions = []
        for city in cities[:10]:  # Limit to 10 cities
            suggestions.append(f"• {city}")
        
        return f"🏙️ **Cities in {state}:**\n" + "\n".join(suggestions)

    tools = [
        GetAttractionSuggestions,
        GetHotelSuggestions,
        GetRestaurantSuggestions,
        GetCitySuggestions,
    ]

    def suggestion_agent_prompt(query):
        instructions = f"""
        You are a focused travel suggestion assistant. Your job is to provide ONLY the specific type of recommendations the user asks for.

        **Available Tools:**
        1. **GetAttractionSuggestions[city, count]** - Get attraction recommendations
        2. **GetHotelSuggestions[city, count, budget_range]** - Get hotel recommendations
        3. **GetRestaurantSuggestions[city, count, cuisine_type]** - Get restaurant recommendations  
        4. **GetCitySuggestions[state]** - Get city recommendations within a state

        **Budget Ranges for Hotels:**
        - "budget": Under $100/night
        - "mid": $100-300/night
        - "luxury": $300+/night
        - "any": All price ranges

        **CRITICAL INSTRUCTIONS:**
        - **ONLY provide the specific type of suggestion the user asks for**
        - If user asks for "dinner suggestions" → ONLY use GetRestaurantSuggestions
        - If user asks for "hotels" → ONLY use GetHotelSuggestions
        - If user asks for "attractions" → ONLY use GetAttractionSuggestions
        - If user asks for "cities" → ONLY use GetCitySuggestions
        - **DO NOT provide multiple types of suggestions unless explicitly requested**
        - Extract the location (city/state) from the query
        - If no specific count is mentioned, default to 5 suggestions
        - Be focused and concise in your response

        **Query:** {query}
        """
        return instructions

    prompt = ChatPromptTemplate.from_messages([
        ("system", suggestion_agent_prompt(query)),
        ("placeholder", "{messages}"),
    ])

    # Provide focused response based on query analysis - skip the complex agent execution
    # This ensures consistency and prevents multiple iterations
    try:
        if suggestion_type == 'restaurant':
            final_result = GetRestaurantSuggestions(location, 5, "any")
        elif suggestion_type == 'hotel':
            final_result = GetHotelSuggestions(location, 5, "any")
        elif suggestion_type == 'attraction':
            final_result = GetAttractionSuggestions(location, 5)
        elif suggestion_type == 'city':
            final_result = GetCitySuggestions(location)
        else:
            final_result = "I couldn't find any suggestions for your request. Please try with a different location or be more specific about what you're looking for."
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        final_result = f"I apologize, but I encountered an error while getting {suggestion_type} suggestions for {location}. Please try again with a different location or be more specific about what you're looking for."
    
    return final_result
