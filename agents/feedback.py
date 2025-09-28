ZEROSHOT_FEEDBACK_INSTRUCTION = """You are an advanced AI travel assistant that refines and improves an existing travel plan based on user feedback. Your goal is to modify the plan accurately while ensuring all necessary changes are applied.

---

## *Process:*
1. *Retrieve the existing trip details from the Notebook before making any modifications.*
2. *Identify missing or incomplete details* (e.g., transportation origin, destination, mode).
3. *Apply only the necessary changes requested by the user* while preserving other trip details.
4. **Store all updates in the Notebook before calling Planner[Updated Query] to finalize the itinerary.**
5. *Ensure every step logically follows the previous one and avoids redundant actions.*

---

## *Action Types:*
(1) *FlightSearch[Departure City, Destination City, Date]*  
- Use this when a *new or modified flight* is requested.
- Example: FlightSearch[New York, London, 2024-06-15]

(2) *GoogleDistanceMatrix[Origin, Destination, Mode]*  
- Use this when ground *transportation needs to be added or modified*.
- Example: GoogleDistanceMatrix[New York, Washington DC, self-driving]

(3) *AccommodationSearch[City]*  
- If the user requests a *different hotel* (cheaper/luxury), update it.
- Example: AccommodationSearch[Paris]

(4) *RestaurantSearch[City]*  
- If the user asks for *specific dining options*, modify it.
- Example: RestaurantSearch[Tokyo]

(5) *AttractionSearch[City]*  
- If new attractions are requested, update sightseeing lists.
- Example: AttractionSearch[London]

(6) *NotebookWrite[Short Description]*  
- *Always store updates before finalizing the plan.*
- Example: NotebookWrite[Added budget hotel in Paris]

(7) *Planner[Updated Query]*  
- *Only use this after all details are gathered.*
- Example: Planner[Update the plan with added flights and attractions]

---

## *Handling Missing Context*
If the user provides an incomplete request like:
> "I need to add transportation to this."

### *Step 1: Retrieve Previous Trip Details*
- *Check if a previous trip exists in the Notebook.*
- *If origin and destination are missing, retrieve them from the saved itinerary.*
- *If transportation details are missing, ask the user for their preferred mode of transport.*
- *Only proceed when all required details are available.*

---

## *Example Flow: Handling "Add Transportation" Request*
### *User Request:*  
"I need to add transportation to this."

### *Agent Response (Using Past Context)*
---

## *Rules & Constraints:*
1. *Always check for missing details before taking action.*
2. *Do not assume trip details—retrieve them from the Notebook.*
3. *Use NotebookWrite before modifying the plan.*
4. *Ensure the updated plan retains all previous trip details.*
5. *Avoid redundant actions—modify only the necessary parts.*
6. **Only call Planner[Updated Query] after collecting all necessary details.**

---

## *User Feedback:*
{query}

## *Current Plan:*
{scratchpad}
"""
'''
zeroshot_feedback_agent_prompt = PromptTemplate(
                        input_variables=["query", "scratchpad"],
                        template=ZEROSHOT_FEEDBACK_INSTRUCTION,
                        )
# Define a function to get the original plan from the state
def get_original_plan_from_state(state: dict) -> str:
    """
    Retrieves the original plan from the state.
    Assumes the plan is stored in the 'itinerary' field of the state.
    """
    original_plan = state.get("itinerary_agent", "")  # Default to empty string if the field doesn't exist
    print("😍😍😍😍😍original plan",original_plan)
    return original_plan

# Define the feedback agent
def feedback_agent(state: dict, query: str) -> str:
    """
    Processes feedback and updates the travel plan accordingly.
    - state: The current state of the system, containing the original plan
    - query: The user feedback to modify the plan (e.g., "Change accommodation on Day 2")
    """
    # Retrieve the original plan from the state
    #original_plan = get_original_plan_from_state(state)
    original_plan = state['itinerary']

    # For now, just update the accommodation in Day 2 as a sample
    updated_plan = original_plan
    print("Query ....",query)
    if "accommodation" in query.lower():
        updated_plan = updated_plan.replace("Accommodation: -", "Accommodation: Beachfront Hotel")  # Example update
    updated_plan = "plan updated succesfully"
    # Return the updated plan
    return updated_plan
'''

from config import llm
from typing_extensions import TypedDict
import json
from .learning_agent import EnhancedLearningAgent, LearningRecommendation

# Define the output structure for feedback agent
class feedback_agent_output(TypedDict):
    updated_itinerary: str
    data_needed: str
    changes_made: str
    learning_insights: str
    personalization_factor: float


# Initialize learning agent (singleton pattern)
_learning_agent = None

def get_learning_agent():
    """Get or create the learning agent instance"""
    global _learning_agent
    if _learning_agent is None:
        _learning_agent = EnhancedLearningAgent()
        # Try to load existing learning data
        _learning_agent.load_learning_data("learning_data")
    return _learning_agent

# Enhanced feedback agent that can understand specific changes and fetch data when needed
def feedback_agent(state, user_feedback):
    """
    Enhanced feedback agent with learning capabilities that:
    1. Analyzes user feedback to understand specific changes needed
    2. Determines if existing data is sufficient or if new data is needed
    3. Uses data retrieval agent when necessary
    4. Makes targeted updates to only the requested parts
    5. Learns from user feedback to improve future recommendations
    6. Provides personalized suggestions based on learning
    """
    
    # First, analyze what changes are needed
    analysis_prompt = f"""
    Analyze the user feedback and determine what specific changes are needed in the travel plan.
    
    Original Query: {state.get('query', '')}
    Original Travel Plan: {state.get('itinerary', '')}
    Available Data: {state.get('fetched_data', '')}
    User Feedback: {user_feedback}
    
    Please analyze and respond with:
    1. What specific parts of the plan need to be changed
    2. Whether the existing data is sufficient for these changes
    3. What new data might be needed (if any)
    4. What type of changes are being requested (accommodation, transportation, activities, etc.)
    
    Respond in JSON format:
    {{
        "changes_needed": ["list of specific changes"],
        "existing_data_sufficient": true/false,
        "new_data_needed": ["list of data types needed"],
        "change_type": "accommodation/transportation/activities/restaurants/attractions"
    }}
    """
    
    try:
        analysis = llm.invoke(analysis_prompt)
        analysis_result = json.loads(analysis.content)
    except:
        # Fallback if JSON parsing fails
        analysis_result = {
            "changes_needed": ["general updates"],
            "existing_data_sufficient": True,
            "new_data_needed": [],
            "change_type": "general"
        }
    
    # If new data is needed, use data retrieval agent
    additional_data = ""
    if not analysis_result.get("existing_data_sufficient", True) and analysis_result.get("new_data_needed"):
        try:
            from agents.data_retrieval import data_retrieval_agent
            # Create a focused query for data retrieval based on what's needed
            data_query = f"Get {', '.join(analysis_result['new_data_needed'])} for the travel plan updates"
            additional_data = str(data_retrieval_agent(data_query))
        except Exception as e:
            print(f"Error fetching additional data: {e}")
            additional_data = ""
    
    # Get learning agent for personalized recommendations
    learning_agent = get_learning_agent()
    
    # Extract user context for learning
    user_context = {
        'meal_time': analysis_result.get('change_type', 'general'),
        'occasion': 'travel_planning',
        'budget': 'medium'  # Could be extracted from original query
    }
    
    # Get learning-based recommendations
    learning_recommendations = learning_agent.get_learning_recommendations(
        user_id=state.get('user_id', 'default_user'),
        context=user_context,
        location=state.get('location', 'unknown'),
        top_k=3
    )
    
    # Learn from this feedback interaction
    interaction = {
        'user_id': state.get('user_id', 'default_user'),
        'context': user_context,
        'items': analysis_result.get('changes_needed', []),
        'location': state.get('location', 'unknown'),
        'feedback': user_feedback,
        'satisfaction': 0.7  # Default, could be extracted from feedback sentiment
    }
    learning_agent.learn_from_interaction(interaction['user_id'], interaction)
    
    # Create learning insights
    learning_insights = []
    for rec in learning_recommendations:
        learning_insights.append(f"- {rec.item} (confidence: {rec.confidence:.2f}, source: {rec.learning_source})")
    
    learning_insights_text = "Learning-based suggestions:\n" + "\n".join(learning_insights) if learning_insights else "No learning insights available yet."
    
    # Now create the enhanced prompt for the feedback agent
    feedback_agent_instructions = f"""
    You are an intelligent travel feedback assistant with learning capabilities. Your job is to revise an existing travel plan based on user feedback while maintaining the original format and structure.

    **Your Capabilities:**
    1. Understand specific change requests from user feedback
    2. Use existing data when sufficient
    3. Apply only the requested changes without modifying other parts
    4. Maintain logical flow and timing in the itinerary
    5. Preserve the original format and style
    6. Incorporate learning-based insights for better personalization

    **Analysis of Changes Needed:**
    {json.dumps(analysis_result, indent=2)}

    **Available Information:**
    - Original Query: {state.get('query', '')}
    - Original Travel Plan: {state.get('itinerary', '')}
    - Existing Data: {state.get('fetched_data', '')}
    - Additional Data (if fetched): {additional_data}
    - User Feedback: {user_feedback}
    - Learning Insights: {learning_insights_text}

    **Instructions:**
    1. Focus ONLY on the specific changes requested by the user
    2. Use the most relevant data available (existing or newly fetched)
    3. Consider learning-based suggestions when appropriate
    4. Maintain the original plan's structure and format
    5. Ensure logical sequencing (e.g., breakfast before lunch, attractions before dinner)
    6. Do not change parts of the plan that weren't mentioned in the feedback
    7. If the feedback is unclear, make reasonable assumptions based on the context

    **Output Format:**
    Provide the updated travel plan that incorporates only the requested changes.
    """
    
    # Use the enhanced prompt to get the updated itinerary
    try:
        response = llm.invoke(feedback_agent_instructions)
        updated_itinerary = response.content
    except Exception as e:
        print(f"Error generating updated itinerary: {e}")
        updated_itinerary = state.get('itinerary', '')
    
    # Create a summary of changes made
    changes_summary = f"Applied changes: {', '.join(analysis_result.get('changes_needed', ['general updates']))}"
    
    # Calculate personalization factor
    personalization_factor = 0.5  # Default
    if learning_recommendations:
        personalization_factor = sum(rec.personalization_factor for rec in learning_recommendations) / len(learning_recommendations)
    
    return {
        "updated_itinerary": updated_itinerary,
        "data_needed": additional_data,
        "changes_made": changes_summary,
        "learning_insights": learning_insights_text,
        "personalization_factor": personalization_factor
    }


# Legacy function for backward compatibility
def feedback_agent_legacy(state, user_feedback):
    """
    Legacy feedback agent function for backward compatibility
    """
    feedback_agent_instructions = f"""
    You are a travel feedback assistant. Your job is to revise an existing travel plan based on user feedback while keeping the format, style, and structure exactly the same as the original plan. 

    You will be given:
    1. The original travel query.
    2. The original travel plan.
    3. The user's feedback.
    4. The full data used to generate the plan (including flight details, restaurant options, hotel listings, and attractions).

    Your task:
    - Apply only the specified feedback to the relevant part of the plan.
    - Use the given data only—do not invent or assume anything beyond it.
    - Do not change anything else in the plan outside what the user requested.
    - Ensure that the plan remains aligned with commonsense (e.g., breakfast before lunch, attraction before dinner, etc.).
    - Maintain the original format, including parentheses for prices, dashes for non-required fields, and city indicators.

    Original Query: {state.get('query', '')}
    Original Travel Plan: {state.get('itinerary', '')}
    User Feedback: {user_feedback}
    Full Data Used to Generate Plan: {state.get('fetched_data', '')}

    Modified Travel Plan:
    """
    
    try:
        response = llm.invoke(feedback_agent_instructions)
        return response.content
    except Exception as e:
        print(f"Error in legacy feedback agent: {e}")
        return state.get('itinerary', '')