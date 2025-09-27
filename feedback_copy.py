from langchain_core.prompts import PromptTemplate

ZEROSHOT_FEEDBACK_INSTRUCTION = """You are an advanced AI travel assistant that refines and improves an existing travel plan based on user feedback. Your goal is to modify the plan accurately while ensuring all necessary changes are applied.

---

## **Process:**
1. **Retrieve the existing trip details from the Notebook before making any modifications.**
2. **Identify missing or incomplete details** (e.g., transportation origin, destination, mode).
3. **Apply only the necessary changes requested by the user** while preserving other trip details.
4. **Store all updates in the Notebook before calling `Planner[Updated Query]` to finalize the itinerary.**
5. **Ensure every step logically follows the previous one and avoids redundant actions.**

---

## **Action Types:**
(1) **FlightSearch[Departure City, Destination City, Date]**  
- Use this when a **new or modified flight** is requested.
- Example: `FlightSearch[New York, London, 2024-06-15]`

(2) **GoogleDistanceMatrix[Origin, Destination, Mode]**  
- Use this when ground **transportation needs to be added or modified**.
- Example: `GoogleDistanceMatrix[New York, Washington DC, self-driving]`

(3) **AccommodationSearch[City]**  
- If the user requests a **different hotel** (cheaper/luxury), update it.
- Example: `AccommodationSearch[Paris]`

(4) **RestaurantSearch[City]**  
- If the user asks for **specific dining options**, modify it.
- Example: `RestaurantSearch[Tokyo]`

(5) **AttractionSearch[City]**  
- If new attractions are requested, update sightseeing lists.
- Example: `AttractionSearch[London]`

(6) **NotebookWrite[Short Description]**  
- **Always store updates before finalizing the plan.**
- Example: `NotebookWrite[Added budget hotel in Paris]`

(7) **Planner[Updated Query]**  
- **Only use this after all details are gathered.**
- Example: `Planner[Update the plan with added flights and attractions]`

---

## **Handling Missing Context**
If the user provides an incomplete request like:
> "_I need to add transportation to this._"

### **Step 1: Retrieve Previous Trip Details**
- **Check if a previous trip exists in the Notebook.**
- **If origin and destination are missing, retrieve them from the saved itinerary.**
- **If transportation details are missing, ask the user for their preferred mode of transport.**
- **Only proceed when all required details are available.**

---

## **Example Flow: Handling "Add Transportation" Request**
### **User Request:**  
_"I need to add transportation to this."_

### **Agent Response (Using Past Context)**
---

## **Rules & Constraints:**
1. **Always check for missing details before taking action.**
2. **Do not assume trip details—retrieve them from the Notebook.**
3. **Use NotebookWrite before modifying the plan.**
4. **Ensure the updated plan retains all previous trip details.**
5. **Avoid redundant actions—modify only the necessary parts.**
6. **Only call `Planner[Updated Query]` after collecting all necessary details.**

---

## **User Feedback:**
{query}

## **Current Plan:**
{scratchpad}
"""

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
    original_plan = get_original_plan_from_state(state)

    # For now, just update the accommodation in Day 2 as a sample
    updated_plan = original_plan
    print("Query ....",query)
    if "accommodation" in query.lower():
        updated_plan = updated_plan.replace("Accommodation: -", "Accommodation: Beachfront Hotel")  # Example update

    # Return the updated plan
    return updated_plan