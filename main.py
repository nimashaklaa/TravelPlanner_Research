# main.py
import json
import os

from google_auth_oauthlib.flow import Flow
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Literal
from langgraph.types import Command

from agents.feedback import feedback_agent
from agents.suggestion import suggestion_agent
from agents.enhanced_itinerary import enhanced_itinerary_agent
from agents.data_retrieval import data_retrieval_agent
from agents.calendar import calendar_agent
from agents.query_checker import query_checker_module
from config import llm  # Import the shared llm from config.py
from pydantic import BaseModel
from fastapi.responses import StreamingResponse,Response
from typing import AsyncIterator
from fastapi.responses import RedirectResponse

import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from fastapi import Request

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)
class State(TypedDict):
    next: str
    message_list : list
    fetched_data : str
    query : str
    itinerary: str
    planning_enhanced: bool  # Track if enhanced planning was used
    planning_score: float    # Store planning quality score

# # Creating the Agent Nodes

def data_retrieval_node(state: State) -> Command[Literal['chatbot']]:
    # print(state["query"])
    response = data_retrieval_agent(state['query'])
    print("Retrieved data:" + str(response))

    if not response:
        result = "couldn't retrieve data which are needed for itinerary agent"
    else:
        result = "data retrieved which are needed for itinerary agent"
        # result = data_retrieval_agent(state["query"])

    new_lst = state["message_list"] + [("ai", "data_retrieval_agent : " + result)]

    return Command(goto='chatbot', update={"next": "chatbot", "message_list": new_lst, "fetched_data": str(response)})


def itinerary_node(state: State) -> Command[Literal['chatbot']]:
    # Use enhanced itinerary agent with planner module
    response = enhanced_itinerary_agent(state["query"], state["fetched_data"], use_planner=True)
    
    # Try to get planning summary for metrics
    try:
        from agents.enhanced_itinerary import get_planning_summary
        summary = get_planning_summary(state["query"], state["fetched_data"])
        planning_score = summary.get('final_score', 0.0) if 'final_score' in summary else 0.0
        planning_enhanced = True
    except:
        planning_score = 0.0
        planning_enhanced = False

    # new_lst = state["message_list"].append(response.content)
    new_lst = state["message_list"] + [("ai", f"enhanced_itinerary_agent (score: {planning_score:.1f}) : " + response)]

    return Command(goto='chatbot', update={
        "next": "chatbot", 
        "message_list": new_lst, 
        "itinerary": str(response),
        "planning_enhanced": planning_enhanced,
        "planning_score": planning_score
    })


def query_checker_node(state: State) -> Command[Literal['chatbot']]:
    result = query_checker_module(state["message_list"])

    new_lst = state["message_list"] + [
        ("ai", "query_checker_module : " + "query built successfully, proceed to next steps")]

    return Command(goto='chatbot', update={"next": "chatbot", "message_list": new_lst, "query": result})


def calendar_node(state: State) -> Command[Literal['chatbot']]:
    # print("QUERY GOING TO CALENDAR AGENT",state["message_list"][-1])
    result = calendar_agent(str(state["message_list"][-1]))

    new_lst = state["message_list"] + [("ai", "calendar_agent : " + result)]

    return Command(goto='chatbot', update={"next": "chatbot", "message_list": new_lst})


def feedback_node(state: State) -> Command[Literal['chatbot']]:
    """
    Enhanced feedback node that can understand specific changes and fetch data when needed
    """
    user_feedback = str(state["message_list"][-1])
    
    # Call the enhanced feedback agent
    feedback_result = feedback_agent(state, user_feedback)
    
    # Handle both new structured response and legacy response
    if isinstance(feedback_result, dict):
        updated_itinerary = feedback_result.get("updated_itinerary", "")
        data_needed = feedback_result.get("data_needed", "")
        changes_made = feedback_result.get("changes_made", "")
        
        # Update the itinerary in the state
        new_lst = state["message_list"] + [("ai", f"feedback_agent : {changes_made} - {updated_itinerary}")]
        
        return Command(goto='chatbot', update={
            "next": "chatbot", 
            "message_list": new_lst,
            "itinerary": updated_itinerary,
            "fetched_data": state.get("fetched_data", "") + (f"\nAdditional data: {data_needed}" if data_needed else "")
        })
    else:
        # Legacy response handling
        new_lst = state["message_list"] + [("ai", "feedback_agent : " + str(feedback_result))]
        return Command(goto='chatbot', update={"next": "chatbot", "message_list": new_lst, "itinerary": str(feedback_result)})


def suggestion_node(state: State) -> Command[Literal['chatbot']]:
    """
    Suggestion node that provides recommendations for attractions, hotels, restaurants, etc.
    without creating full travel plans.
    """
    user_query = str(state["message_list"][-1])
    
    # Call the suggestion agent
    suggestions = suggestion_agent(user_query)
    
    new_lst = state["message_list"] + [("ai", "suggestion_agent : " + suggestions)]
    
    return Command(goto='chatbot', update={"next": "chatbot", "message_list": new_lst})

# Human Input

def human_interrupt(state: State) -> Command[Literal['chatbot']]:
    return


# Chatbot

chatbot_prompt = """
You are a dedicated travel planning chatbot designed to create personalized and well-organized trips.

Your Responsibilities:

1. Greeting & Introduction: Start by greeting the user and explaining that you're here to help plan their trip. Inform them that currently you can assist them with:
   - Calendar checks
   - Add events to user's Google calendar
   - Data retrieval for travel details (e.g., restaurants, flights, attractions)
   - Itinerary planning
   - Suggestions for attractions, hotels, restaurants in specific areas

2. Query Construction : Before moving forward, gather essential details for building a travel query. You will ask the user for:
   - Departure location(required)
   - Destination(required)
   - Budget(optional)
   - Preferences (optional: activities, accommodations, food, etc.)

3. Information Collection:
   Inquire about specific details that are required for each agent as follows:
   - **calendar_agent**: User’s travel dates and any potential conflicts with Google Calendar events.
   - **data_retrieval_agent**: Budget and activity preferences to gather relevant data for the trip (e.g., flights, restaurants, local attractions).
   - **itinerary_agent**: Once data is gathered, this agent will create a personalized itinerary for the user.

4. Agent Routing: Based on the collected information, determine which agent to route the user to:
   - **calendar_agent**: Check for any calendar conflicts with the user's travel dates and add calendar events to the google calendar.
   - **data_retrieval_agent**: Fetch the relevant travel data after the user provides their preferences and budget.
   - **itinerary_agent**: Generate the itinerary after gathering travel data.
   - **feedback_agent**: Handle user feedback and make specific changes to existing travel plans. This agent can understand what changes are needed and fetch additional data if required.
   - **suggestion_agent**: Provide recommendations for attractions, hotels, restaurants, etc. in specific areas without creating full travel plans.
   - **human_interrupt**: Allow the user to interact directly and make any changes to their itinerary or provide additional information.

5. Response Handling:
   - **Structured Output**: Ensure all responses are in JSON format with the following keys:
     - `next`: The next agent to route to (`calendar_agent`, `data_retrieval_agent`, `itinerary_agent`, `feedback_agent`, `suggestion_agent`, `human_interrupt`, or `FINISH`).
     - `messages`: The message content to send to the user.

6. Information Validation: If any required information is missing or incomplete, gather more information from the user.

7. Finalization: Once the itinerary is complete, ask the user if they would like to add anything else to the plan or finalize the itinerary.

Communication Style:
- Use a friendly, clear, and informative tone.
- Ensure all questions are concise and easy to understand.
- Make the user feel supported and assured throughout the planning process.
- When you are done with the itinerary, always route to the `human_interrupt` agent for review or additional input.

Example Workflow:
1. **Chatbot**: "Hello! I'm here to help you plan your perfect trip. Could you please share your preferred destinations and travel dates?"
2. **User**: "I'm traveling to New Jersey from March 10 to March 17."
3. **Chatbot**: "Noted! Now, could you share your budget and preferences for activities, restaurants, or accommodations?"
4. **User**: [Provides details]
5. **Chatbot**: "Let me check your calendar for any conflicts with these dates..."
6. **calendar_agent**: [Checks user’s Google Calendar for conflicts]
7. **Chatbot**: "I see that you have a meeting on March 12 at 3 PM. Would you like to adjust your travel dates or keep this event?"
8. **User**: "I'll keep it, but please add a reminder."
9. **Chatbot**: [Routes to `data_retrieval_agent` to fetch relevant data]
10. **data_retrieval_agent**: [Fetches data: restaurants, flight information, and local attractions in New Jersey]
11. **Chatbot**: "I’ve gathered all the details! Now I’m creating your personalized itinerary. Just a moment."
12. **itinerary_agent**: [Creates itinerary]
13. **Chatbot**: "Here's your personalized itinerary! Would you like to add anything else?"
14. **User**: "No, that's all. Thank you!"
15. **Chatbot**: "Would you like me to add this event to your Google Calendar?"
16. **User**: "Yes, please."
17. **Chatbot**: "Let me add this event your calendar.Event details : 2nd March - 5th March Trip to Miami from NewYork."
18. **calendar_agent**: [Adds events to Google Calendar]
19. **Chatbot**: "FINISH"
"""


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal['itinerary_agent', 'human_interrupt', 'calendar_agent', 'data_retrieval_agent', 'feedback_agent', 'suggestion_agent', 'FINISH']
    messages: str


def chatbot_node(state: State) -> Command[Literal[
    'human_interrupt', 'query_checker_module', 'calendar_agent', 'itinerary_agent', 'data_retrieval_agent', 'feedback_agent', 'suggestion_agent', '__end__']]:
    messages = [
                   {"role": "system", "content": chatbot_prompt}
               ] + state["message_list"]

    response = llm.with_structured_output(Router).invoke(messages)

    new_lst = state["message_list"] + [("ai", "chatbot : " + response["messages"])]

    goto = response["next"]

    if goto == "FINISH":
        goto = END

    if goto == "data_retrieval_agent" and state["query"] == "":
        goto = "query_checker_module"


    return Command(goto=goto, update={"next": goto, "message_list": new_lst})

# Initialize the state graph

builder = StateGraph(State)
builder.add_edge(START, "chatbot")
builder.add_node("chatbot", chatbot_node)
builder.add_node("itinerary_agent", itinerary_node)
builder.add_node("data_retrieval_agent", data_retrieval_node)
builder.add_node("calendar_agent", calendar_node)
builder.add_node("feedback_agent", feedback_node)
builder.add_node("suggestion_agent", suggestion_node)
builder.add_node("human_interrupt", human_interrupt)
builder.add_node("query_checker_module",query_checker_node)

# Compile the graph
graph = builder.compile()

class ChatInput(BaseModel):
    ipt: str

@app.post('/chat')
def chat(input_data: ChatInput):
    ipt = input_data.ipt

    if not os.path.exists("graph_state.json"):
        with open("graph_state.json", "w") as json_file:
            initial_state = {
                "message_list": [("user", "Hi")],
                "fetched_data": "",
                "query": ""
            }
            json.dump(initial_state, json_file, indent=4)

    with open("graph_state.json", "r") as json_file:
        current_state = json.load(json_file)

    current_state["next"] = 'chatbot'
    current_state["message_list"].append(['user', ipt])

    initial_state = current_state


    for s in graph.stream(initial_state, subgraphs=True, interrupt_before=["human_interrupt"], stream_mode="values"):
        print(s[1])
        if "message_list" in s[1] and s[1]['message_list'][-1][0] == "ai":
            current_state["message_list"].append(["ai", s[1]['message_list'][-1][1]])
            current_state["query"] = s[1]["query"]
            current_state["fetched_data"] = s[1]["fetched_data"]
            current_state["itinerary"] = s[1]["itinerary"]
            with open("graph_state.json", "w") as json_file:
                json.dump(current_state, json_file, indent=4)

            print("message: " + s[1]['message_list'][-1][1])
            print("next_node: " + s[1]["next"])
            print("query:" + s[1]['query'])
            print("fetched_data:" + s[1]["fetched_data"])
            print("itinerary:" + s[1]["itinerary"])
            print("----")
            print("\n")
            print(s[1]["next"] + "\n")
    return current_state["message_list"]


@app.post("/chat_stream")
async def chat_stream(human_input: ChatInput):
    # Check if the file exists
    # return {"reply": f"You said: {human_input.message}"}
    if os.path.exists("graph_state.json"):
        with open("graph_state.json", "r") as json_file:
            current_state = json.load(json_file)
    else:
        # If the file doesn't exist, create a new one with default values
        current_state = {
            "message_list": [("user", "Hi")],
            "query": "",
            "fetched_data": "",
            "itinerary": "",
        }
        with open("graph_state.json", "w") as json_file:
            json.dump(current_state, json_file)

    current_state["next"] = 'chatbot'
    current_state["message_list"].append(['user', human_input.ipt])
    initial_state = current_state

    # Start the graph stream

    async def message_stream() -> AsyncIterator[str]:
        for s in graph.stream(initial_state, subgraphs=True, interrupt_before=["human_interrupt"],
                              stream_mode="values"):
            if "message_list" in s[1] and s[1]['message_list'][-1][0] == "ai":
                current_state["message_list"].append(["ai", s[1]['message_list'][-1][1]])
                current_state["query"] = s[1]["query"]
                current_state["fetched_data"] = s[1]["fetched_data"]
                current_state["itinerary"] = s[1].get("itinerary")

                with open("graph_state.json", "w") as json_file:
                    json.dump(current_state, json_file, indent=4)

                print("message: " + s[1]['message_list'][-1][1])
                print("next_node: " + s[1].get("next", ""))
                print("query: " + current_state["query"])
                print("fetched_data: " + current_state["fetched_data"])
                print(f"itinerary: {current_state.get('itinerary', 'None')}")
                print("----\n")

                yield s[1]['message_list'][-1][1] + "\n"

    return StreamingResponse(message_stream(), media_type="text/plain")

class UserProfile(BaseModel):
    userId: str

@app.post("/save_profile")

async def save_profile(profile: UserProfile):
    user_id = profile.userId

    with open("user_profiles.txt", "a") as file:
        file.write(f"{user_id}\n")

    return {"status": "success", "message": f"User ID {user_id} saved."}


GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
REDIRECT_URI = "http://localhost:5000/auth/callback"
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# 📍 TEMP STORAGE (can be replaced with DB)
TOKEN_DIR = "user_tokens"
os.makedirs(TOKEN_DIR, exist_ok=True)

# === ROUTES ===

@app.get("/auth/google")
def auth_google(userId: str):
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
        state=userId
    )
    return RedirectResponse(auth_url)


@app.get("/auth/callback")
def auth_callback(request: Request):
    full_url = str(request.url)
    state = request.query_params.get("state")

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI]
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(authorization_response=full_url)

    creds = flow.credentials

    # Save credentials per user
    with open(f"{TOKEN_DIR}/{state}.json", "w") as f:
        json.dump({
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes
        }, f)

    # return JSONResponse({"message": f"Google Calendar connected for user {state}"})
    return RedirectResponse(url=f"http://localhost:5173/?connected=true")