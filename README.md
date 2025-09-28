# TravelPlanner 0.9

A comprehensive AI-powered travel planning application that leverages a hierarchical multi-agent system to create personalized itineraries, check calendar availability, and manage travel bookings through intelligent coordination and adaptive learning.

## System Architecture

Our multi-agent system follows a hierarchical structure where a supervising chatbot agent controls and coordinates interactions between specialized agents while maintaining direct communication with users. The architecture is represented as a state graph consisting of nodes (agents) and edges (transition conditions), enabling dynamic routing based on user needs and system state.

**Key Architectural Features:**
- **Centralized Coordination**: A supervising chatbot agent orchestrates all agent interactions
- **Dynamic Routing**: Intelligent decision-making for agent selection based on context and user requirements
- **State Management**: Persistent state tracking across agent interactions
- **Adaptive Learning**: Novel learning mechanisms that improve recommendations over time
- **Global Scalability**: Handles any location worldwide without hard-coding

## Features

- 🤖 **Hierarchical Multi-Agent System**: Coordinated agents with intelligent routing and state management
- 🧠 **Adaptive Learning**: Novel learning mechanisms that improve recommendations over time
- 🌍 **Global Scalability**: Handles any location worldwide through hierarchical context learning
- 📅 **Google Calendar Integration**: Check availability and add events to your calendar
- 🔍 **Intelligent Data Retrieval**: Context-aware fetching of travel data including flights, restaurants, and attractions
- 📋 **Personalized Itinerary Generation**: Create customized travel itineraries based on user preferences and learning
- 💬 **Interactive Chat Interface**: Natural language interaction with intelligent conversation flow
- 🔄 **Real-time Streaming**: Live updates during the planning process with state persistence
- 🎯 **Context-Aware Recommendations**: Learning-enhanced suggestions that adapt to user behavior and preferences

## Agent Architecture

### Central Coordinator
- **Supervising Chatbot Agent**: Central orchestrator that coordinates all agent interactions, manages state transitions, and maintains conversation flow with users

### Specialized Agents
- **Calendar Agent**: Manages Google Calendar integration for availability checks and event creation
- **Data Retrieval Agent**: Fetches relevant travel data based on user preferences and learned context
- **Itinerary Agent**: Creates personalized travel itineraries using retrieved data and learned preferences
- **Query Checker**: Validates and processes user queries to ensure proper routing
- **Feedback Agent**: Handles user feedback and refinements with learning-enhanced recommendations
- **Suggestion Agent**: Provides context-aware recommendations for attractions, hotels, and restaurants

### Learning System
- **Context Learner**: Hierarchical learning system that adapts to user preferences across different locations and contexts
- **Similarity Learner**: Transfer learning between similar contexts for improved recommendations
- **Pattern Discovery**: Unsupervised discovery of user behavior patterns from interactions
- **Learning Agent**: Integrated ensemble learning that combines all learning approaches for optimal recommendations

## How the Hierarchical System Works

The system operates through a sophisticated state graph where:

1. **User Interaction**: Users communicate with the supervising chatbot agent through natural language
2. **Query Analysis**: The chatbot analyzes user input and determines the appropriate next step
3. **Agent Routing**: Based on context and requirements, the chatbot routes to specialized agents:
   - **Query Checker**: Validates and structures user queries
   - **Calendar Agent**: Checks availability and manages calendar events
   - **Data Retrieval Agent**: Fetches relevant travel data
   - **Itinerary Agent**: Creates personalized travel plans
   - **Feedback Agent**: Handles refinements and improvements
   - **Suggestion Agent**: Provides recommendations
4. **State Management**: Each agent updates the shared state and notifies the chatbot
5. **Learning Integration**: The learning system continuously improves recommendations based on user interactions
6. **Dynamic Routing**: The chatbot determines the next agent or requests additional user input

This hierarchical approach ensures efficient coordination, maintains conversation context, and enables adaptive learning that improves over time.

## Prerequisites

- Python 3.8+
- Google Cloud Console project with Calendar API enabled
- Google OAuth2 credentials

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd TravelPlanner_0.9
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_CLIENT_ID=your_google_client_id
   GOOGLE_CLIENT_SECRET=your_google_client_secret
   OPENAI_API_KEY=your_openai_api_key
   ```

5. **Set up Google Calendar credentials**
   - Download your OAuth2 credentials from Google Cloud Console
   - Place the credentials file in the `google_credentials/` directory
   - Ensure the file is named `credentials.json`

## Running the Application

### Using Uvicorn (Recommended)

```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

The application will be available at:
- **API**: http://localhost:5000
- **Interactive API Docs**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

### Alternative: Direct Python

```bash
python main.py
```

## API Endpoints

### Chat Endpoints
- `POST /chat` - Send a message to the travel planner
- `POST /chat_stream` - Stream responses from the travel planner

### Authentication
- `GET /auth/google?userId={userId}` - Initiate Google OAuth flow
- `GET /auth/callback` - Handle Google OAuth callback

### Profile Management
- `POST /save_profile` - Save user profile information

## Usage Examples

### Basic Travel Planning
```bash
curl -X POST "http://localhost:5000/chat" \
     -H "Content-Type: application/json" \
     -d '{"ipt": "I want to plan a trip from Boston to Newark for 3 days from March 5th to March 7th, 2022, with a budget of $1,700"}'
```

### Stream Chat
```bash
curl -X POST "http://localhost:5000/chat_stream" \
     -H "Content-Type: application/json" \
     -d '{"ipt": "Check my calendar for March 5-7, 2022"}'
```

## Project Structure

```
TravelPlanner_0.9/
├── agents/
│   ├── calendar.py          # Google Calendar integration
│   ├── data_retrieval.py    # Travel data fetching
│   ├── itinerary.py         # Itinerary generation
│   ├── query_checker.py     # Query validation
│   └── feedback.py          # User feedback handling
├── google_credentials/      # Google OAuth credentials
├── user_tokens/            # User authentication tokens
├── main.py                 # FastAPI application
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Configuration

### Google Calendar Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google Calendar API
4. Create OAuth2 credentials
5. Download the credentials JSON file
6. Place it in `google_credentials/credentials.json`

### Environment Variables
Make sure to set the following environment variables:
- `GOOGLE_CLIENT_ID`: Your Google OAuth2 client ID
- `GOOGLE_CLIENT_SECRET`: Your Google OAuth2 client secret
- `OPENAI_API_KEY`: Your OpenAI API key

## Development

### Running in Development Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload --log-level debug
```

### Testing
The application includes comprehensive error handling and logging. Check the console output for detailed information about agent interactions and any issues.

## Troubleshooting

### Common Issues

1. **Google Calendar Authentication Error**
   - Ensure credentials.json is in the correct location
   - Check that Google Calendar API is enabled
   - Verify OAuth2 credentials are correct

2. **OpenAI API Errors**
   - Verify OPENAI_API_KEY is set correctly
   - Check API key permissions and billing

3. **Port Already in Use**
   - Change the port: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
   - Or kill the process using the port

### Debug Mode
Enable debug logging by setting the log level:
```bash
uvicorn main:app --host 0.0.0.0 --port 5000 --reload --log-level debug
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Create an issue in the repository

---

**Note**: This is version 0.9 of TravelPlanner. The application is in active development and may have breaking changes in future versions.
