from app.api.models import StartConversationRequest, StartConversationResponse

# Create a request
request = StartConversationRequest(conversation_id="test-123")
print(f"Request: {request}")
print(f"Conversation ID: {request.conversation_id}")

# Create a response
response = StartConversationResponse(
    conversation_id="test-123",
    question="What is your name?",
    question_id="q1",
    done=False
)
print(f"\nResponse: {response}")
print(f"Question: {response.question}")

# Pydantic can convert to dict or JSON
print(f"\nAs dict: {response.model_dump()}")
print(f"As JSON: {response.model_dump_json()}")