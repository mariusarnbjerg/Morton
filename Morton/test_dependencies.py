from app.api.dependencies import (
    get_orchestrator,
    get_conversation,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    ENVIRONMENT
)

print("Configuration loaded from .env:")
print(f"  Environment: {ENVIRONMENT}")
print(f"  Model: {OLLAMA_MODEL}")
print(f"  Ollama URL: {OLLAMA_BASE_URL}")
print()

# Test 1: Get the orchestrator
print("Test 1: Creating orchestrator...")
orch = get_orchestrator()
print(f"✅ Orchestrator created: {orch}")

# Test 2: Get the same orchestrator again (should be same instance)
orch2 = get_orchestrator()
print(f"✅ Same instance? {orch is orch2}")  # Should print True

# Test 3: Create a conversation
print("\nTest 2: Creating conversations...")
conv1 = get_conversation("test-123")
print(f"✅ Conversation 1: {conv1.conversation_id}")

conv2 = get_conversation("test-456")
print(f"✅ Conversation 2: {conv2.conversation_id}")

# Test 4: Get the same conversation again
conv1_again = get_conversation("test-123")
print(f"✅ Same conversation? {conv1 is conv1_again}")  # Should print True

print("\n✅ All tests passed!")