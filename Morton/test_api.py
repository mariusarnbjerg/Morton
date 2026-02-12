"""
End-to-end test script for the AnæstesiCare API.
Tests the complete conversation flow.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"
CONVERSATION_ID = f"test-{int(time.time())}"  # Unique ID each run


def print_section(title):
    """Pretty print section headers"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_response(response):
    """Pretty print API response"""
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_1_health():
    """Test health check endpoint"""
    print_section("Test 1: Health Check")

    response = requests.get("http://localhost:8000/health")
    print_response(response)

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    print("✅ Health check passed!\n")


def test_2_start_conversation():
    """Start a new conversation"""
    print_section("Test 2: Start Conversation")

    response = requests.post(
        f"{BASE_URL}/conversations/start",
        json={"conversation_id": CONVERSATION_ID}
    )
    print_response(response)

    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == CONVERSATION_ID
    assert "question" in data
    assert data["done"] == False

    print(f"✅ Conversation started!")
    print(f"   First question: {data['question']}\n")

    return data


def test_3_answer_question(answer_text):
    """Answer the current question"""
    print_section(f"Test 3: Answer Question")
    print(f"Answer: '{answer_text}'\n")

    response = requests.post(
        f"{BASE_URL}/conversations/{CONVERSATION_ID}/answer",
        json={"answer": answer_text}
    )
    print_response(response)

    assert response.status_code == 200
    data = response.json()

    if not data["done"]:
        print(f"✅ Question answered!")
        print(f"   Next question: {data['question']}\n")
    else:
        print(f"✅ Questionnaire complete!\n")

    return data


def test_4_check_state():
    """Get current conversation state"""
    print_section("Test 4: Check Conversation State")

    response = requests.get(f"{BASE_URL}/conversations/{CONVERSATION_ID}/state")
    print_response(response)

    assert response.status_code == 200
    data = response.json()

    print(f"✅ State retrieved!")
    print(f"   Progress: {data['answered_count']}/{data['total_questions']}")
    print(f"   Current: {data['current_question']}\n")

    return data


def test_5_chat(question):
    """Ask the chatbot a question"""
    print_section(f"Test 5: Chat with Bot")
    print(f"Question: '{question}'\n")

    response = requests.post(
        f"{BASE_URL}/conversations/{CONVERSATION_ID}/chat",
        json={"message": question}
    )
    print_response(response)

    assert response.status_code == 200
    data = response.json()

    print(f"✅ Chatbot responded!")
    print(f"   Answer preview: {data['answer'][:100]}...\n")

    return data


def test_6_list_conversations():
    """List all active conversations"""
    print_section("Test 6: List All Conversations")

    response = requests.get(f"{BASE_URL}/conversations")
    print_response(response)

    assert response.status_code == 200
    data = response.json()

    print(f"✅ Found {data['count']} active conversation(s)\n")

    return data


def test_7_complete_questionnaire():
    """Answer all remaining questions to complete the questionnaire"""
    print_section("Test 7: Complete Questionnaire")

    # Keep answering until done
    answers = ["45", "Yes", "Penicillin", "No"]  # Sample answers
    answer_index = 0

    while True:
        # Check state
        state = requests.get(f"{BASE_URL}/conversations/{CONVERSATION_ID}/state").json()

        if state["done"]:
            print("✅ Questionnaire already complete!\n")
            break

        # Answer current question
        if answer_index < len(answers):
            answer = answers[answer_index]
        else:
            answer = "No"  # Default answer for any remaining questions

        print(f"Answering: '{state['current_question']}'")
        print(f"   → '{answer}'")

        response = requests.post(
            f"{BASE_URL}/conversations/{CONVERSATION_ID}/answer",
            json={"answer": answer}
        )

        data = response.json()
        answer_index += 1

        if data["done"]:
            print("\n✅ Questionnaire complete!\n")
            break


def test_8_get_summary():
    """Get the final summary"""
    print_section("Test 8: Get Final Summary")

    response = requests.get(f"{BASE_URL}/conversations/{CONVERSATION_ID}/summary")

    if response.status_code == 200:
        print_response(response)
        print("✅ Summary generated!\n")
        return response.json()
    else:
        print(f"⚠️  Summary not available (status: {response.status_code})")
        print("   This is OK if questionnaire isn't complete yet.\n")
        return None


def test_9_cleanup():
    """Delete the test conversation"""
    print_section("Test 9: Cleanup")

    response = requests.delete(f"{BASE_URL}/conversations/{CONVERSATION_ID}")
    print_response(response)

    assert response.status_code == 200

    print(f"✅ Test conversation deleted\n")


def run_all_tests():
    """Run the complete test suite"""
    print("\n" + "=" * 60)
    print("  AnæstesiCare API - End-to-End Test Suite")
    print("=" * 60)
    print(f"\nConversation ID: {CONVERSATION_ID}\n")

    try:
        # Run tests in sequence
        test_1_health()
        test_2_start_conversation()
        test_3_answer_question("John Doe")
        test_4_check_state()
        test_5_chat("Why do you need my name?")
        test_3_answer_question("45")  # Answer another question
        test_4_check_state()
        test_6_list_conversations()

        # Optional: Complete and get summary
        # test_7_complete_questionnaire()
        # test_8_get_summary()

        test_9_cleanup()

        # Success!
        print("\n" + "=" * 60)
        print("  ✅ ALL TESTS PASSED!")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print("\n" + "=" * 60)
        print(f"  ❌ TEST FAILED: Assertion error")
        print("=" * 60)
        print(f"\n{e}\n")
        raise

    except requests.exceptions.ConnectionError:
        print("\n" + "=" * 60)
        print("  ❌ ERROR: Cannot connect to API server!")
        print("=" * 60)
        print("\n  Make sure the server is running:")
        print("  uvicorn app.api.server:app --reload\n")
        raise

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  ❌ UNEXPECTED ERROR: {type(e).__name__}")
        print("=" * 60)
        print(f"\n{e}\n")
        raise


if __name__ == "__main__":
    # Check if server is running first
    try:
        requests.get("http://localhost:8000/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: API server is not running!")
        print("\nStart it with:")
        print("  uvicorn app.api.server:app --reload\n")
        exit(1)

    # Run the tests
    run_all_tests()