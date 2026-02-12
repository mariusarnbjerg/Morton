import { useState, useEffect } from 'react';
import './App.css';

// Types matching your API
interface Question {
  text: string;
  questionId: string;
}

function App() {
  const [conversationId] = useState(`patient-${Date.now()}`);
  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState('');
  const [chatMessage, setChatMessage] = useState('');
  const [chatResponse, setChatResponse] = useState('');
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [isDone, setIsDone] = useState(false);
  const [loading, setLoading] = useState(false);

  // Start the conversation when component loads
  useEffect(() => {
    startConversation();
  }, []);

  // API calls
  const API_BASE = 'http://localhost:8000/api/v1';

  const startConversation = async () => {
    try {
      const response = await fetch(`${API_BASE}/conversations/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: conversationId })
      });
      const data = await response.json();

      setCurrentQuestion({ text: data.question, questionId: data.question_id });
      setMessages([{ role: 'bot', content: data.question }]);
      setIsDone(data.done);
    } catch (error) {
      console.error('Failed to start conversation:', error);
      alert('Could not connect to API. Make sure the server is running!');
    }
  };

  const submitAnswer = async () => {
    if (!answer.trim()) return;

    setLoading(true);

    // Add user answer to messages
    setMessages(prev => [...prev, { role: 'user', content: answer }]);

    try {
      const response = await fetch(`${API_BASE}/conversations/${conversationId}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answer })
      });
      const data = await response.json();

      if (data.done) {
        setIsDone(true);
        setMessages(prev => [...prev, { role: 'bot', content: 'Questionnaire complete! Thank you.' }]);
      } else {
        setCurrentQuestion({ text: data.question, questionId: data.question_id });
        setMessages(prev => [...prev, { role: 'bot', content: data.question }]);
      }

      setAnswer(''); // Clear input
    } catch (error) {
      console.error('Failed to submit answer:', error);
    } finally {
      setLoading(false);
    }
  };

  const askChatbot = async () => {
    if (!chatMessage.trim()) return;

    setLoading(true);
    setChatResponse('');

    // Add user question to messages
    setMessages(prev => [...prev, { role: 'user-chat', content: `❓ ${chatMessage}` }]);

    try {
      const response = await fetch(`${API_BASE}/conversations/${conversationId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: chatMessage })
      });
      const data = await response.json();

      setChatResponse(data.answer);
      setMessages(prev => [...prev, { role: 'bot-chat', content: `💬 ${data.answer}` }]);
      setChatMessage(''); // Clear input
    } catch (error) {
      console.error('Failed to chat:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        <h1>🏥 Pre-Anesthesia Questionnaire</h1>
        <p className="subtitle">Please answer the questions below</p>

        {/* Message History */}
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-content">{msg.content}</div>
            </div>
          ))}
        </div>

        {/* Main Input Area */}
        {!isDone && currentQuestion && (
          <div className="input-section">
            <h3>Current Question:</h3>
            <p className="current-question">{currentQuestion.text}</p>

            <input
              type="text"
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && submitAnswer()}
              placeholder="Type your answer here..."
              disabled={loading}
              className="input-field"
            />

            <button onClick={submitAnswer} disabled={loading} className="btn btn-primary">
              {loading ? 'Sending...' : 'Submit Answer'}
            </button>
          </div>
        )}

        {/* Chat Section */}
        {!isDone && (
          <div className="chat-section">
            <h3>Have a question?</h3>
            <p className="hint">Ask the chatbot anything about the questionnaire</p>

            <input
              type="text"
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && askChatbot()}
              placeholder="e.g., Why do you need this information?"
              disabled={loading}
              className="input-field"
            />

            <button onClick={askChatbot} disabled={loading} className="btn btn-secondary">
              {loading ? 'Asking...' : 'Ask Chatbot'}
            </button>
          </div>
        )}

        {/* Completion Message */}
        {isDone && (
          <div className="completion">
            <h2>✅ Thank you!</h2>
            <p>Your questionnaire has been submitted successfully.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;