import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface Question {
  question: string;
  context?: string;
}

function App() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState('');
  const [transcript, setTranscript] = useState<Message[]>([]);
  const [chatMessages, setChatMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [questionnaireComplete, setQuestionnaireComplete] = useState(false);
  const [showChat, setShowChat] = useState(false);
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  
  // Suggested questions for the chatbot
  const suggestedQuestions = [
    "What should I do to prepare for my surgery?",
    "What risks are associated with full anesthesia?",
    "How long does it take to wake up after anesthesia?",
    "What side effects can I expect?",
    "Will I feel pain during the surgery?",
    "Can I eat or drink before my surgery?"
  ];

  useEffect(() => {
    startConversation();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const startConversation = async () => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('🚀 Starting conversation...');
      console.log('API Base:', API_BASE);
      console.log('Full URL:', `${API_BASE}/api/v1/conversations/start`);
      
      // CRITICAL FIX: The /start endpoint likely expects NO BODY or an empty object
      // Try with no body first
      const response = await fetch(`${API_BASE}/api/v1/conversations/start`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        // Empty body or no body - depending on what backend expects
        body: JSON.stringify({})
      });

      console.log('Response status:', response.status);
      console.log('Response ok:', response.ok);

      if (!response.ok) {
        // Get error details
        const errorText = await response.text();
        console.error('❌ Error response:', errorText);
        
        try {
          const errorJson = JSON.parse(errorText);
          console.error('❌ Error details:', errorJson);
          
          // Show validation errors if present
          if (errorJson.detail && Array.isArray(errorJson.detail)) {
            const validationErrors = errorJson.detail
              .map((err: any) => `${err.loc?.join('.')}: ${err.msg}`)
              .join('; ');
            throw new Error(`Validation error: ${validationErrors}`);
          }
        } catch (parseError) {
          // If not JSON, show raw error
          console.error('❌ Could not parse error as JSON');
        }
        
        throw new Error(`Failed to start conversation: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('✅ Success! Response data:', data);
      
      setConversationId(data.conversation_id);
      setCurrentQuestion({
        question: data.question,
        context: data.context,
      });
      setTranscript(data.transcript || []);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to start conversation';
      setError(errorMessage);
      console.error('❌ Error starting conversation:', err);
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async () => {
    if (!conversationId || !answer.trim()) return;

    try {
      setLoading(true);
      setError(null);
      
      console.log('📝 Submitting answer...');
      console.log('Conversation ID:', conversationId);
      console.log('Answer:', answer.trim());
      
      const requestBody = { answer: answer.trim() };
      console.log('Request body:', requestBody);
      
      const response = await fetch(
        `${API_BASE}/api/v1/conversations/${conversationId}/answer`,
        {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(requestBody),
        }
      );

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Error response:', errorText);
        throw new Error(`Failed to submit answer: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('✅ Answer submitted successfully:', data);
      
      if (data.next_question) {
        setCurrentQuestion({
          question: data.next_question,
          context: data.context,
        });
        setTranscript(data.transcript || []);
        setAnswer('');
      } else {
        // Questionnaire is complete
        console.log('🎉 Questionnaire complete!');
        setQuestionnaireComplete(true);
        setTranscript(data.transcript || []);
        setShowChat(true);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to submit answer';
      setError(errorMessage);
      console.error('❌ Error submitting answer:', err);
    } finally {
      setLoading(false);
    }
  };

  const sendChatMessage = async (message?: string) => {
    const messageToSend = message || chatInput.trim();
    if (!conversationId || !messageToSend) return;

    const userMessage: Message = { role: 'user', content: messageToSend };
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');

    try {
      setLoading(true);
      setError(null);

      console.log('💬 Sending chat message...');
      console.log('Conversation ID:', conversationId);
      console.log('Message:', messageToSend);

      const requestBody = { message: messageToSend };
      console.log('Request body:', requestBody);

      const response = await fetch(
        `${API_BASE}/api/v1/conversations/${conversationId}/chat`,
        {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(requestBody),
        }
      );

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Error response:', errorText);
        throw new Error(`Failed to send chat message: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('✅ Chat response received:', data);
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
      };
      setChatMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
      setError(errorMessage);
      console.error('❌ Error sending chat message:', err);
      // Remove the user message if request failed
      setChatMessages(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (showChat) {
        sendChatMessage();
      } else {
        submitAnswer();
      }
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>
              </svg>
            </div>
            <div>
              <h1 className="logo-text">AnaestesiCare</h1>
              <p className="logo-subtitle">AI-supported assessment platform</p>
            </div>
          </div>
          <div className="header-actions">
            <button className="header-btn secondary">Doctor Portal</button>
            <button className="header-btn primary">Patient Portal</button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="content-container">
          
          {/* Info Banner - shown before chat starts */}
          {!showChat && (
            <div className="info-banner">
              <div className="info-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" fill="currentColor"/>
                </svg>
              </div>
              <div className="info-content">
                <h3 className="info-title">Information Assistant for Anesthesia</h3>
                <p className="info-text">
                  Your guide to help you understand anesthesia better. I can provide general information 
                  to help you feel better prepared and informed.
                </p>
                <p className="info-note">
                  <strong>Important:</strong> This chat is for educational and informational purposes only 
                  and does not replace medical advice from your healthcare team. For specific concerns about 
                  your situation, always consult with your anesthesiologist or surgical team.
                </p>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="error-banner">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor"/>
              </svg>
              <span>{error}</span>
            </div>
          )}

          {/* Questionnaire Section */}
          {!questionnaireComplete && currentQuestion && (
            <div className="questionnaire-card">
              <div className="card-header">
                <div className="card-icon questionnaire-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-2 10h-4v4h-2v-4H7v-2h4V7h2v4h4v2z" fill="currentColor"/>
                  </svg>
                </div>
                <h2 className="card-title">Pre-Anesthesia Questionnaire</h2>
              </div>
              <p className="card-subtitle">Please answer the questions below</p>

              {currentQuestion.context && (
                <div className="question-context">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" fill="currentColor"/>
                  </svg>
                  <span>{currentQuestion.context}</span>
                </div>
              )}

              <div className="question-section">
                <label className="question-label">Current Question:</label>
                <div className="question-text">
                  <span className="question-marker">?</span>
                  <span>{currentQuestion.question}</span>
                </div>
              </div>

              <div className="answer-section">
                <label className="input-label" htmlFor="answer-input">
                  Your Answer:
                </label>
                <textarea
                  id="answer-input"
                  className="answer-input"
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your answer here..."
                  rows={4}
                  disabled={loading}
                />
                <button
                  className="submit-btn"
                  onClick={submitAnswer}
                  disabled={!answer.trim() || loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Processing...
                    </>
                  ) : (
                    <>
                      Submit Answer
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                        <path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z" fill="currentColor"/>
                      </svg>
                    </>
                  )}
                </button>
              </div>

              {transcript.length > 0 && (
                <div className="question-note">
                  <strong>Have a question?</strong>
                  <p>You can ask questions about the questionnaire at the bottom of this page</p>
                </div>
              )}
            </div>
          )}

          {/* Completion Message */}
          {questionnaireComplete && !showChat && (
            <div className="completion-card">
              <div className="completion-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>
                </svg>
              </div>
              <h2>Questionnaire Complete!</h2>
              <p>Thank you for completing the pre-anesthesia questionnaire.</p>
              <button className="primary-btn" onClick={() => setShowChat(true)}>
                Start Chat with Assistant
              </button>
            </div>
          )}

          {/* Chat Section */}
          {showChat && (
            <div className="chat-card">
              <div className="card-header">
                <div className="card-icon chat-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z" fill="currentColor"/>
                  </svg>
                </div>
                <div>
                  <h2 className="card-title">Anesthesia Information Assistant</h2>
                  <p className="card-subtitle">Ask me anything about your upcoming anesthesia</p>
                </div>
              </div>

              {/* Chat Messages */}
              <div className="chat-messages">
                {chatMessages.length === 0 && (
                  <div className="chat-welcome">
                    <div className="welcome-icon">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>
                      </svg>
                    </div>
                    <h3>Hello! I'm here to help with questions about anesthesia.</h3>
                    <p>I can provide general information to help you feel better prepared and informed. 
                    Ask me anything or choose from the suggested questions below.</p>
                  </div>
                )}

                {chatMessages.map((msg, idx) => (
                  <div key={idx} className={`chat-message ${msg.role}`}>
                    <div className="message-avatar">
                      {msg.role === 'assistant' ? (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>
                        </svg>
                      ) : (
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                          <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="currentColor"/>
                        </svg>
                      )}
                    </div>
                    <div className="message-content">
                      <div className="message-text">{msg.content}</div>
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>

              {/* Suggested Questions */}
              {chatMessages.length === 0 && (
                <div className="suggested-questions">
                  <label className="suggested-label">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="currentColor"/>
                    </svg>
                    Suggested Questions:
                  </label>
                  <div className="question-chips">
                    {suggestedQuestions.map((question, idx) => (
                      <button
                        key={idx}
                        className="question-chip"
                        onClick={() => sendChatMessage(question)}
                        disabled={loading}
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Chat Input */}
              <div className="chat-input-container">
                <input
                  type="text"
                  className="chat-input"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask your question about anesthesia..."
                  disabled={loading}
                />
                <button
                  className="send-btn"
                  onClick={() => sendChatMessage()}
                  disabled={!chatInput.trim() || loading}
                  aria-label="Send message"
                >
                  {loading ? (
                    <span className="spinner-small"></span>
                  ) : (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                      <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor"/>
                    </svg>
                  )}
                </button>
              </div>

              <p className="chat-disclaimer">
                Press Enter to send • This is for general information only
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;