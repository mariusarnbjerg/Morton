import { useState, useEffect } from "react";
import AppShell from "./components/layout/AppShell";
import ChatHeader from "./components/chat/ChatHeader";
import MessageList, { UIMessage } from "./components/chat/MessageList";
import ChatInput from "./components/chat/ChatInput";

// Typescript interface der svarer til et spørgsmål fra API'et
interface Question {
  text: string;
  questionId: string;
}

// Typescript interface der svarer til en besked i chatten
interface Message {
  role: 'bot' | 'user' | 'bot-chat' | 'user-chat' | 'validation-error';
  content: string;
  timestamp: number;
}

function App() {

    // Herunder defineres nogle state variabler - Når disse ændres, opdaters UI automatisk
    const [conversationId] = useState(`patient-${Date.now()}`); // Sikrer at conversation ID'et er unikt
    const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
    const [answer, setAnswer] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [isDone, setIsDone] = useState(false);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState({ current: 0, total: 0 });

  // Start the conversation when component loads
  useEffect(() => {
    startConversation();
  }, []); // Empty array means this only runs once when it loads

  // API configuration
  const API_BASE = 'http://localhost:8000/api/v1';

// Function that's in charge of starting the conversation
  const startConversation = async () => {
    try {
      const response = await fetch(`${API_BASE}/conversations/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: conversationId })
      });
      const data = await response.json();

      setCurrentQuestion({ text: data.question, questionId: data.question_id });
      setMessages([{
        role: 'bot',
        content: data.question,
        timestamp: Date.now()
      }]);
      setIsDone(data.done);

      // Get initial state for progress
      fetchProgress();
    } catch (error) {
      console.error('Failed to start conversation:', error);
      alert('Could not connect to API. Make sure the server is running!');
    }
  };

    // Gets the progress of the questions
  const fetchProgress = async () => {
    try {
      const response = await fetch(`${API_BASE}/conversations/${conversationId}/state`);
      const data = await response.json();
      setProgress({
        current: data.answered_count || 0,
        total: data.total_questions || 0
      });
    } catch (error) {
      console.error('Failed to fetch progress:', error);
    }
  };

    // Function in charge of submitting content into the input field
  const submitAnswer = async () => {
      // If the answer string is empty, return and don't allow messages to be sent
    if (!answer.trim()) return;

    setLoading(true);

    // Add user answer to messages
    setMessages(prev => [...prev, {
      role: 'user',
      content: answer,
      timestamp: Date.now()
    }]);

    try {
      const response = await fetch(`${API_BASE}/conversations/${conversationId}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answer })
      });
      const data = await response.json();

      // Handle validation failure
      if (data.validation_failed) {
        const parts = data.bot_text.split('---\nPlease answer:\n');

        if (parts.length === 2) {
          setMessages(prev => [...prev, {
            role: 'validation-error',
            content: parts[0].trim(),
            timestamp: Date.now()
          }]);

          setMessages(prev => [...prev, {
            role: 'bot',
            content: parts[1].trim(),
            timestamp: Date.now()
          }]);
        } else {
          setMessages(prev => [...prev, {
            role: 'validation-error',
            content: data.bot_text,
            timestamp: Date.now()
          }]);
        }

        setAnswer('');
        setLoading(false);
        return;
      }

      // Handle auto-detected question (switch to chat mode)
      if (data.bot_text && data.bot_text.includes('---\nBack to the questionnaire:\n')) {
        const parts = data.bot_text.split('---\nBack to the questionnaire:\n');

        setMessages(prev => [...prev, {
          role: 'bot-chat',
          content: `💡 Jeg opdagede at du stillede et spørgsmål, så lad mig besvare det:\n\n${parts[0].trim()}`,
          timestamp: Date.now()
        }]);

        setCurrentQuestion({
          text: parts[1].trim(),
          questionId: data.question_id || currentQuestion?.questionId || ''
        });

        setMessages(prev => [...prev, {
          role: 'bot',
          content: parts[1].trim(),
          timestamp: Date.now()
        }]);

        setAnswer('');
        setLoading(false);
        fetchProgress();
        return;
      }

      // Normal flow: valid answer accepted
      if (data.done) {
        setIsDone(true);
        setMessages(prev => [...prev, {
          role: 'bot',
          content: '✅ Spørgeskema fuldført! Tak for dine svar.',
          timestamp: Date.now()
        }]);
      } else {
        setCurrentQuestion({ text: data.question, questionId: data.question_id });
        setMessages(prev => [...prev, {
          role: 'bot',
          content: data.question,
          timestamp: Date.now()
        }]);
      }

      setAnswer('');
      fetchProgress();
    } catch (error) {
      console.error('Failed to submit answer:', error);
      setMessages(prev => [...prev, {
        role: 'validation-error',
        content: 'Fejl ved indsendelse af svar. Prøv venligst igen.',
        timestamp: Date.now()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitAnswer();
    }
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    const messagesContainer = document.querySelector('.messages-container');
    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
              </svg>
            </div>
            <div>
              <h1 className="logo-title">AnæstesiCare</h1>
              <p className="logo-subtitle">AI-understøttet præ-anæstesi vurdering</p>
            </div>
          </div>

          {progress.total > 0 && (
            <div className="progress-indicator">
              <span className="progress-text">
                Spørgsmål {progress.current} af {progress.total}
              </span>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${(progress.current / progress.total) * 100}%` }}
                />
              </div>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        <div className="chat-container">
          {/* Messages */}
          <div className="messages-container">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message-wrapper ${msg.role}`}>
                <div className="message-bubble">
                  <div className="message-content">{msg.content}</div>
                  <div className="message-time">
                    {new Date(msg.timestamp).toLocaleTimeString('da-DK', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </div>
                </div>
              </div>
            ))}

            {loading && (
              <div className="message-wrapper bot">
                <div className="message-bubble">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          {!isDone && currentQuestion && (
            <div className="input-container">
              <div className="current-question-banner">
                <span className="question-label">Aktuelt spørgsmål:</span>
                <span className="question-text">{currentQuestion.text}</span>
              </div>

              <div className="input-wrapper">
                <textarea
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Skriv dit svar her..."
                  disabled={loading}
                  className="input-field"
                  rows={2}
                />

                <button
                  onClick={submitAnswer}
                  disabled={loading || !answer.trim()}
                  className="submit-button"
                >
                  {loading ? (
                    <span>Sender...</span>
                  ) : (
                    <>
                      <span>Send</span>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                      </svg>
                    </>
                  )}
                </button>
              </div>

              <p className="input-hint">
                💡 Tip: Stil bare dit spørgsmål direkte - systemet registrerer automatisk om du spørger eller svarer
              </p>
            </div>
          )}

          {/* Completion */}
          {isDone && (
            <div className="completion-container">
              <div className="completion-content">
                <div className="completion-icon">✓</div>
                <h2 className="completion-title">Tak for dine svar!</h2>
                <p className="completion-text">
                  Dit spørgeskema er nu fuldført og sendt til gennemsyn.
                </p>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;