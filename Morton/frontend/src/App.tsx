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

export default function App() {

    // Herunder defineres nogle state variabler - Når disse ændres, opdaters UI automatisk
    const [conversationId] = useState(`patient-${Date.now()}`); // Sikrer at conversation ID'et er unikt
    const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
    const [answer, setAnswer] = useState('');
    const [messages, setMessages] = useState<UIMessage[]>([]);
    const [isDone, setIsDone] = useState(false);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState({ current: 0, total: 0 });

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
    <AppShell>
      <div className="grid gap-4">
        <ChatHeader
          title="Informationsassistent om bedøvelse"
          subtitle="Din guide til at forstå bedøvelse"
        />

        <MessageList messages={messages} loading={loading} />

        {!isDone && (
          <ChatInput
            currentQuestion={currentQuestion?.text ?? null}
            value={answer}
            disabled={loading}
            onChange={setAnswer}
            onSend={submitAnswer}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submitAnswer();
              }
            }}
          />
        )}

        {isDone && (
          <div className="bg-white border rounded-2xl shadow-sm p-8 text-center">
            <div className="mx-auto h-14 w-14 rounded-full bg-emerald-500 text-white grid place-items-center text-2xl font-bold">
              ✓
            </div>
            <div className="mt-4 text-xl font-bold">Tak for dine svar!</div>
            <div className="mt-2 text-slate-600">
              Dit spørgeskema er nu fuldført og sendt til gennemsyn.
            </div>
          </div>
        )}
      </div>
    </AppShell>
  );
}