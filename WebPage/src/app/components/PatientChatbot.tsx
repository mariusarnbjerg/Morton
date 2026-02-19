import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Lightbulb } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Input } from '@/app/components/ui/input';
import { apiClient } from '@/api/client';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const suggestedQuestions = [
  "Hvad skal jeg gøre for at forberede mig til min operation?",
  "Hvilke risici er der ved fuld bedøvelse?",
  "Hvor lang tid tager det at vågne efter bedøvelse?",
  "Må jeg spise eller drikke før mit indgreb?",
  "Hvilke bivirkninger kan jeg forvente?",
  "Vil jeg mærke smerte under indgrebet?"
];

export function PatientChatbot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize conversation when component mounts
  useEffect(() => {
    initializeConversation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const initializeConversation = async () => {
    if (isInitialized) return;

    try {
      setIsLoading(true);
      setError(null);
      console.log('🔄 Initializing conversation...');

      const response = await apiClient.startConversation(); // returns conversation_id + maybe question
      setConversationId(response.conversation_id);
      setIsInitialized(true);

      console.log('✅ Conversation initialized with ID:', response.conversation_id);

      const welcomeMessage: Message = {
        id: 'welcome',
        text:
          "Hej! Jeg er her for at hjælpe med at besvare dine spørgsmål om bedøvelse og dit kommende indgreb. Jeg kan give generel information, så du kan føle dig bedre forberedt og informeret.\n\n" +
          "Husk, at denne chat kun er til undervisnings- og informationsformål og ikke erstatter medicinsk rådgivning fra dit sundhedsteam. Ved specifikke bekymringer om din situation bør du altid tale med din anæstesilæge eller operationsteam.\n\n" +
          "Hvordan kan jeg hjælpe dig i dag?",
        sender: 'bot',
        timestamp: new Date(),
      };

      // If backend provides a first question, show it as a separate bot message
      const backendQuestion = response.question?.trim();
      const initialMessages: Message[] = [welcomeMessage];

      if (backendQuestion) {
        initialMessages.push({
          id: 'backend-question',
          text: backendQuestion,
          sender: 'bot',
          timestamp: new Date(),
        });
      }

      setMessages(initialMessages);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'Failed to initialize conversation';
      setError(errorMessage);
      console.error('❌ Initialization error:', err);

      const errorMsg: Message = {
        id: 'error-init',
        text: `Beklager, jeg kunne ikke oprette forbindelse til serveren. Fejl: ${errorMessage}`,
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages([errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (text: string) => {
    if (!text.trim()) return;
    if (!conversationId) {
      console.error('❌ No conversation ID available');
      setError('Conversation not initialized. Please refresh the page.');
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      text: text.trim(),
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    setError(null);

    try {
      console.log('📤 Sending message to API...');
      const response = await apiClient.sendChatMessage(conversationId, text.trim());

      const botText = response.text?.trim() || '(tomt svar fra serveren)';

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: botText,
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);
      console.log('✅ Response received and displayed');
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'Failed to send message';
      setError(errorMessage);
      console.error('❌ Send message error:', err);

      const errorMsg: Message = {
        id: 'error-' + Date.now(),
        text: `Beklager, jeg kunne ikke behandle din besked. Fejl: ${errorMessage}`,
        sender: 'bot',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestedQuestion = (question: string) => {
    handleSendMessage(question);
  };

  return (
    <div className="h-full bg-gradient-to-b from-blue-50 to-white flex flex-col">
      {/* Header - Kompakt */}
      <div className="bg-white border-b border-slate-200 shadow-sm flex-shrink-0">
        <div className="max-w-4xl mx-auto px-3 py-1.5">
          <div className="flex items-center gap-2">
            <div className="size-7 rounded-full bg-blue-100 flex items-center justify-center">
              <Bot className="size-3.5 text-blue-600" />
            </div>

            <div className="min-w-0">
              <h1 className="text-sm font-medium text-slate-900 leading-tight">
                Informationsassistent om bedøvelse
              </h1>
              <p className="text-[10px] text-slate-600 leading-tight">
                Din guide til at forstå bedøvelse
              </p>
            </div>
          </div>

          <div className="mt-1.5 bg-blue-50 border border-blue-200 rounded-md px-2 py-0.5 text-[10px] text-blue-900 leading-snug">
            <span className="font-semibold">Vigtigt:</span>{' '}
            Denne chatbot giver kun generel uddannelsesinformation. Den erstatter ikke rådgivning fra din anæstesilæge eller dit sundhedsteam.
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2">
          <div className="max-w-4xl mx-auto flex items-center gap-2 text-sm text-red-800">
            <svg className="size-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <span>{error}</span>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.sender === 'bot' && (
                <div className="size-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                  <Bot className="size-4 text-blue-600" />
                </div>
              )}

              <div
                className={`max-w-[70%] rounded-lg px-3 py-2 ${
                  message.sender === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white border border-slate-200 text-slate-700'
                }`}
              >
                <p className="whitespace-pre-line leading-relaxed text-sm">{message.text}</p>
              </div>

              {message.sender === 'user' && (
                <div className="size-8 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0 mt-1">
                  <User className="size-4 text-slate-600" />
                </div>
              )}
            </div>
          ))}

          {/* Loading indicator */}
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="size-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                <Bot className="size-4 text-blue-600" />
              </div>
              <div className="bg-white border border-slate-200 rounded-lg px-3 py-2">
                <div className="flex gap-1">
                  <span className="size-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="size-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="size-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Suggested questions */}
      {messages.length <= 2 && !isLoading && (
        <div className="bg-slate-50 border-t border-slate-200 p-3 flex-shrink-0">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="size-3.5 text-amber-600" />
              <h3 className="text-xs text-slate-700">Foreslåede spørgsmål:</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {suggestedQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSuggestedQuestion(question)}
                  disabled={isLoading || !conversationId}
                  className="text-left p-2 bg-white border border-slate-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-xs text-slate-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Input */}
      <div className="bg-white border-t border-slate-200 p-3 flex-shrink-0">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !isLoading) {
                  handleSendMessage(inputValue);
                }
              }}
              placeholder="Skriv dit spørgsmål om bedøvelse..."
              className="flex-1 bg-slate-50 border-slate-300"
              disabled={isLoading || !conversationId}
            />
            <Button
              onClick={() => handleSendMessage(inputValue)}
              disabled={!inputValue.trim() || isLoading || !conversationId}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <Send className="size-4" />
            </Button>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Tryk Enter for at sende • Dette er kun til generel information
          </p>
        </div>
      </div>
    </div>
  );
}
