import { useRef, useEffect } from 'react';
import { Send, Bot, User, Lightbulb } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Input } from '@/app/components/ui/input';

// Matcher din API-chat state (fra App)
export type ChatRole = 'bot' | 'user' | 'bot-chat' | 'user-chat' | 'validation-error';

export interface ChatMessage {
  role: ChatRole;
  content: string;
  timestamp: number;
}

type Question = { text: string; questionId: string } | null;

type Props = {
  messages: ChatMessage[];
  answer: string;
  setAnswer: (v: string) => void;
  submitAnswer: () => void;
  loading: boolean;
  isDone: boolean;
  progress: { current: number; total: number };
  currentQuestion: Question;
};

const suggestedQuestions = [
  'Hvad skal jeg gøre for at forberede mig til min operation?',
  'Hvilke risici er der ved fuld bedøvelse?',
  'Hvor lang tid tager det at vågne efter bedøvelse?',
  'Må jeg spise eller drikke før mit indgreb?',
  'Hvilke bivirkninger kan jeg forvente?',
  'Vil jeg mærke smerte under indgrebet?',
];

export function PatientChatbot({
  messages,
  answer,
  setAnswer,
  submitAnswer,
  loading,
  isDone,
  progress,
  currentQuestion,
}: Props) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = () => {
    if (!answer.trim()) return;
    submitAnswer();
  };

  const handleSuggestedQuestion = (question: string) => {
    setAnswer(question);
    // send direkte, så det føles “klik-og-svar”
    // (vi bruger microtask så state når at opdatere inden submit)
    queueMicrotask(() => submitAnswer());
  };

  // UI mapping: hvilken “avatar” skal vises
  const isUser = (role: ChatRole) => role === 'user' || role === 'user-chat';
  const isBot = (role: ChatRole) =>
    role === 'bot' || role === 'bot-chat' || role === 'validation-error';

  // Bubble style baseret på role
  const bubbleClass = (role: ChatRole) => {
    if (isUser(role)) return 'bg-blue-600 text-white';
    if (role === 'validation-error') return 'bg-red-50 border border-red-200 text-red-800';
    if (role === 'bot-chat') return 'bg-amber-50 border border-amber-200 text-amber-900';
    return 'bg-white border border-slate-200 text-slate-700';
  };

  return (
    <div className="h-full bg-gradient-to-b from-blue-50 to-white flex flex-col">
      {/* Header - Kompakt */}
      <div className="bg-white border-b border-slate-200 shadow-sm flex-shrink-0">
        <div className="max-w-4xl mx-auto px-1 py-5">
          <div className="flex items-center gap-2">
            <div className="size-8 rounded-full bg-blue-100 flex items-center justify-center">
              <Bot className="size-5 text-blue-600" />
            </div>

            <div className="min-w-0">
              <h1 className="text-lg font-medium text-slate-900 leading-tight">
                Informationsassistent om bedøvelse
              </h1>
              <p className="text-[10px] text-slate-600 leading-tight">
                Din guide til at forstå bedøvelse
              </p>
            </div>

            {/* Progress (hvis du vil vise den her også)*/}
            {progress.total > 0 && (
              <div className="ml-auto text-[16px] text-slate-600">
                Spørgsmål {progress.current} / {progress.total}
              </div>
            )}
          </div>

          {/* Vigtig bar - kompakt */}
          <div className="mt-1.5 bg-blue-50 border border-blue-200 rounded-md px-2 py-0.5 text-[12px] text-blue-900 leading-snug">
            <span className="font-semibold">Vigtigt:</span>{' '}
            Denne chatbot giver kun generel uddannelsesinformation. Den erstatter ikke rådgivning fra din
            anæstesilæge eller dit sundhedsteam.
          </div>

          {/* Aktuelt spørgsmål banner (fra spørgeskema-flowet)
          {!isDone && currentQuestion && (
            <div className="mt-1.5 bg-slate-50 border border-slate-200 rounded-md px-2 py-0.5 text-[12px] text-slate-700 leading-snug">
              <span className="font-semibold">Aktuelt spørgsmål:</span> {currentQuestion.text}
            </div>
          )}*/}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.map((message, idx) => (
            <div
              key={`${message.timestamp}-${idx}`}
              className={`flex gap-3 ${isUser(message.role) ? 'justify-end' : 'justify-start'}`}
            >
              {isBot(message.role) && (
                <div
                  className={`size-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 ${
                    message.role === 'validation-error'
                      ? 'bg-red-100'
                      : message.role === 'bot-chat'
                        ? 'bg-amber-100'
                        : 'bg-blue-100'
                  }`}
                >
                  <Bot
                    className={`size-4 ${
                      message.role === 'validation-error'
                        ? 'text-red-700'
                        : message.role === 'bot-chat'
                          ? 'text-amber-700'
                          : 'text-blue-600'
                    }`}
                  />
                </div>
              )}

              <div className={`max-w-[70%] rounded-lg px-3 py-2 ${bubbleClass(message.role)}`}>
                <p className="whitespace-pre-line leading-relaxed text-sm">{message.content}</p>

                <div className="mt-1 text-[10px] opacity-70">
                  {new Date(message.timestamp).toLocaleTimeString('da-DK', {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </div>
              </div>

              {isUser(message.role) && (
                <div className="size-8 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0 mt-1">
                  <User className="size-4 text-slate-600" />
                </div>
              )}
            </div>
          ))}

          {/* Loading indicator (matcher din gamle “typing”)*/}
          {loading && (
            <div className="flex gap-3 justify-start">
              <div className="size-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                <Bot className="size-4 text-blue-600" />
              </div>
              <div className="max-w-[70%] rounded-lg px-3 py-2 bg-white border border-slate-200 text-slate-700">
                <p className="text-sm">Skriver…</p>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Foreslåede spørgsmål (vis kun i starten)
      {messages.length === 1 && !isDone && (
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
                  disabled={loading}
                  className="text-left p-2 bg-white border border-slate-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-xs text-slate-700 disabled:opacity-50"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}*/}

      {/* Input */}
      <div className="bg-white border-t border-slate-200 p-3 flex-shrink-0">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <Input
              key={`input-${messages.length}`}
              autoFocus
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Skriv dit svar / spørgsmål..."
              className="flex-1 bg-slate-50 border-slate-300"
              disabled={loading || isDone}
            />
            <Button
              onClick={handleSend}
              disabled={loading || isDone || !answer.trim()}
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