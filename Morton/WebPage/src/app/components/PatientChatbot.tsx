import { useRef, useEffect } from 'react';
import { Send, Bot, User, Download } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Input } from '@/app/components/ui/input';

export interface ChatMessage {
  role: 'bot' | 'user';
  content: string;
  timestamp: number;
}

type Props = {
  messages: ChatMessage[];
  answer: string;
  setAnswer: (v: string) => void;
  submitAnswer: () => void;
  loading: boolean;
  isDone: boolean;
  answeredCount: number;
  conversationId: string;
};

export function PatientChatbot({
  messages,
  answer,
  setAnswer,
  submitAnswer,
  loading,
  isDone,
  answeredCount,
  conversationId
}: Props) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = () => {
    if (!answer.trim()) return;
    submitAnswer();
  };

const downloadTranscript = () => {
    const lines = messages.map((m) => {
      const time = new Date(m.timestamp).toLocaleTimeString('da-DK', {
        hour: '2-digit',
        minute: '2-digit'
      });
      const speaker = m.role === 'user' ? 'Patient' : 'Assistant';
      return `[${time}] ${speaker}: ${m.content}`;
    });

    const text = [
      `${conversationId} — transcript`,
      `Date: ${new Date().toLocaleDateString('da-DK')}`,
      '─'.repeat(50),
      '',
      ...lines,
    ].join('\n');

    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${conversationId}.txt`; // transcript filename
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="h-full bg-gradient-to-b from-blue-50 to-white flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 shadow-sm flex-shrink-0">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="size-8 rounded-full bg-blue-100 flex items-center justify-center">
              <Bot className="size-5 text-blue-600" />
            </div>
            <div className="min-w-0">
              <h1 className="text-lg font-medium text-slate-900 leading-tight">
                Clinical assistant for anesthesia
              </h1>
              <p className="text-xs text-slate-500 leading-tight">
                Preperation for anesthesia
              </p>
            </div>
            {answeredCount > 0 && (
              <div className="ml-auto flex items-center gap-3">
                <span className="text-sm text-slate-500">{answeredCount} question(s) answered</span>
                <button
                  onClick={downloadTranscript}
                  title="Download transcript"
                  className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 border border-slate-200 hover:border-slate-400 rounded-md px-2 py-1 transition-colors"
                >
                  <Download className="size-3.5" />
                  Download transcript
                </button>
              </div>
            )}
          </div>

          <div className="mt-2 bg-blue-50 border border-blue-200 rounded-md px-3 py-1.5 text-xs text-blue-900">
            <span className="font-semibold">Important:</span>{' '}
            This assistant only provides general information. It does not replace guidance from medical professionals.
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.map((message, idx) => (
            <div
              key={`${message.timestamp}-${idx}`}
              className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.role === 'bot' && (
                <div className="size-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                  <Bot className="size-4 text-blue-600" />
                </div>
              )}

              <div
                className={`max-w-[70%] rounded-lg px-3 py-2 ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white border border-slate-200 text-slate-700'
                }`}
              >
                <p className="whitespace-pre-line leading-relaxed text-sm">{message.content}</p>
                <div className="mt-1 text-[10px] opacity-60">
                  {new Date(message.timestamp).toLocaleTimeString('da-DK', {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </div>
              </div>

              {message.role === 'user' && (
                <div className="size-8 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0 mt-1">
                  <User className="size-4 text-slate-600" />
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-3">
              <div className="size-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                <Bot className="size-4 text-blue-600" />
              </div>
              <div className="flex gap-1">
                <span className="h-2 w-2 rounded-full bg-slate-300 inline-block animate-bounce" />
                <span className="h-2 w-2 rounded-full bg-slate-300 inline-block animate-bounce [animation-delay:120ms]" />
                <span className="h-2 w-2 rounded-full bg-slate-300 inline-block animate-bounce [animation-delay:240ms]" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

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
              placeholder={isDone ? 'The consultation has ended' : 'Type your answer or question…'}
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
          <p className="text-xs text-slate-400 mt-1">
            Press Enter to send
          </p>
        </div>
      </div>
    </div>
  );
}