type Props = {
  currentQuestion: string | null;
  value: string;
  disabled: boolean;
  onChange: (v: string) => void;
  onSend: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
};

export default function ChatInput({
  currentQuestion,
  value,
  disabled,
  onChange,
  onSend,
  onKeyDown,
}: Props) {
  return (
    <div className="bg-white border rounded-2xl shadow-sm p-4">
      {currentQuestion && (
        <div className="mb-3 bg-blue-50 border border-blue-100 rounded-xl p-3">
          <div className="text-xs font-semibold text-blue-700 uppercase tracking-wide">
            Aktuelt spørgsmål
          </div>
          <div className="text-sm font-medium text-slate-800 mt-1">{currentQuestion}</div>
        </div>
      )}

      <div className="flex gap-3 items-end">
        <textarea
          className="flex-1 min-h-[44px] max-h-32 resize-none rounded-xl border px-3 py-3 text-sm focus:outline-none focus:ring-4 focus:ring-blue-100 focus:border-blue-500"
          placeholder="Skriv dit svar her..."
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
        />

        <button
          className="h-11 px-4 rounded-xl bg-blue-600 text-white font-semibold text-sm disabled:bg-slate-300"
          onClick={onSend}
          disabled={disabled || !value.trim()}
        >
          Send
        </button>
      </div>

      <div className="mt-3 text-xs text-slate-500 italic">
        Tip: Stil bare dit spørgsmål direkte — systemet registrerer automatisk om du spørger eller svarer.
      </div>
    </div>
  );
}