type Role = "bot" | "user" | "bot-chat" | "user-chat" | "validation-error";

export type UIMessage = {
  role: Role;
  content: string;
  timestamp: number;
};

type Props = {
  messages: UIMessage[];
  loading: boolean;
};

export default function MessageList({ messages, loading }: Props) {
  return (
    <div className="bg-slate-50 border rounded-2xl shadow-sm p-4 h-[60vh] overflow-y-auto">
      <div className="space-y-4">
        {messages.map((m, i) => {
          const isUser = m.role === "user" || m.role === "user-chat";
          const isError = m.role === "validation-error";
          const bubbleBase =
            "max-w-[70%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm";

          const bubbleClass = isUser
            ? `${bubbleBase} bg-blue-600 text-white ml-auto`
            : isError
              ? `${bubbleBase} bg-white border border-red-200 text-slate-800`
              : `${bubbleBase} bg-white border text-slate-800`;

          return (
            <div key={i} className={`flex items-end gap-2 ${isUser ? "justify-end" : "justify-start"}`}>
              {!isUser && (
                <div className="h-8 w-8 rounded-full bg-slate-200 grid place-items-center text-sm">
                  🤖
                </div>
              )}

              <div className={bubbleClass}>
                <div className="whitespace-pre-wrap break-words">{m.content}</div>
                <div className={`mt-2 text-xs ${isUser ? "text-white/70" : "text-slate-400"}`}>
                  {new Date(m.timestamp).toLocaleTimeString("da-DK", { hour: "2-digit", minute: "2-digit" })}
                </div>
              </div>

              {isUser && (
                <div className="h-8 w-8 rounded-full bg-blue-100 grid place-items-center text-sm">
                  👤
                </div>
              )}
            </div>
          );
        })}

        {loading && (
          <div className="flex items-end gap-2">
            <div className="h-8 w-8 rounded-full bg-slate-200 grid place-items-center text-sm">🤖</div>
            <div className="bg-white border rounded-2xl px-4 py-3 text-sm shadow-sm">
              <div className="flex gap-1">
                <span className="h-2 w-2 rounded-full bg-slate-300 inline-block animate-bounce" />
                <span className="h-2 w-2 rounded-full bg-slate-300 inline-block animate-bounce [animation-delay:120ms]" />
                <span className="h-2 w-2 rounded-full bg-slate-300 inline-block animate-bounce [animation-delay:240ms]" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}