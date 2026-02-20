type ChatHeaderProps = {
  title: string;
  subtitle: string;
};

export default function ChatHeader({ title, subtitle }: ChatHeaderProps) {
  return (
    <div className="bg-white border rounded-2xl shadow-sm overflow-hidden">
      <div className="px-5 py-4 border-b">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-indigo-50 grid place-items-center">
            🤖
          </div>
          <div>
            <div className="font-semibold">{title}</div>
            <div className="text-sm text-slate-500">{subtitle}</div>
          </div>
        </div>
      </div>

      <div className="px-5 py-4">
        <div className="text-sm bg-blue-50 border border-blue-100 rounded-xl p-3 text-slate-700">
          <span className="font-semibold">Vigtigt:</span> Denne chatbot giver kun generel
          uddannelsesinformation. Den erstatter ikke rådgivning fra din anæstesilæge eller dit sundhedsteam.
        </div>
      </div>
    </div>
  );
}