type AppShellProps = {
  children: React.ReactNode;
};

export default function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Top bar */}
      <header className="bg-white border-b">
        <div className="mx-auto max-w-6xl px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-blue-600 text-white grid place-items-center font-semibold">
              A
            </div>
            <div>
              <div className="font-semibold leading-tight">AnæstesiCare</div>
              <div className="text-sm text-slate-500 leading-tight">
                AI-understøttet præ-anæstesi vurdering
              </div>
            </div>
          </div>

          <div className="flex gap-2">
            <button className="px-3 py-2 rounded-lg border bg-slate-100 text-sm font-semibold">
              Lægevisning
            </button>
            <button className="px-3 py-2 rounded-lg bg-slate-900 text-white text-sm font-semibold">
              Patientvisning
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
    </div>
  );
}