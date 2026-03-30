import { useEffect, useMemo, useState } from 'react';
import { Calendar, Search, User, Stethoscope } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { PatientSearch } from '@/app/components/PatientSearch';
import { CalendarView } from '@/app/components/CalendarView';
import { PatientDetails } from '@/app/components/PatientDetails';
import { PatientSummaryModal } from '@/app/components/PatientSummaryModal';
import { PatientChatbot } from '@/app/components/PatientChatbot';
import { ConsultationSummary, type SummaryData } from '@/app/components/ConsultationSummary';

type View = 'search' | 'calendar' | 'details' | 'chatbot' | 'consultation-summary';
type UserRole = 'doctor' | 'patient';

export interface Message {
  role: 'bot' | 'user';
  content: string;
  timestamp: number;
}

const API_BASE = 'http://localhost:8000/api/v1';

export default function App() {
  const [userRole, setUserRole] = useState<UserRole>('doctor');
  const [currentView, setCurrentView] = useState<View>('search');
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [modalPatient, setModalPatient] = useState<Patient | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // ---- Chatbot state ----
  const conversationId = useMemo(() => `patient-${Date.now()}`, []);
  const [messages, setMessages] = useState<Message[]>([]);
  const [answer, setAnswer] = useState('');
  const [isDone, setIsDone] = useState(false);
  const [loading, setLoading] = useState(false);
  const [answeredCount, setAnsweredCount] = useState(0);

    // ---- Summary state (shared between patient completion & doctor view) ----
  const [consultationSummary, setConsultationSummary] = useState<SummaryData | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);

  // ---- Fetch summary when consultation completes ----
  useEffect(() => {
    if (!isDone || consultationSummary || summaryLoading) return;

    const fetchSummary = async () => {
      setSummaryLoading(true);
      try {
        const res = await fetch(`${API_BASE}/conversations/${conversationId}/summary`);
        if (res.ok) {
          const data = await res.json();
          setConsultationSummary(data);
        }
      } catch {
        // Summary fetch failed — doctor can still use other views
      } finally {
        setSummaryLoading(false);
      }
    };

    fetchSummary();
  }, [isDone, conversationId, consultationSummary, summaryLoading]);

   // ---- Doctor navigation ----
  const handlePatientSelect = (patient: Patient) => {
    setSelectedPatient(patient);
    setCurrentView('details');
  };

  const handlePatientClick = (patient: Patient) => {
    setModalPatient(patient);
    setIsModalOpen(true);
  };

  const handleViewDetails = (patient: Patient) => {
    setIsModalOpen(false);
    setSelectedPatient(patient);
    setCurrentView('details');
  };

  const handleBackToCalendar = () => {
    setSelectedPatient(null);
    setCurrentView('calendar');
  };

  const switchRole = (role: UserRole) => {
    setUserRole(role);
    if (role === 'patient') {
      setCurrentView('chatbot');
    } else {
      // When switching to doctor view, show summary if one exists
      setCurrentView(consultationSummary ? 'consultation-summary' : 'search');
    }
    setSelectedPatient(null);
  };

  // ---- Start conversation ----
  const startConversation = async () => {
    try {
      const res = await fetch(`${API_BASE}/conversations/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: conversationId }),
      });
      const data = await res.json();
      setMessages([{ role: 'bot', content: data.bot_text, timestamp: Date.now() }]);
      setIsDone(data.done);
    } catch {
      setMessages([{
        role: 'bot',
        content: 'Kunne ikke forbinde til API. Er serveren startet?',
        timestamp: Date.now(),
      }]);
    }
  };

  useEffect(() => {
    if (userRole !== 'patient') return;
    if (messages.length > 0) return;
    startConversation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userRole]);

  // ---- Submit any message (answer or question) ----
  const submitAnswer = async () => {
    if (!answer.trim() || loading) return;

    const userMessage = answer.trim();
    setAnswer('');
    setLoading(true);

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: userMessage, timestamp: Date.now() },
    ]);

    try {
      const res = await fetch(`${API_BASE}/conversations/${conversationId}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: 'bot', content: data.bot_text, timestamp: Date.now() },
      ]);

      setIsDone(data.done);

      // Update answered count from state endpoint (fire-and-forget)
      fetch(`${API_BASE}/conversations/${conversationId}/state`)
        .then((r) => r.json())
        .then((s) => setAnsweredCount(s.answered_count ?? 0))
        .catch(() => {});
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          content: 'Der opstod en fejl. Prøv venligst igen.',
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // ---- Determine if the "Latest consultation" tab should show ----
  const showSummaryTab = consultationSummary != null;

  return (
     <div className="h-screen flex flex-col overflow-hidden">
      {/* Top Navigation Bar */}
      <div className="bg-white border-b border-slate-200 shadow-sm flex-shrink-0">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="size-10 rounded-lg bg-blue-600 flex items-center justify-center">
                <Stethoscope className="size-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl text-slate-900">AnæstesiCare</h1>
                <p className="text-xs text-slate-500">AI-supported assessment tool</p>
              </div>
            </div>

            <div className="flex items-center gap-2 bg-slate-100 p-1 rounded-lg">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => switchRole('doctor')}
                className={
                  userRole === 'doctor'
                    ? 'bg-slate-800 text-white shadow-sm hover:bg-slate-900 hover:text-white'
                    : 'text-slate-600 hover:bg-slate-200 hover:text-slate-900'
                }
              >
                <Stethoscope className="size-4 mr-2" />
                Doctor's view
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => switchRole('patient')}
                className={
                  userRole === 'patient'
                    ? 'bg-slate-800 text-white shadow-sm hover:bg-slate-900 hover:text-white'
                    : 'text-slate-600 hover:bg-slate-200 hover:text-slate-900'
                }
              >
                <User className="size-4 mr-2" />
                Patient view
              </Button>
            </div>
          </div>

          {userRole === 'doctor' && (
            <div className="flex items-center gap-2 mt-4 border-t border-slate-200 pt-4">
              {showSummaryTab && (
                <Button
                  variant={currentView === 'consultation-summary' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setCurrentView('consultation-summary')}
                  className={currentView === 'consultation-summary' ? '' : 'text-slate-600'}
                >
                  <Stethoscope className="size-4 mr-2" />
                  Latest consultation
                  <span className="ml-1.5 inline-flex items-center justify-center size-2 rounded-full bg-emerald-400" />
                </Button>
              )}

              <Button
                variant={currentView === 'search' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setCurrentView('search')}
                className={currentView === 'search' ? '' : 'text-slate-600'}
              >
                <Search className="size-4 mr-2" />
                Patient search
              </Button>

              <Button
                variant={currentView === 'calendar' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setCurrentView('calendar')}
                className={currentView === 'calendar' ? '' : 'text-slate-600'}
              >
                <Calendar className="size-4 mr-2" />
                Time schedule
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {userRole === 'doctor' ? (
          <>
            {currentView === 'consultation-summary' && consultationSummary && (
              <ConsultationSummary
                summary={consultationSummary}
                onDismiss={() => setCurrentView('search')}
              />
            )}
            {currentView === 'search' && <PatientSearch onPatientSelect={handlePatientSelect} />}
            {currentView === 'calendar' && <CalendarView onPatientClick={handlePatientClick} />}
            {currentView === 'details' && selectedPatient && (
              <PatientDetails patient={selectedPatient} onBack={handleBackToCalendar} />
            )}
            <PatientSummaryModal
              patient={modalPatient}
              isOpen={isModalOpen}
              onClose={() => setIsModalOpen(false)}
              onViewDetails={handleViewDetails}
            />
          </>
        ) : (
          <PatientChatbot
            messages={messages}
            answer={answer}
            setAnswer={setAnswer}
            submitAnswer={submitAnswer}
            loading={loading}
            isDone={isDone}
            answeredCount={answeredCount}
            conversationId={conversationId}
          />
        )}
      </div>
    </div>
  );
}