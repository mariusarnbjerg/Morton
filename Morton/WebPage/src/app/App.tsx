import { useEffect, useMemo, useState } from 'react';
import { Calendar, Search, User, Stethoscope } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { PatientSearch } from '@/app/components/PatientSearch';
import { CalendarView } from '@/app/components/CalendarView';
import { PatientDetails } from '@/app/components/PatientDetails';
import { PatientSummaryModal } from '@/app/components/PatientSummaryModal';
import { PatientChatbot } from '@/app/components/PatientChatbot';
import type { Patient } from '@/data/mockPatients';

type View = 'search' | 'calendar' | 'details' | 'chatbot';
type UserRole = 'doctor' | 'patient';

// ---- Chatbot types (fra din gamle App) ----
interface Question {
  text: string;
  questionId: string;
}

export interface Message {
  role: 'bot' | 'user' | 'bot-chat' | 'user-chat' | 'validation-error';
  content: string;
  timestamp: number;
}

export default function App() {
  const [userRole, setUserRole] = useState<UserRole>('doctor');
  const [currentView, setCurrentView] = useState<View>('search');
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [modalPatient, setModalPatient] = useState<Patient | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // ---- Chatbot state (integreret) ----
  const conversationId = useMemo(() => `patient-${Date.now()}`, []);
  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [answer, setAnswer] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isDone, setIsDone] = useState(false);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });

  const API_BASE = 'http://localhost:8000/api/v1';

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
      setCurrentView('search');
    }
    setSelectedPatient(null);
  };

  // ---- Chatbot logic ----
  const fetchProgress = async () => {
    try {
      const response = await fetch(`${API_BASE}/conversations/${conversationId}/state`);
      const data = await response.json();
      setProgress({
        current: data.answered_count || 0,
        total: data.total_questions || 0,
      });
    } catch (error) {
      console.error('Failed to fetch progress:', error);
    }
  };

  const startConversation = async () => {
    try {
      const response = await fetch(`${API_BASE}/conversations/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: conversationId }),
      });

      const data = await response.json();

      setCurrentQuestion({ text: data.question, questionId: data.question_id });
      setMessages([
        {
          role: 'bot',
          content: data.question,
          timestamp: Date.now(),
        },
      ]);
      setIsDone(Boolean(data.done));

      fetchProgress();
    } catch (error) {
      console.error('Failed to start conversation:', error);
      // Du kan vælge at vise den i UI i stedet:
      setMessages([
        {
          role: 'validation-error',
          content: 'Kunne ikke forbinde til API. Er serveren startet?',
          timestamp: Date.now(),
        },
      ]);
    }
  };

  // Start kun samtalen når man går i patient/chatbot-visning
  useEffect(() => {
    if (userRole !== 'patient') return;

    // Undgå at starte igen hvis vi allerede har beskeder
    if (messages.length > 0) return;

    startConversation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userRole]);

  const submitAnswer = async () => {
    if (!answer.trim()) return;

    setLoading(true);

    // Add user answer to messages
    setMessages((prev) => [
      ...prev,
      {
        role: 'user',
        content: answer,
        timestamp: Date.now(),
      },
    ]);

    try {
      const response = await fetch(`${API_BASE}/conversations/${conversationId}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answer }),
      });

      const data = await response.json();

      // Handle validation failure
      if (data.validation_failed) {
        const parts = String(data.bot_text || '').split('---\nPlease answer:\n');

        if (parts.length === 2) {
          setMessages((prev) => [
            ...prev,
            { role: 'validation-error', content: parts[0].trim(), timestamp: Date.now() },
            { role: 'bot', content: parts[1].trim(), timestamp: Date.now() },
          ]);
        } else {
          setMessages((prev) => [
            ...prev,
            { role: 'validation-error', content: String(data.bot_text || '').trim(), timestamp: Date.now() },
          ]);
        }

        setAnswer('');
        setLoading(false);
        return;
      }

      // Handle auto-detected question (switch to chat mode)
      if (data.bot_text && String(data.bot_text).includes('---\nBack to the questionnaire:\n')) {
        const parts = String(data.bot_text).split('---\nBack to the questionnaire:\n');

        setMessages((prev) => [
          ...prev,
          {
            role: 'bot-chat',
            content:
              `💡 Jeg opdagede at du stillede et spørgsmål, så lad mig besvare det:\n\n${parts[0].trim()}`,
            timestamp: Date.now(),
          },
        ]);

        setCurrentQuestion({
          text: parts[1].trim(),
          questionId: data.question_id || currentQuestion?.questionId || '',
        });

        setMessages((prev) => [
          ...prev,
          {
            role: 'bot',
            content: parts[1].trim(),
            timestamp: Date.now(),
          },
        ]);

        setAnswer('');
        setLoading(false);
        fetchProgress();
        return;
      }

      // Normal flow: valid answer accepted
      if (data.done) {
        setIsDone(true);
        setMessages((prev) => [
          ...prev,
          {
            role: 'bot',
            content: '✅ Spørgeskema fuldført! Tak for dine svar.',
            timestamp: Date.now(),
          },
        ]);
      } else {
        setCurrentQuestion({ text: data.question, questionId: data.question_id });
        setMessages((prev) => [
          ...prev,
          {
            role: 'bot',
            content: data.question,
            timestamp: Date.now(),
          },
        ]);
      }

      setAnswer('');
      fetchProgress();
    } catch (error) {
      console.error('Failed to submit answer:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'validation-error',
          content: 'Fejl ved indsendelse af svar. Prøv venligst igen.',
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Top Navigation Bar */}
      <div className="bg-white border-b border-slate-200 shadow-sm flex-shrink-0">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo/Brand */}
            <div className="flex items-center gap-3">
              <div className="size-10 rounded-lg bg-blue-600 flex items-center justify-center">
                <Stethoscope className="size-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl text-slate-900">AnæstesiCare</h1>
                <p className="text-xs text-slate-500">AI-understøttet vurderingsplatform</p>
              </div>
            </div>

            {/* Role Switcher */}
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
                Lægevisning
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
                Patientvisning
              </Button>
            </div>
          </div>

          {/* Doctor Navigation */}
          {userRole === 'doctor' && (
            <div className="flex items-center gap-2 mt-4 border-t border-slate-200 pt-4">
              <Button
                variant={currentView === 'search' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setCurrentView('search')}
                className={currentView === 'search' ? '' : 'text-slate-600'}
              >
                <Search className="size-4 mr-2" />
                Patientsøgning
              </Button>

              <Button
                variant={currentView === 'calendar' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setCurrentView('calendar')}
                className={currentView === 'calendar' ? '' : 'text-slate-600'}
              >
                <Calendar className="size-4 mr-2" />
                Tidsplan
              </Button>
            </div>
          )}

          {/* Progress badge kan også vises her i headeren når patient
          {userRole === 'patient' && progress.total > 0 && (
            <div className="mt-4 border-t border-slate-200 pt-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">
                  Spørgsmål {progress.current} af {progress.total}
                </span>
                <div className="h-2 w-48 bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-600"
                    style={{ width: `${(progress.current / progress.total) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          )}*/}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {userRole === 'doctor' ? (
          <>
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
            progress={progress}
            currentQuestion={currentQuestion}
          />
        )}
      </div>
    </div>
  );
}