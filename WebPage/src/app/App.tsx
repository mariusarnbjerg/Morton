import { useState } from 'react';
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

export default function App() {
  const [userRole, setUserRole] = useState<UserRole>('doctor');
  const [currentView, setCurrentView] = useState<View>('search');
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [modalPatient, setModalPatient] = useState<Patient | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

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
                <p className="text-xs text-slate-500">
                  AI-understøttet vurderingsplatform
                </p>
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
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {userRole === 'doctor' ? (
          <>
            {currentView === 'search' && (
              <PatientSearch onPatientSelect={handlePatientSelect} />
            )}

            {currentView === 'calendar' && (
              <CalendarView onPatientClick={handlePatientClick} />
            )}

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
          <PatientChatbot />
        )}
      </div>
    </div>
  );
}
