import { useState } from 'react';
import { ChevronLeft, ChevronRight, Clock, User } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { getScheduledPatients, type Patient } from '@/data/mockPatients';
import { format, addDays, subDays } from 'date-fns';

interface CalendarViewProps {
  onPatientClick: (patient: Patient) => void;
}

export function CalendarView({ onPatientClick }: CalendarViewProps) {
  const [selectedDate, setSelectedDate] = useState(new Date('2026-01-24'));
  const scheduledPatients = getScheduledPatients(selectedDate);

  const getAsaBadgeColor = (score: number) => {
    if (score === 1) return 'bg-green-100 text-green-800 border-green-200';
    if (score === 2) return 'bg-blue-100 text-blue-800 border-blue-200';
    if (score === 3) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    if (score === 4) return 'bg-orange-100 text-orange-800 border-orange-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  const timeSlots = [
    '08:00', '08:30', '09:00', '09:30', '10:00', '10:30',
    '11:00', '11:30', '12:00', '12:30', '13:00', '13:30',
    '14:00', '14:30', '15:00', '15:30', '16:00'
  ];

  const getPatientAtTime = (time: string) => {
    return scheduledPatients.find(p => p.procedureTime === time);
  };

  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl text-slate-900 mb-2">Operationsplan</h1>
          <p className="text-slate-600">Se og administrér dagens indgreb</p>
        </div>

        {/* Date Navigation */}
        <Card className="p-4 mb-6 bg-white border-slate-200">
          <div className="flex items-center justify-between">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedDate(subDays(selectedDate, 1))}
              className="border-slate-300"
            >
              <ChevronLeft className="size-4 mr-1" />
              Forrige dag
            </Button>

            <div className="text-center">
              <h2 className="text-xl text-slate-900">{format(selectedDate, 'EEEE, MMMM d, yyyy')}</h2>
              <p className="text-sm text-slate-500 mt-1">
                {scheduledPatients.length} indgreb planlagt
              </p>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedDate(addDays(selectedDate, 1))}
              className="border-slate-300"
            >
              Næste dag
              <ChevronRight className="size-4 ml-1" />
            </Button>
          </div>
        </Card>

        {/* Schedule Grid */}
        <div className="grid grid-cols-1 gap-1">
          {timeSlots.map((time) => {
            const patient = getPatientAtTime(time);

            return (
              <div key={time} className="flex gap-3">
                {/* Time Column */}
                <div className="w-20 pt-2 text-right flex-shrink-0">
                  <div className="flex items-center justify-end gap-1 text-xs text-slate-600">
                    <Clock className="size-3" />
                    <span>{time}</span>
                  </div>
                </div>

                {/* Appointment Column */}
                <div className="flex-1">
                  {patient ? (
                    <button
                      onClick={() => onPatientClick(patient)}
                      className="w-full text-left"
                    >
                      <Card className="p-2.5 bg-white border-slate-200 hover:border-slate-400 hover:shadow-md transition-all cursor-pointer">
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex items-start gap-2 flex-1 min-w-0">
                            <div className="size-8 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0">
                              <User className="size-4 text-slate-600" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-0.5">
                                <h3 className="text-sm text-slate-900">
                                  {patient.firstName} {patient.lastName}
                                </h3>
                                <span className="text-xs text-slate-500">({patient.age} år)</span>
                              </div>
                              <p className="text-xs text-slate-700">{patient.scheduledProcedure}</p>
                              {patient.comorbidities.length > 0 && (
                                <p className="text-xs text-slate-500 mt-0.5">
                                  {patient.comorbidities.slice(0, 2).join(', ')}
                                  {patient.comorbidities.length > 2 && ` +${patient.comorbidities.length - 2} flere`}
                                </p>
                              )}
                            </div>
                          </div>
                          <Badge className={`${getAsaBadgeColor(patient.asaScore)} flex-shrink-0 text-xs px-2 py-0.5`}>
                            ASA {patient.asaScore}
                          </Badge>
                        </div>
                      </Card>
                    </button>
                  ) : (
                    <div className="h-full min-h-[40px] border border-dashed border-slate-200 rounded-lg bg-slate-50/50"></div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
