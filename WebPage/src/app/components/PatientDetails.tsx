import { ArrowLeft, AlertCircle, Brain, Activity, FileText, Pill, AlertTriangle } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { Progress } from '@/app/components/ui/progress';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/app/components/ui/collapsible';
import type { Patient } from '@/data/mockPatients';
import { format } from 'date-fns';
import { useState } from 'react';

interface PatientDetailsProps {
  patient: Patient;
  onBack: () => void;
}

export function PatientDetails({ patient, onBack }: PatientDetailsProps) {
  const [openSections, setOpenSections] = useState({
    summary: true,
    explanation: true,
    history: false,
    medications: false
  });

  const toggleSection = (section: keyof typeof openSections) => {
    setOpenSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const getAsaBadgeColor = (score: number) => {
    if (score === 1) return 'bg-green-100 text-green-800 border-green-200';
    if (score === 2) return 'bg-blue-100 text-blue-800 border-blue-200';
    if (score === 3) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    if (score === 4) return 'bg-orange-100 text-orange-800 border-orange-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  const getImpactColor = (impact: string) => {
    if (impact.includes('Low risk')) return 'text-green-700';
    if (impact.includes('Medium risk')) return 'text-yellow-700';
    if (impact.includes('High risk')) return 'text-red-700';
    return 'text-slate-700';
  };

  return (
    <div className="min-h-screen bg-slate-50 p-4">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-4">
          <Button variant="ghost" onClick={onBack} className="mb-2 -ml-2 h-8 text-sm">
            <ArrowLeft className="size-4 mr-2" />
            Tilbage til kalender
          </Button>

          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-2xl text-slate-900 mb-1">
                {patient.firstName} {patient.lastName}
              </h1>
              <div className="flex items-center gap-3 text-sm text-slate-600">
                <span>CPR: {patient.cprNumber}</span>
                <span>•</span>
                <span>{patient.age} år</span>
                <span>•</span>
                <span>{patient.gender === 'M' ? 'Mand' : 'Kvinde'}</span>
              </div>
            </div>
            <Badge className={`${getAsaBadgeColor(patient.asaScore)} text-base px-3 py-1`}>
              ASA {patient.asaScore}
            </Badge>
          </div>
        </div>

        {/* Surgery Details */}
        <Card className="p-4 mb-3 bg-white border-slate-200">
          <h2 className="text-sm text-slate-900 mb-3 flex items-center gap-2">
            <Activity className="size-4" />
            Planlagt indgreb
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-slate-500 mb-0.5">Indgreb</p>
              <p className="text-sm text-slate-900">{patient.scheduledProcedure}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-0.5">Dato og tidspunkt</p>
              <p className="text-sm text-slate-900">
                {format(patient.procedureDate, 'EEEE, MMMM d, yyyy')} kl. {patient.procedureTime}
              </p>
            </div>
          </div>
        </Card>

        {/* ASA Classification Details */}
        <Card className="p-4 mb-3 bg-white border-slate-200">
          <h2 className="text-sm text-slate-900 mb-3 flex items-center gap-2">
            <AlertCircle className="size-4" />
            ASA-klassifikation
          </h2>
          <div className="bg-blue-50 p-3 rounded-lg border border-blue-100 mb-3">
            <p className="text-sm text-slate-700">{patient.asaExplanation}</p>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="text-xs text-slate-500 mb-0.5">BMI</p>
              <p className="text-sm text-slate-900">{patient.bmi}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-0.5">Rygestatus</p>
              <p className="text-sm text-slate-900">{patient.smokingStatus}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-0.5">Risikofaktorer</p>
              <p className="text-sm text-slate-900">{patient.riskIndicators.length}</p>
            </div>
          </div>

          {patient.comorbidities.length > 0 && (
            <div className="mt-3 pt-3 border-t border-slate-200">
              <p className="text-xs text-slate-500 mb-2">Komorbiditeter</p>
              <div className="flex flex-wrap gap-1.5">
                {patient.comorbidities.map((condition, idx) => (
                  <Badge key={idx} variant="outline" className="bg-slate-50 text-slate-700 border-slate-300 text-xs px-2 py-0.5">
                    {condition}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </Card>

        {/* AI Explainability Section */}
        <Card className="p-4 mb-3 bg-white border-slate-200">
          <Collapsible open={openSections.explanation} onOpenChange={() => toggleSection('explanation')}>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between">
                <h2 className="text-sm text-slate-900 flex items-center gap-2">
                  <Brain className="size-4" />
                  Forklarbarhed af AI-model
                </h2>
                <span className="text-xs text-slate-500">
                  {openSections.explanation ? 'Skjul' : 'Vis'} detaljer
                </span>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3">
              <div className="mb-4">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs text-slate-600">Model-sikkerhed</span>
                  <span className="text-xs text-slate-900">{Math.round(patient.modelConfidence * 100)}%</span>
                </div>
                <Progress value={patient.modelConfidence * 100} className="h-1.5 mb-1" />
                <p className="text-xs text-slate-500">
                  Denne score afspejler AI’ens sikkerhed baseret på tilgængelige kliniske data
                </p>
              </div>

              <div className="space-y-2">
                <p className="text-xs text-slate-600 mb-2">Bidragende faktorer og vægtning:</p>
                {patient.aiExplanation.map((factor, idx) => (
                  <div key={idx} className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                    <div className="flex items-start justify-between mb-1.5">
                      <span className="text-sm text-slate-900">{factor.factor}</span>
                      <Badge variant="outline" className="bg-white text-slate-700 border-slate-300 text-xs px-2 py-0.5">
                        Vægt: {Math.round(factor.weight * 100)}%
                      </Badge>
                    </div>
                    <div className="mb-1.5">
                      <Progress value={factor.weight * 100} className="h-1" />
                    </div>
                    <p className={`text-xs ${getImpactColor(factor.impact)}`}>
                      {factor.impact}
                    </p>
                  </div>
                ))}
              </div>

              <div className="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="size-3.5 text-amber-700 mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-amber-800">
                    <p className="mb-0.5">Bemærkning om klinisk ansvar:</p>
                    <p className="text-amber-700">
                      Denne AI-vurdering er et beslutningsstøtteværktøj. Endelige kliniske beslutninger skal træffes af kvalificerede anæstesilæger baseret på en fuldstændig patientvurdering.
                    </p>
                  </div>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        </Card>

        {/* AI Summary */}
        <Card className="p-4 mb-3 bg-white border-slate-200">
          <Collapsible open={openSections.summary} onOpenChange={() => toggleSection('summary')}>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between">
                <h2 className="text-sm text-slate-900 flex items-center gap-2">
                  <FileText className="size-4" />
                  AI-genereret opsummering
                </h2>
                <span className="text-xs text-slate-500">
                  {openSections.summary ? 'Skjul' : 'Vis'} detaljer
                </span>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3">
              <p className="text-sm text-slate-700 leading-relaxed">{patient.aiSummary}</p>
            </CollapsibleContent>
          </Collapsible>
        </Card>

        {/* Full Medical History */}
        <Card className="p-4 mb-3 bg-white border-slate-200">
          <Collapsible open={openSections.history} onOpenChange={() => toggleSection('history')}>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between">
                <h2 className="text-sm text-slate-900 flex items-center gap-2">
                  <FileText className="size-4" />
                  Fuld sygehistorie
                </h2>
                <span className="text-xs text-slate-500">
                  {openSections.history ? 'Skjul' : 'Vis'} detaljer
                </span>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3">
              <p className="text-sm text-slate-700 leading-relaxed mb-3">{patient.fullHistory}</p>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                  <p className="text-xs text-slate-500 mb-1.5">Allergier</p>
                  <div className="space-y-1">
                    {patient.allergies.map((allergy, idx) => (
                      <p key={idx} className="text-sm text-slate-700">{allergy}</p>
                    ))}
                  </div>
                </div>
              </div>
            </CollapsibleContent>
          </Collapsible>
        </Card>

        {/* Medications */}
        <Card className="p-4 mb-3 bg-white border-slate-200">
          <Collapsible open={openSections.medications} onOpenChange={() => toggleSection('medications')}>
            <CollapsibleTrigger className="w-full">
              <div className="flex items-center justify-between">
                <h2 className="text-sm text-slate-900 flex items-center gap-2">
                  <Pill className="size-4" />
                  Nuværende medicin
                </h2>
                <span className="text-xs text-slate-500">
                  {openSections.medications ? 'Skjul' : 'Vis'} detaljer
                </span>
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3">
              {patient.currentMedications.length > 0 ? (
                <div className="space-y-1.5">
                  {patient.currentMedications.map((med, idx) => (
                    <div key={idx} className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
                      <p className="text-sm text-slate-700">{med}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500">Ingen nuværende medicin</p>
              )}
            </CollapsibleContent>
          </Collapsible>
        </Card>

        {/* Clinician Notes */}
        {patient.notes && (
          <Card className="p-4 mb-3 bg-amber-50 border-amber-200">
            <div className="flex items-start gap-2.5">
              <AlertTriangle className="size-4 text-amber-700 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h2 className="text-sm text-amber-900 mb-1.5">Kliniske noter</h2>
                <p className="text-sm text-amber-800 leading-relaxed">{patient.notes}</p>
              </div>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
