import { X, AlertTriangle, TrendingUp } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/app/components/ui/dialog';
import { Badge } from '@/app/components/ui/badge';
import { Button } from '@/app/components/ui/button';
import { Progress } from '@/app/components/ui/progress';
import type { Patient } from '@/data/mockPatients';

interface PatientSummaryModalProps {
  patient: Patient | null;
  isOpen: boolean;
  onClose: () => void;
  onViewDetails: (patient: Patient) => void;
}

export function PatientSummaryModal({ patient, isOpen, onClose, onViewDetails }: PatientSummaryModalProps) {
  if (!patient) return null;

  const getAsaBadgeColor = (score: number) => {
    if (score === 1) return 'bg-green-100 text-green-800 border-green-200';
    if (score === 2) return 'bg-blue-100 text-blue-800 border-blue-200';
    if (score === 3) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    if (score === 4) return 'bg-orange-100 text-orange-800 border-orange-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  const getSeverityColor = (severity: 'low' | 'medium' | 'high') => {
    if (severity === 'low') return 'text-green-700 bg-green-50 border-green-200';
    if (severity === 'medium') return 'text-yellow-700 bg-yellow-50 border-yellow-200';
    return 'text-red-700 bg-red-50 border-red-200';
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-xl">
            {patient.firstName} {patient.lastName}, {patient.age} år
          </DialogTitle>
          <DialogDescription className="text-sm text-slate-500">
            En detaljeret opsummering af patientens helbred og risikofaktorer.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {/* ASA Score */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-600">ASA-klassifikation</span>
              <Badge className={`${getAsaBadgeColor(patient.asaScore)} text-base px-3 py-1`}>
                ASA {patient.asaScore}
              </Badge>
            </div>
            <p className="text-sm text-slate-700">{patient.asaExplanation}</p>
          </div>

          {/* Key Risk Indicators */}
          {patient.riskIndicators.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <AlertTriangle className="size-4 text-amber-600" />
                <h3 className="text-slate-900">Vigtige risikomarkører</h3>
              </div>
              <div className="space-y-2">
                {patient.riskIndicators.map((indicator, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded-lg border ${getSeverityColor(indicator.severity)}`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm">{indicator.name}</span>
                      <span className="text-sm">{indicator.value}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Comorbidities */}
          {patient.comorbidities.length > 0 && (
            <div>
              <h3 className="text-slate-900 mb-2">Komorbiditeter</h3>
              <div className="flex flex-wrap gap-2">
                {patient.comorbidities.map((condition, idx) => (
                  <Badge key={idx} variant="outline" className="bg-slate-50 text-slate-700 border-slate-300">
                    {condition}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Clinical Summary */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="size-4 text-blue-600" />
              <h3 className="text-slate-900">Patientopsummering</h3>
            </div>
            <p className="text-sm text-slate-700 leading-relaxed bg-blue-50 p-4 rounded-lg border border-blue-100">
              {patient.aiSummary}
            </p>
          </div>

          {/* Model Confidence */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-600">AI-modelsikkerhed</span>
              <span className="text-sm text-slate-900">{Math.round(patient.modelConfidence * 100)}%</span>
            </div>
            <Progress value={patient.modelConfidence * 100} className="h-2" />
            <p className="text-xs text-slate-500 mt-1">
              Baseret på {patient.aiExplanation.length} kliniske faktorer
            </p>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-3 gap-4 pt-4 border-t border-slate-200">
            <div>
              <p className="text-xs text-slate-500 mb-1">BMI</p>
              <p className="text-slate-900">{patient.bmi}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Rygestatus</p>
              <p className="text-slate-900 text-sm">{patient.smokingStatus}</p>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Køn</p>
              <p className="text-slate-900">{patient.gender === 'M' ? 'Mand' : 'Kvinde'}</p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-4 border-t border-slate-200">
            <Button onClick={() => onViewDetails(patient)} className="flex-1">
              Se fulde detaljer
            </Button>
            <Button onClick={onClose} variant="outline" className="flex-1">
              Luk
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
