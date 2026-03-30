import { useState } from 'react';
import {
  User,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  MessageSquare,
  Shield,
  FileText,
  Cigarette,
  Heart,
  Pill,
  HelpCircle,
  Clock,
} from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';

// ── Types matching the API summary response ──────────────────────────────

interface PatientInfo {
  name: string;
  age: number;
  allergies: string;
  prior_anesthesia: 'yes' | 'no' | 'unknown';
  current_surgery: string;
  asa_classification: string;
  smoking_status: string;
  fasting_compliance: string;
}

interface QAItem {
  question_id: string;
  question: string;
  answer: string;
}

interface RedFlag {
  category: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  details?: string;
  description?: string;
}

interface PatientQuestion {
  question: string;
  answer: string;
}

export interface SummaryData {
  patient: PatientInfo;
  questionnaire_answers: QAItem[];
  patient_questions: PatientQuestion[];
  red_flags: RedFlag[];
  notes: string;
}

interface Props {
  summary: SummaryData;
  onDismiss: () => void;
}

// ── Helpers ──────────────────────────────────────────────────────────────

const ASA_LABELS: Record<string, { label: string; desc: string; color: string }> = {
  'ASA-I':        { label: 'ASA I',  desc: 'Healthy patient',                     color: 'bg-emerald-100 text-emerald-800 border-emerald-300' },
  'ASA-II':       { label: 'ASA II', desc: 'Mild systemic disease',               color: 'bg-sky-100 text-sky-800 border-sky-300' },
  'ASA-III':      { label: 'ASA III', desc: 'Severe systemic disease',             color: 'bg-amber-100 text-amber-800 border-amber-300' },
  'ASA-IV':       { label: 'ASA IV', desc: 'Severe life-threatening disease',      color: 'bg-orange-100 text-orange-800 border-orange-300' },
  'ASA-V':        { label: 'ASA V',  desc: 'Moribund patient',                    color: 'bg-red-100 text-red-800 border-red-300' },
  'not_assessed': { label: 'N/A',    desc: 'Insufficient information',            color: 'bg-slate-100 text-slate-600 border-slate-300' },
};

const SMOKING_LABELS: Record<string, string> = {
  never: 'Never smoked',
  former: 'Former smoker',
  current_light: 'Current smoker (light)',
  current_heavy: 'Current smoker (heavy)',
  unknown: 'Unknown',
};

const FASTING_LABELS: Record<string, string> = {
  compliant: 'Compliant',
  non_compliant: 'Non-compliant',
  unclear: 'Unclear',
  not_discussed: 'Not discussed',
};

const SEVERITY_STYLES: Record<string, string> = {
  low:      'bg-slate-50 border-slate-200 text-slate-700',
  medium:   'bg-amber-50 border-amber-200 text-amber-800',
  high:     'bg-orange-50 border-orange-300 text-orange-800',
  critical: 'bg-red-50 border-red-300 text-red-800',
};

const SEVERITY_BADGE: Record<string, string> = {
  low:      'bg-slate-200 text-slate-700',
  medium:   'bg-amber-200 text-amber-800',
  high:     'bg-orange-200 text-orange-900',
  critical: 'bg-red-200 text-red-900',
};

const CATEGORY_LABELS: Record<string, string> = {
  difficult_airway: 'Difficult airway',
  cardiac_risk: 'Cardiac risk',
  medication_interaction: 'Medication interaction',
  allergy_concern: 'Allergy concern',
  fasting_violation: 'Fasting violation',
  pregnancy_concern: 'Pregnancy concern',
  bleeding_risk: 'Bleeding risk',
  respiratory_risk: 'Respiratory risk',
  other: 'Other',
};

// ── Collapsible Section ──────────────────────────────────────────────────

function Section({
  title,
  icon,
  defaultOpen = false,
  badge,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  defaultOpen?: boolean;
  badge?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Card className="bg-white border-slate-200 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full px-5 py-4 flex items-center gap-3 hover:bg-slate-50 transition-colors"
      >
        <span className="text-slate-400">{icon}</span>
        <span className="text-sm font-medium text-slate-900 flex-1 text-left">{title}</span>
        {badge}
        {open ? (
          <ChevronDown className="size-4 text-slate-400" />
        ) : (
          <ChevronRight className="size-4 text-slate-400" />
        )}
      </button>
      {open && <div className="px-5 pb-5 border-t border-slate-100 pt-4">{children}</div>}
    </Card>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

export function ConsultationSummary({ summary, onDismiss }: Props) {
  const { patient, questionnaire_answers, patient_questions, red_flags, notes } = summary;

  const asa = ASA_LABELS[patient.asa_classification] ?? ASA_LABELS['not_assessed'];

  return (
    <div className="min-h-full bg-slate-50 py-8 px-4">
      <div className="max-w-4xl mx-auto space-y-4">

        {/* ── Header bar ─────────────────────────────────────────── */}
        <div className="flex items-start justify-between gap-4 mb-2">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Clock className="size-4 text-slate-400" />
              <span className="text-xs text-slate-500 uppercase tracking-wide">
                Latest consultation
              </span>
            </div>
            <h2 className="text-2xl text-slate-900">{patient.name || 'Unknown patient'}</h2>
            <p className="text-sm text-slate-500 mt-0.5">
              {patient.age ? `${patient.age} years` : ''}
              {patient.current_surgery ? ` · ${patient.current_surgery}` : ''}
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={onDismiss} className="text-slate-500 mt-1">
            Dismiss
          </Button>
        </div>

        {/* ── Quick-glance cards ──────────────────────────────────── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Card className="p-3 bg-white border-slate-200">
            <p className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">ASA</p>
            <Badge className={`${asa.color} border text-xs`}>{asa.label}</Badge>
            <p className="text-[11px] text-slate-500 mt-1">{asa.desc}</p>
          </Card>

          <Card className="p-3 bg-white border-slate-200">
            <p className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">Allergies</p>
            <p className="text-sm text-slate-800 leading-snug">
              {patient.allergies || 'None'}
            </p>
          </Card>

          <Card className="p-3 bg-white border-slate-200">
            <p className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">Smoking</p>
            <div className="flex items-center gap-1.5">
              <Cigarette className="size-3.5 text-slate-400" />
              <p className="text-sm text-slate-800">
                {SMOKING_LABELS[patient.smoking_status] ?? patient.smoking_status}
              </p>
            </div>
          </Card>

          <Card className="p-3 bg-white border-slate-200">
            <p className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">Prior anaesthesia</p>
            <p className="text-sm text-slate-800 capitalize">{patient.prior_anesthesia}</p>
          </Card>
        </div>

        {/* ── Red flags ──────────────────────────────────────────── */}
        {red_flags.length > 0 && (
          <Section
            title="Red flags"
            icon={<AlertTriangle className="size-4" />}
            defaultOpen={true}
            badge={
              <Badge className="bg-red-100 text-red-700 border border-red-200 text-xs">
                {red_flags.length}
              </Badge>
            }
          >
            <div className="space-y-2">
              {red_flags.map((flag, i) => (
                <div
                  key={i}
                  className={`rounded-lg border p-3 ${SEVERITY_STYLES[flag.severity] ?? SEVERITY_STYLES.low}`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Badge className={`${SEVERITY_BADGE[flag.severity] ?? ''} text-[10px] uppercase px-1.5 py-0`}>
                      {flag.severity}
                    </Badge>
                    <span className="text-xs font-medium">
                      {CATEGORY_LABELS[flag.category] ?? flag.category}
                    </span>
                  </div>
                  <p className="text-sm leading-relaxed">{flag.details || flag.description || ''}</p>
                </div>
              ))}
            </div>
          </Section>
        )}

        {/* ── Clinical notes ─────────────────────────────────────── */}
        {notes && (
          <Section
            title="Clinical notes"
            icon={<FileText className="size-4" />}
            defaultOpen={true}
          >
            <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-line">{notes}</p>
          </Section>
        )}

        {/* ── Full Q & A transcript ──────────────────────────────── */}
        <Section
          title="Full questionnaire responses"
          icon={<MessageSquare className="size-4" />}
          defaultOpen={false}
          badge={
            <span className="text-xs text-slate-400">{questionnaire_answers.length} items</span>
          }
        >
          <div className="divide-y divide-slate-100">
            {questionnaire_answers.map((qa) => (
              <div key={qa.question_id} className="py-3 first:pt-0 last:pb-0">
                <p className="text-xs font-medium text-slate-500 mb-1 flex items-center gap-1.5">
                  <span className="inline-block bg-slate-100 text-slate-500 rounded px-1.5 py-0.5 text-[10px] font-mono uppercase">
                    {qa.question_id}
                  </span>
                  {qa.question}
                </p>
                <p className="text-sm text-slate-800 leading-relaxed pl-1">{qa.answer}</p>
              </div>
            ))}
          </div>
        </Section>

        {/* ── Patient-initiated questions ─────────────────────────── */}
        {patient_questions.length > 0 && (
          <Section
            title="Questions asked by patient"
            icon={<HelpCircle className="size-4" />}
            defaultOpen={false}
            badge={
              <span className="text-xs text-slate-400">{patient_questions.length}</span>
            }
          >
            <div className="space-y-3">
              {patient_questions.map((pq, i) => (
                <div key={i} className="bg-slate-50 rounded-lg p-3 border border-slate-100">
                  <p className="text-sm font-medium text-slate-800 mb-1">"{pq.question}"</p>
                  <p className="text-sm text-slate-600 leading-relaxed">{pq.answer}</p>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>
    </div>
  );
}