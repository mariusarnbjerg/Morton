import { useState } from 'react';
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  MessageSquare,
  FileText,
  Cigarette,
  Heart,
  Pill,
  Clock,
  Stethoscope,
  Brain,
  Wind,
  Droplet,
  Activity,
} from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';

// ── Types matching the API summary response ──────────────────────────────

interface PatientInfo {
  name: string;
  age: number;
}

interface AnesthesiaHistory {
  prior_anesthesia: 'yes' | 'no' | 'unknown';
  prior_anesthesia_issues: string;
  family_reaction: 'yes' | 'no' | 'unknown';
  family_reaction_type: string;
}

interface Airway {
  difficult_airway: 'yes' | 'no' | 'unknown';
  sleep_apnea: string;
  dental: string;
}

interface AllergiesAndMedications {
  allergies: string;
  prescription_medications: string;
  blood_thinners: string;
  supplements: string;
}

interface Cardiopulmonary {
  cardiac_history: string;
  respiratory_history: string;
  exertional_dyspnea: 'yes' | 'no' | 'unknown';
}

interface GIAndRecentHealth {
  acid_reflux: string;
  recent_health_changes: string;
}

interface Lifestyle {
  smoking_status: 'never' | 'former' | 'current_light' | 'current_heavy' | 'unknown';
  smoking_details: string;
  alcohol: string;
  recreational_drugs: string;
}

interface Psychological {
  anxiety_level: string;
  additional_history: string;
  additional_questions: string;
}

interface RedFlag {
  category: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  details?: string;
  description?: string;
}

interface QAItem {
  question_id: string;
  question: string;
  answer: string;
}

interface TranscriptMessage {
  role: 'patient' | 'assistant';
  content: string;
  timestamp: number;
}

export interface SummaryData {
  // LLM-generated structured interpretation
  patient: PatientInfo;
  anesthesia_history: AnesthesiaHistory;
  airway: Airway;
  allergies_and_medications: AllergiesAndMedications;
  cardiopulmonary: Cardiopulmonary;
  gi_and_recent_health: GIAndRecentHealth;
  lifestyle: Lifestyle;
  psychological: Psychological;
  red_flags: RedFlag[];
  asa_classification: string;
  notes: string;
  // Deterministically populated by the orchestrator
  questionnaire_answers: QAItem[];
  raw_transcript: TranscriptMessage[];
}

interface Props {
  summary: SummaryData;
  onDismiss: () => void;
}

// ── Helpers ──────────────────────────────────────────────────────────────

const ASA_LABELS: Record<string, { label: string; desc: string; color: string }> = {
  'ASA-I':        { label: 'ASA I',  desc: 'Healthy patient',                     color: 'bg-emerald-100 text-emerald-800 border-emerald-300' },
  'ASA-II':       { label: 'ASA II', desc: 'Mild systemic disease',               color: 'bg-sky-100 text-sky-800 border-sky-300' },
  'ASA-III':      { label: 'ASA III', desc: 'Severe systemic disease',            color: 'bg-amber-100 text-amber-800 border-amber-300' },
  'ASA-IV':       { label: 'ASA IV', desc: 'Severe life-threatening disease',     color: 'bg-orange-100 text-orange-800 border-orange-300' },
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

const YESNO_LABELS: Record<string, string> = {
  yes: 'Yes',
  no: 'No',
  unknown: 'Unknown',
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
  pregnancy_concern: 'Pregnancy concern',
  bleeding_risk: 'Bleeding risk',
  respiratory_risk: 'Respiratory risk',
  family_anesthesia_reaction: 'Family anesthesia reaction',
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

// ── Field row helper ─────────────────────────────────────────────────────

function Field({ label, value }: { label: string; value: string | undefined }) {
  return (
    <div className="py-2 border-b border-slate-100 last:border-b-0">
      <p className="text-[11px] uppercase tracking-wide text-slate-400 mb-0.5">{label}</p>
      <p className="text-sm text-slate-800 leading-snug">{value || '—'}</p>
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

export function ConsultationSummary({ summary, onDismiss }: Props) {
  const {
    patient,
    anesthesia_history,
    airway,
    allergies_and_medications,
    cardiopulmonary,
    gi_and_recent_health,
    lifestyle,
    psychological,
    red_flags,
    asa_classification,
    notes,
    questionnaire_answers,
    raw_transcript,
  } = summary;

  const asa = ASA_LABELS[asa_classification] ?? ASA_LABELS['not_assessed'];

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
              {allergies_and_medications.allergies || 'None'}
            </p>
          </Card>

          <Card className="p-3 bg-white border-slate-200">
            <p className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">Smoking</p>
            <div className="flex items-center gap-1.5">
              <Cigarette className="size-3.5 text-slate-400" />
              <p className="text-sm text-slate-800">
                {SMOKING_LABELS[lifestyle.smoking_status] ?? lifestyle.smoking_status}
              </p>
            </div>
          </Card>

          <Card className="p-3 bg-white border-slate-200">
            <p className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">Prior anaesthesia</p>
            <p className="text-sm text-slate-800 capitalize">{anesthesia_history.prior_anesthesia}</p>
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

        {/* ── Anesthesia history ─────────────────────────────────── */}
        <Section
          title="Anesthesia history"
          icon={<Stethoscope className="size-4" />}
          defaultOpen={false}
        >
          <Field label="Prior anesthesia" value={YESNO_LABELS[anesthesia_history.prior_anesthesia]} />
          <Field label="Prior anesthesia issues" value={anesthesia_history.prior_anesthesia_issues} />
          <Field label="Family reaction" value={YESNO_LABELS[anesthesia_history.family_reaction]} />
          <Field label="Family reaction type" value={anesthesia_history.family_reaction_type} />
        </Section>

        {/* ── Airway ─────────────────────────────────────────────── */}
        <Section
          title="Airway"
          icon={<Wind className="size-4" />}
          defaultOpen={false}
        >
          <Field label="Difficult airway" value={YESNO_LABELS[airway.difficult_airway]} />
          <Field label="Sleep apnea / CPAP" value={airway.sleep_apnea} />
          <Field label="Dental" value={airway.dental} />
        </Section>

        {/* ── Allergies & medications ────────────────────────────── */}
        <Section
          title="Allergies & medications"
          icon={<Pill className="size-4" />}
          defaultOpen={false}
        >
          <Field label="Allergies" value={allergies_and_medications.allergies} />
          <Field label="Prescription medications" value={allergies_and_medications.prescription_medications} />
          <Field label="Blood thinners" value={allergies_and_medications.blood_thinners} />
          <Field label="Supplements & OTC" value={allergies_and_medications.supplements} />
        </Section>

        {/* ── Cardiopulmonary ────────────────────────────────────── */}
        <Section
          title="Cardiopulmonary"
          icon={<Heart className="size-4" />}
          defaultOpen={false}
        >
          <Field label="Cardiac history" value={cardiopulmonary.cardiac_history} />
          <Field label="Respiratory history" value={cardiopulmonary.respiratory_history} />
          <Field label="Shortness of breath on exertion" value={YESNO_LABELS[cardiopulmonary.exertional_dyspnea]} />
        </Section>

        {/* ── GI & recent health ─────────────────────────────────── */}
        <Section
          title="Recent health"
          icon={<Droplet className="size-4" />}
          defaultOpen={false}
        >
          <Field label="Acid reflux" value={gi_and_recent_health.acid_reflux} />
          <Field label="Recent health changes" value={gi_and_recent_health.recent_health_changes} />
        </Section>

        {/* ── Lifestyle ──────────────────────────────────────────── */}
        <Section
          title="Lifestyle"
          icon={<Activity className="size-4" />}
          defaultOpen={false}
        >
          <Field label="Smoking status" value={SMOKING_LABELS[lifestyle.smoking_status]} />
          <Field label="Smoking details" value={lifestyle.smoking_details} />
          <Field label="Alcohol" value={lifestyle.alcohol} />
          <Field label="Recreational drugs" value={lifestyle.recreational_drugs} />
        </Section>

        {/* ── Psychological ──────────────────────────────────────── */}
        <Section
          title="Psychological & additional"
          icon={<Brain className="size-4" />}
          defaultOpen={false}
        >
          <Field label="Anxiety level" value={psychological.anxiety_level} />
          <Field label="Additional history" value={psychological.additional_history} />
          <Field label="Additional questions" value={psychological.additional_questions} />
        </Section>

        {/* ── Full Q & A transcript (deterministic) ──────────────── */}
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

        {/* ── Full raw transcript (deterministic) ───────────────── */}
        {raw_transcript.length > 0 && (
          <Section
            title="Full conversation transcript"
            icon={<MessageSquare className="size-4" />}
            defaultOpen={false}
            badge={
              <span className="text-xs text-slate-400">{raw_transcript.length} messages</span>
            }
          >
            <div className="space-y-2">
              {raw_transcript.map((msg, i) => (
                <div
                  key={i}
                  className={`rounded-lg p-3 border ${
                    msg.role === 'patient'
                      ? 'bg-blue-50 border-blue-100'
                      : 'bg-slate-50 border-slate-100'
                  }`}
                >
                  <p className="text-[10px] uppercase tracking-wide text-slate-500 mb-1 font-medium">
                    {msg.role === 'patient' ? 'Patient' : 'Assistant'}
                  </p>
                  <p className="text-sm text-slate-800 leading-relaxed whitespace-pre-line">
                    {msg.content}
                  </p>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>
    </div>
  );
}