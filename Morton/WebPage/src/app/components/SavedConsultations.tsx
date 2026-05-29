import { useState, useEffect } from 'react';
import { User } from 'lucide-react';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import type { SummaryData } from '@/app/components/ConsultationSummary';

const API_BASE = 'http://localhost:8000/api/v1';

const ASA_BADGE_COLORS: Record<string, string> = {
  'ASA-I': 'bg-green-100 text-green-800 border-green-200',
  'ASA-II': 'bg-blue-100 text-blue-800 border-blue-200',
  'ASA-III': 'bg-yellow-100 text-yellow-800 border-yellow-200',
  'not_assessed': 'bg-slate-100 text-slate-600 border-slate-200',
};

interface SavedConsultation {
  filename: string;
  patient_name: string;
  patient_age: number | null;
  asa_classification: string;
  ml_asa_class: string | null;
  conversation_id: string | null;
}

interface SavedConsultationsProps {
  onConsultationSelect: (summary: SummaryData) => void;
}

export function SavedConsultations({ onConsultationSelect }: SavedConsultationsProps) {
  const [consultations, setConsultations] = useState<SavedConsultation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/consultations`)
      .then((res) => res.json())
      .then((data) => setConsultations(data.consultations ?? []))
      .catch(() => setConsultations([]))
      .finally(() => setLoading(false));
  }, []);

  const handleSelect = async (filename: string) => {
    try {
      const res = await fetch(`${API_BASE}/consultations/${filename}`);
      if (res.ok) {
        const summary: SummaryData = await res.json();
        onConsultationSelect(summary);
      }
    } catch {}
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-start justify-center pt-32 px-4">
      <div className="w-full max-w-3xl">
        <div className="text-center mb-12">
          <h1 className="text-3xl text-slate-900 mb-2">Saved consultations</h1>
          <p className="text-slate-600">{consultations.length} consultation(s)</p>
        </div>

        {loading ? (
          <p className="text-center text-slate-400 py-8">Loading...</p>
        ) : consultations.length > 0 ? (
          <Card className="bg-white border-slate-200 divide-y divide-slate-100">
            {consultations.map((c) => (
              <button
                key={c.filename}
                onClick={() => handleSelect(c.filename)}
                className="w-full px-4 py-4 hover:bg-slate-50 transition-colors text-left flex items-center gap-4"
              >
                <div className="size-10 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0">
                  <User className="size-5 text-slate-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-slate-900">{c.patient_name}</h3>
                  <p className="text-xs text-slate-400 font-mono">{c.conversation_id ?? c.filename}</p>
                  {c.patient_age && (
                    <p className="text-sm text-slate-500">{c.patient_age} years</p>
                  )}
                </div>
                <Badge className={`${ASA_BADGE_COLORS[c.asa_classification] ?? ASA_BADGE_COLORS['not_assessed']} border text-xs`}>
                  {c.asa_classification ?? 'N/A'}
                </Badge>
              </button>
            ))}
          </Card>
        ) : (
          <p className="text-center text-slate-400 py-8">No saved consultations yet</p>
        )}
      </div>
    </div>
  );
}