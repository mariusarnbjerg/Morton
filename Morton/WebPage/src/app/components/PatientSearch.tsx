import { useState } from 'react';
import { Search, User } from 'lucide-react';
import { Input } from '@/app/components/ui/input';
import { Card } from '@/app/components/ui/card';
import { Badge } from '@/app/components/ui/badge';
import { searchPatients, type Patient } from '@/data/mockPatients';
import { format } from 'date-fns';

interface PatientSearchProps {
  onPatientSelect: (patient: Patient) => void;
}

export function PatientSearch({ onPatientSelect }: PatientSearchProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<Patient[]>([]);
  const [showResults, setShowResults] = useState(false);

  const handleSearch = (value: string) => {
    setQuery(value);
    if (value.trim()) {
      const searchResults = searchPatients(value);
      setResults(searchResults);
      setShowResults(true);
    } else {
      setResults([]);
      setShowResults(false);
    }
  };

  const handleSelectPatient = (patient: Patient) => {
    setQuery('');
    setShowResults(false);
    onPatientSelect(patient);
  };

  const getAsaBadgeColor = (score: number) => {
    if (score === 1) return 'bg-green-100 text-green-800 border-green-200';
    if (score === 2) return 'bg-blue-100 text-blue-800 border-blue-200';
    if (score === 3) return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    if (score === 4) return 'bg-orange-100 text-orange-800 border-orange-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-start justify-center pt-32 px-4">
      <div className="w-full max-w-3xl">
        <div className="text-center mb-12">
          <h1 className="text-3xl text-slate-900 mb-2">Patientsøgning</h1>
          <p className="text-slate-600">Søg på CPR-nummer eller patientnavn</p>
        </div>

        <div className="relative">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 size-5 text-slate-400" />
            <Input
              type="text"
              placeholder="Indtast CPR-nummer eller navn..."
              value={query}
              onChange={(e) => handleSearch(e.target.value)}
              className="pl-12 h-14 text-base bg-white border-slate-200 focus:border-slate-400"
            />
          </div>

          {showResults && (
            <Card className="absolute top-full mt-2 w-full bg-white border-slate-200 shadow-lg max-h-[500px] overflow-auto z-10">
              {results.length > 0 ? (
                <div className="divide-y divide-slate-100">
                  {results.map((patient) => (
                    <button
                      key={patient.id}
                      onClick={() => handleSelectPatient(patient)}
                      className="w-full px-4 py-4 hover:bg-slate-50 transition-colors text-left flex items-start gap-4"
                    >
                      <div className="size-10 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0 mt-1">
                        <User className="size-5 text-slate-600" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-4 mb-1">
                          <div>
                            <h3 className="text-slate-900">
                              {patient.firstName} {patient.lastName}
                            </h3>
                            <p className="text-sm text-slate-500">{patient.cprNumber}</p>
                          </div>
                          <Badge className={`${getAsaBadgeColor(patient.asaScore)} flex-shrink-0`}>
                            ASA {patient.asaScore}
                          </Badge>
                        </div>
                        <div className="text-sm text-slate-600 mt-2">
                          <p>{patient.scheduledProcedure}</p>
                          <p className="text-slate-500 mt-1">
                            {format(patient.procedureDate, 'MMM d, yyyy')} kl. {patient.procedureTime}
                          </p>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="px-4 py-8 text-center text-slate-500">
                  Ingen patienter fundet, der matcher "{query}"
                </div>
              )}
            </Card>
          )}
        </div>

        <div className="mt-16 text-center text-sm text-slate-500">
          <p>Hurtig adgang: Skriv patientnavn eller CPR-nummer for at starte</p>
        </div>
      </div>
    </div>
  );
}
