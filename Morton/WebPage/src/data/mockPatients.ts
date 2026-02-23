export interface Patient {
  id: string;
  cprNumber: string;
  firstName: string;
  lastName: string;
  age: number;
  gender: 'M' | 'F';
  asaScore: number;
  asaExplanation: string;
  scheduledProcedure: string;
  procedureDate: Date;
  procedureTime: string;
  bmi: number;
  smokingStatus: 'Ikke-ryger' | 'Tidligere ryger' | 'Nuværende ryger';
  comorbidities: string[];
  riskIndicators: {
    name: string;
    value: string;
    severity: 'low' | 'medium' | 'high';
  }[];
  aiSummary: string;
  modelConfidence: number;
  aiExplanation: {
    factor: string;
    weight: number;
    impact: string;
  }[];
  fullHistory: string;
  allergies: string[];
  currentMedications: string[];
  notes: string;
}

export const mockPatients: Patient[] = [
  {
    id: '1',
    cprNumber: '010190-1234',
    firstName: 'Anna',
    lastName: 'Hansen',
    age: 35,
    gender: 'F',
    asaScore: 1,
    asaExplanation: 'Rask patient uden systemisk sygdom',
    scheduledProcedure: 'Laparoskopisk kolecystektomi',
    procedureDate: new Date('2026-01-24'),
    procedureTime: '08:00',
    bmi: 23.5,
    smokingStatus: 'Ikke-ryger',
    comorbidities: [],
    riskIndicators: [],
    aiSummary:
      'Rask 35-årig kvinde planlagt til elektiv laparoskopisk kolecystektomi. Ingen væsentlig sygehistorie. Normal BMI, ikke-ryger. Ingen kontraindikationer for anæstesi identificeret.',
    modelConfidence: 0.95,
    aiExplanation: [
      { factor: 'Alder', weight: 0.15, impact: 'Lav risiko - optimal aldersgruppe' },
      { factor: 'BMI', weight: 0.20, impact: 'Lav risiko - normal vægt' },
      { factor: 'Komorbiditeter', weight: 0.40, impact: 'Lav risiko - ingen kroniske tilstande' },
      { factor: 'Rygestatus', weight: 0.15, impact: 'Lav risiko - ikke-ryger' },
      { factor: 'Indgrebstype', weight: 0.10, impact: 'Lav risiko - rutineindgreb' }
    ],
    fullHistory:
      'Patienten præsenterer sig med symptomgivende galdesten. Nylig ultralyd bekræfter galdesten. Ingen tidligere operationer. Ingen familiehistorie med anæstesikomplikationer.',
    allergies: ['Ingen kendte'],
    currentMedications: [],
    notes: 'Patienten er bekymret for bedøvelse - beroliget og fået udleveret informationsmateriale.'
  },
  {
    id: '2',
    cprNumber: '150565-5678',
    firstName: 'Erik',
    lastName: 'Nielsen',
    age: 60,
    gender: 'M',
    asaScore: 3,
    asaExplanation: 'Svær systemisk sygdom, som ikke er livstruende',
    scheduledProcedure: 'Total hoftealloplastik',
    procedureDate: new Date('2026-01-24'),
    procedureTime: '08:30',
    bmi: 31.2,
    smokingStatus: 'Tidligere ryger',
    comorbidities: ['Type 2-diabetes', 'Hypertension', 'Artrose'],
    riskIndicators: [
      { name: 'Forhøjet HbA1c', value: '7.8%', severity: 'medium' },
      { name: 'Blodtryk', value: '145/92', severity: 'medium' },
      { name: 'BMI', value: '31.2', severity: 'medium' }
    ],
    aiSummary:
      '60-årig mand med kontrolleret type 2-diabetes og hypertension. BMI svarer til fedme grad I. Tidligere ryger (stoppede for 5 år siden). Planlagt til elektiv total hoftealloplastik. Øget perioperativ risiko pga. flere komorbiditeter, men tilstandene er tilstrækkeligt kontrollerede.',
    modelConfidence: 0.88,
    aiExplanation: [
      { factor: 'Alder', weight: 0.15, impact: 'Moderat risiko - højere alder' },
      { factor: 'BMI', weight: 0.20, impact: 'Moderat risiko - fedme grad I' },
      { factor: 'Komorbiditeter', weight: 0.40, impact: 'Moderat-høj risiko - diabetes og hypertension' },
      { factor: 'Rygestatus', weight: 0.15, impact: 'Lav risiko - stoppede for 5+ år siden' },
      { factor: 'Indgrebstype', weight: 0.10, impact: 'Moderat risiko - større ortopædkirurgi' }
    ],
    fullHistory:
      'Progredierende artrose i højre hofte med nedsat mobilitet. Diabetes diagnosticeret for 8 år siden, behandles med metformin. Hypertension kontrolleret med ACE-hæmmer. Ingen tidligere anæstesikomplikationer. Arbejdskapacitet begrænset af hoftesmerter.',
    allergies: ['Penicillin (udslæt)'],
    currentMedications: ['Metformin 1000mg x 2', 'Lisinopril 10mg dagligt', 'Aspirin 81mg dagligt'],
    notes:
      'Kardiologisk vurdering foreligger. Fortsæt aspirin perioperativt efter ortopædkirurgisk protokol. Pause metformin på operationsmorgenen.'
  },
  {
    id: '3',
    cprNumber: '230445-9101',
    firstName: 'Maria',
    lastName: 'Andersen',
    age: 81,
    gender: 'F',
    asaScore: 4,
    asaExplanation: 'Svær systemisk sygdom, som er en konstant trussel mod livet',
    scheduledProcedure: 'Akut operation for femurfraktur',
    procedureDate: new Date('2026-01-22'),
    procedureTime: '14:00',
    bmi: 21.8,
    smokingStatus: 'Ikke-ryger',
    comorbidities: ['Kongestiv hjertesvigt', 'Atrieflimren', 'Kronisk nyresygdom stadie 3'],
    riskIndicators: [
      { name: 'Ejektionsfraktion', value: '35%', severity: 'high' },
      { name: 'GFR', value: '42 mL/min', severity: 'high' },
      { name: 'INR', value: '2.3', severity: 'medium' }
    ],
    aiSummary:
      '81-årig kvinde med nyligt fald og femurfraktur, som kræver akut kirurgisk behandling. Betydelig hjertesygdom med nedsat ejektionsfraktion og kronisk atrieflimren. Kronisk nyresygdom komplicerer væske- og medicinhåndtering. Høj perioperativ risiko, men operationen er nødvendig.',
    modelConfidence: 0.82,
    aiExplanation: [
      { factor: 'Alder', weight: 0.15, impact: 'Høj risiko - høj alder og skrøbelighed' },
      { factor: 'BMI', weight: 0.20, impact: 'Lav risiko - normal vægt' },
      { factor: 'Komorbiditeter', weight: 0.40, impact: 'Høj risiko - svær hjerte- og nyresygdom' },
      { factor: 'Rygestatus', weight: 0.15, impact: 'Lav risiko - ikke-ryger' },
      { factor: 'Indgrebstype', weight: 0.10, impact: 'Høj risiko - akut operation' }
    ],
    fullHistory:
      'Fald i hjemmet i går aftes. Røntgen bekræfter disloceret collum femoris-fraktur. Kendt hjertesvigt med EF 35% (ekkokardiografi for 3 måneder siden). AF i warfarinbehandling. Stabil CKD stadie 3. Bor selv med hjemmehjælp.',
    allergies: ['Kontrastmiddel (anafylaksi)'],
    currentMedications: ['Warfarin 5mg dagligt', 'Furosemid 40mg dagligt', 'Metoprolol 50mg x 2', 'Digoxin 0.125mg dagligt'],
    notes:
      'HØJ RISIKO - Akut case. Warfarin reverseret med vitamin K og FFP. Kardiologi konsulteret. Plan for arteriel linje og forsigtig væskebehandling pga. hjertesvigt og CKD. Intensivseng reserveret postoperativt.'
  },
  {
    id: '4',
    cprNumber: '050295-2468',
    firstName: 'Lars',
    lastName: 'Petersen',
    age: 31,
    gender: 'M',
    asaScore: 2,
    asaExplanation: 'Let systemisk sygdom uden funktionsbegrænsning',
    scheduledProcedure: 'ACL-rekonstruktion',
    procedureDate: new Date('2026-01-23'),
    procedureTime: '09:00',
    bmi: 26.8,
    smokingStatus: 'Ikke-ryger',
    comorbidities: ['Kontrolleret astma'],
    riskIndicators: [{ name: 'Astmakontrol', value: 'Velkontrolleret', severity: 'low' }],
    aiSummary:
      'Sund 31-årig mandlig atlet med velkontrolleret astma. Pådraget ACL-ruptur under fodbold. Ingen andre medicinske problemer. Minimal anæstesirisiko med passende astmahåndtering.',
    modelConfidence: 0.93,
    aiExplanation: [
      { factor: 'Alder', weight: 0.15, impact: 'Lav risiko - ung og sund' },
      { factor: 'BMI', weight: 0.20, impact: 'Lav risiko - let overvægtig men atletisk' },
      { factor: 'Komorbiditeter', weight: 0.40, impact: 'Lav risiko - kun mild velkontrolleret astma' },
      { factor: 'Rygestatus', weight: 0.15, impact: 'Lav risiko - ikke-ryger' },
      { factor: 'Indgrebstype', weight: 0.10, impact: 'Lav risiko - rutine ortopædkirurgi' }
    ],
    fullHistory:
      'ACL-ruptur under konkurrencefodbold for 6 uger siden. MR bekræfter komplet ruptur. Astma siden barndommen, velkontrolleret med inhalationssteroid. Bruger sjældent behovsinhalator. Ingen indlæggelser pga. astma. Aktiv livsstil, træner regelmæssigt.',
    allergies: ['Ingen kendte'],
    currentMedications: ['Fluticason inhalator 110mcg x 2', 'Salbutamol inhalator pn'],
    notes: 'Anbefal at fortsætte inhalationssteroid. Medbring behovsinhalator på operationsdagen. Overvej at undgå histaminfrigørende midler.'
  },
  {
    id: '5',
    cprNumber: '110778-3579',
    firstName: 'Sofie',
    lastName: 'Jensen',
    age: 48,
    gender: 'F',
    asaScore: 2,
    asaExplanation: 'Let systemisk sygdom uden funktionsbegrænsning',
    scheduledProcedure: 'Thyroidektomi',
    procedureDate: new Date('2026-01-24'),
    procedureTime: '11:30',
    bmi: 28.5,
    smokingStatus: 'Nuværende ryger',
    comorbidities: ['Hypothyreose', 'GERD'],
    riskIndicators: [
      { name: 'Rygning', value: '15 pakkeår', severity: 'medium' },
      { name: 'BMI', value: '28.5', severity: 'low' }
    ],
    aiSummary:
      '48-årig kvinde med multinodøs struma, som kræver thyroidektomi. Nuværende ryger med moderat rygehistorik øger respiratorisk risiko. Hypothyreose er velkontrolleret. GERD behandles medicinsk. Samlet set moderat risikoprofil.',
    modelConfidence: 0.89,
    aiExplanation: [
      { factor: 'Alder', weight: 0.15, impact: 'Lav risiko - midaldrende' },
      { factor: 'BMI', weight: 0.20, impact: 'Lav risiko - let overvægtig' },
      { factor: 'Komorbiditeter', weight: 0.40, impact: 'Lav risiko - velkontrollerede tilstande' },
      { factor: 'Rygestatus', weight: 0.15, impact: 'Moderat risiko - aktiv ryger' },
      { factor: 'Indgrebstype', weight: 0.10, impact: 'Moderat risiko - luftvejskirurgi' }
    ],
    fullHistory:
      'Tiltagende multinodøs struma over de seneste 2 år. Let synkebesvær ved store piller. Ingen stemmeforandringer. TSH normal på levothyroxin. GERD-symptomer kontrolleret med PPI. Rådgivet om rygestop præoperativt, men fortsætter.',
    allergies: ['Codein (kvalme)'],
    currentMedications: ['Levothyroxin 100mcg dagligt', 'Omeprazol 20mg dagligt'],
    notes:
      'Luftvejsvurdering normal, ingen tegn på tracheakompression. Informeret om øget lungerisiko ved rygning. Plan aspiration-forebyggelse pga. GERD.'
  },
  {
    id: '6',
    cprNumber: '280352-7890',
    firstName: 'Peter',
    lastName: 'Christensen',
    age: 74,
    gender: 'M',
    asaScore: 3,
    asaExplanation: 'Svær systemisk sygdom, som ikke er livstruende',
    scheduledProcedure: 'Carotis endarterektomi',
    procedureDate: new Date('2026-01-25'),
    procedureTime: '08:30',
    bmi: 27.3,
    smokingStatus: 'Tidligere ryger',
    comorbidities: ['Koronar hjertesygdom', 'Hyperlipidæmi', 'Tidligere apopleksi'],
    riskIndicators: [
      { name: 'Carotisstenose', value: '85% venstre', severity: 'high' },
      { name: 'Tidligere AMI', value: 'for 3 år siden', severity: 'medium' },
      { name: 'Apopleksi/TIA', value: 'TIA for 6 måneder siden', severity: 'high' }
    ],
    aiSummary:
      '74-årig mand med betydelig kardiovaskulær sygdom, herunder tidligere AMI og TIA. Svær carotisstenose kræver endarterektomi. Kompleks hjerte- og cerebrovaskulær risikoprofil kræver nøje hæmodynamisk styring.',
    modelConfidence: 0.85,
    aiExplanation: [
      { factor: 'Alder', weight: 0.15, impact: 'Moderat-høj risiko - ældre' },
      { factor: 'BMI', weight: 0.20, impact: 'Lav risiko - normal vægt' },
      { factor: 'Komorbiditeter', weight: 0.40, impact: 'Høj risiko - betydelig kardiovaskulær sygdom' },
      { factor: 'Rygestatus', weight: 0.15, impact: 'Lav risiko - stoppet med at ryge' },
      { factor: 'Indgrebstype', weight: 0.10, impact: 'Høj risiko - carotiskirurgi' }
    ],
    fullHistory:
      'AMI for 3 år siden behandlet med stents i LAD og RCA. God funktionel bedring. TIA for 6 måneder siden førte til udredning med påvist svær venstresidig carotisstenose. Nylig belastningstest viser tilstrækkelig hjertereserve. I dobbelt trombocythæmmende behandling.',
    allergies: ['Ingen kendte'],
    currentMedications: ['Aspirin 81mg dagligt', 'Clopidogrel 75mg dagligt', 'Atorvastatin 80mg dagligt', 'Metoprolol 25mg x 2'],
    notes:
      'KOMPLEKS CASE - Kardiologi og neurologi har godkendt. Fortsæt aspirin, pause clopidogrel 5 dage præoperativt. Plan for vågen monitorering vs lokalanæstesi med sedation. Stram BT-kontrol er essentiel.'
  }
];

export function searchPatients(query: string): Patient[] {
  if (!query.trim()) return mockPatients;

  const lowerQuery = query.toLowerCase();
  return mockPatients.filter(patient =>
    patient.firstName.toLowerCase().includes(lowerQuery) ||
    patient.lastName.toLowerCase().includes(lowerQuery) ||
    patient.cprNumber.includes(query) ||
    `${patient.firstName} ${patient.lastName}`.toLowerCase().includes(lowerQuery)
  );
}

export function getPatientById(id: string): Patient | undefined {
  return mockPatients.find(p => p.id === id);
}

export function getScheduledPatients(date: Date): Patient[] {
  return mockPatients
    .filter(p => p.procedureDate.toDateString() === date.toDateString())
    .sort((a, b) => a.procedureTime.localeCompare(b.procedureTime));
}
