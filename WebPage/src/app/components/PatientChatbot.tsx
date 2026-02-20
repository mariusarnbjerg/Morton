import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Lightbulb } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Input } from '@/app/components/ui/input';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const suggestedQuestions = [
  "Hvad skal jeg gøre for at forberede mig til min operation?",
  "Hvilke risici er der ved fuld bedøvelse?",
  "Hvor lang tid tager det at vågne efter bedøvelse?",
  "Må jeg spise eller drikke før mit indgreb?",
  "Hvilke bivirkninger kan jeg forvente?",
  "Vil jeg mærke smerte under indgrebet?"
];

const botResponses: { [key: string]: string } = {
  "prepare": "For at forberede dig til din operation:\n\n• Spis eller drik ikke noget i 8 timer før dit indgreb (medmindre du får andre instruktioner)\n• Tag din faste medicin med en lille slurk vand (tjek med din læge først)\n• Fjern alle smykker, kontaktlinser og tandproteser\n• Brug behageligt, løstsiddende tøj\n• Sørg for, at nogen kan køre dig hjem\n• Følg eventuelle specifikke instruktioner fra dit operationsteam\n\nDin anæstesilæge vil gennemgå det hele med dig inden indgrebet.",
  "risks": "Fuld bedøvelse er meget sikker for de fleste patienter. Almindelige, midlertidige effekter inkluderer:\n\n• Kvalme eller opkast (kan behandles med medicin)\n• Ondt i halsen fra vejrtrækningsrør\n• Træthed og forvirring i nogle timer\n• Let hæshed\n\nAlvorlige komplikationer er sjældne, men kan inkludere:\n• Allergiske reaktioner\n• Vejrtrækningsbesvær\n• Ændringer i blodtryk eller hjerterytme\n\nDin anæstesilæge overvåger dig nøje under hele indgrebet og justerer medicinen efter behov for at holde dig sikker. De vil tale med dig om dine specifikke risikofaktorer under din præoperative vurdering.",
  "wake": "De fleste patienter begynder at vågne inden for 5-15 minutter efter bedøvelsen stoppes, men fuld restitution tager længere tid:\n\n• Første opvågning: 5-15 minutter\n• Klar nok til at svare: 30-60 minutter\n• Føler sig mere normal: 4-6 timer\n• Fuld restitution: 24 timer\n\nDen præcise tid varierer afhængigt af:\n• Type og varighed af operation\n• Hvilken medicin der anvendes\n• Din alder og generelle helbred\n• Din individuelle reaktion på bedøvelse\n\nDu vil blive overvåget tæt i opvågningsafsnittet, indtil du er stabil og har det godt.",
  "eat": "**Fastevejledning** (typisk):\n\n**Spis eller drik IKKE:**\n• 8 timer før operation ved fast føde\n• 6 timer før ved lette måltider\n• 2 timer før ved klare væsker (vand, sort kaffe, klar juice)\n\n**Hvorfor er dette vigtigt?**\nBedøvelse afslapper dine muskler, inklusive dem der forhindrer maveindhold i at komme op. En tom mave mindsker risikoen for aspiration (maveindhold der kommer ned i lungerne), hvilket kan være farligt.\n\n**Vigtigt:**\nFølg altid de specifikke instruktioner fra dit operationsteam, da retningslinjer kan variere afhængigt af dit indgreb og din sygehistorie. Hvis du ved et uheld spiser eller drikker, så informér din anæstesilæge med det samme.",
  "effects": "Almindelige bivirkninger efter bedøvelse inkluderer:\n\n**De første timer:**\n• Træthed og døsighed\n• Kvalme eller opkast\n• Ondt i halsen\n• Tør mund\n• Følelse af kulde eller rysten\n• Forvirring eller hukommelseshuller\n\n**De næste 24-48 timer:**\n• Let svimmelhed\n• Hovedpine\n• Muskelømhed\n• Svært ved at koncentrere sig\n\n**De fleste bivirkninger:**\n• Er midlertidige og milde\n• Forsvinder inden for 24-48 timer\n• Kan håndteres med medicin\n• Overvåges tæt af sundhedspersonalet\n\nKontakt din sundhedsudbyder, hvis du oplever alvorlige symptomer eller noget bekymrende efter du er kommet hjem.",
  "pain": "**Under indgrebet:**\nNej, du vil IKKE mærke smerte under operationen. Bedøvelse sikrer, at du er helt bevidstløs og smertefri. Du vil ikke mærke, se, høre eller huske noget.\n\n**Hvordan virker det:**\n• Fuld bedøvelse bringer dig i en dyb søvn\n• Smertestillende medicin gives løbende\n• Dine vitale værdier overvåges konstant\n• Anæstesilægen justerer medicinen efter behov\n\n**Efter indgrebet:**\nDu kan opleve smerte, når bedøvelsen aftager, men:\n• Smertestillende gives, før du vågner\n• Dit smerteniveau vurderes regelmæssigt\n• Du kan få ekstra medicin efter behov\n• Smertelindring er en prioritet for dit behandlingsteam\n\nTøv aldrig med at sige til sygeplejerskerne, hvis du har smerter - der findes mange effektive muligheder for at holde dig komfortabel.",
  "default": "Jeg forstår, at du har spørgsmål om din bedøvelse og dit indgreb. Jeg kan give generel information, men jeg kan ikke erstatte den personlige behandling og rådgivning fra dit sundhedsteam.\n\nFor specifikke spørgsmål om din situation, kan du:\n• Tale med din anæstesilæge ved din præoperative samtale\n• Ringe til dit operationsteams afdeling\n• Stille spørgsmål på dagen for dit indgreb\n\nEr der et generelt emne om bedøvelse, jeg kan hjælpe med at forklare? Du kan også vælge blandt de foreslåede spørgsmål nedenfor."
};

function getBotResponse(userMessage: string): string {
  const lowerMessage = userMessage.toLowerCase();

  if (lowerMessage.includes('prepare') || lowerMessage.includes('preparation')) {
    return botResponses.prepare;
  } else if (lowerMessage.includes('risk') || lowerMessage.includes('danger') || lowerMessage.includes('safe')) {
    return botResponses.risks;
  } else if (lowerMessage.includes('wake') || lowerMessage.includes('recovery') || lowerMessage.includes('conscious')) {
    return botResponses.wake;
  } else if (lowerMessage.includes('eat') || lowerMessage.includes('drink') || lowerMessage.includes('fast') || lowerMessage.includes('food')) {
    return botResponses.eat;
  } else if (lowerMessage.includes('side effect') || lowerMessage.includes('after') || lowerMessage.includes('expect')) {
    return botResponses.effects;
  } else if (lowerMessage.includes('pain') || lowerMessage.includes('hurt') || lowerMessage.includes('feel')) {
    return botResponses.pain;
  }

  return botResponses.default;
}

export function PatientChatbot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hej! Jeg er her for at hjælpe med at besvare dine spørgsmål om bedøvelse og dit kommende indgreb. Jeg kan give generel information, så du kan føle dig bedre forberedt og informeret.\n\nHusk, at denne chat kun er til undervisnings- og informationsformål og ikke erstatter medicinsk rådgivning fra dit sundhedsteam. Ved specifikke bekymringer om din situation bør du altid tale med din anæstesilæge eller operationsteam.\n\nHvordan kan jeg hjælpe dig i dag?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = (text: string) => {
    if (!text.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: text.trim(),
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    setTimeout(() => {
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: getBotResponse(text),
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    }, 800);
  };

  const handleSuggestedQuestion = (question: string) => {
    handleSendMessage(question);
  };

  return (
    <div className="h-full bg-gradient-to-b from-blue-50 to-white flex flex-col">
      {/* Header - Kompakt */}
      <div className="bg-white border-b border-slate-200 shadow-sm flex-shrink-0">
        <div className="max-w-4xl mx-auto px-3 py-1.5">
          <div className="flex items-center gap-2">
            <div className="size-7 rounded-full bg-blue-100 flex items-center justify-center">
              <Bot className="size-3.5 text-blue-600" />
            </div>

            <div className="min-w-0">
              <h1 className="text-sm font-medium text-slate-900 leading-tight">
                Informationsassistent om bedøvelse
              </h1>
              <p className="text-[10px] text-slate-600 leading-tight">
                Din guide til at forstå bedøvelse
              </p>
            </div>
          </div>

          {/* Vigtig bar - kompakt */}
          <div className="mt-1.5 bg-blue-50 border border-blue-200 rounded-md px-2 py-0.5 text-[10px] text-blue-900 leading-snug">
            <span className="font-semibold">Vigtigt:</span>{" "}
            Denne chatbot giver kun generel uddannelsesinformation. Den erstatter ikke rådgivning fra din anæstesilæge eller dit sundhedsteam.
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.sender === 'bot' && (
                <div className="size-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                  <Bot className="size-4 text-blue-600" />
                </div>
              )}

              <div
                className={`max-w-[70%] rounded-lg px-3 py-2 ${
                  message.sender === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white border border-slate-200 text-slate-700'
                }`}
              >
                <p className="whitespace-pre-line leading-relaxed text-sm">{message.text}</p>
              </div>

              {message.sender === 'user' && (
                <div className="size-8 rounded-full bg-slate-200 flex items-center justify-center flex-shrink-0 mt-1">
                  <User className="size-4 text-slate-600" />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Foreslåede spørgsmål */}
      {messages.length === 1 && (
        <div className="bg-slate-50 border-t border-slate-200 p-3 flex-shrink-0">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="size-3.5 text-amber-600" />
              <h3 className="text-xs text-slate-700">Foreslåede spørgsmål:</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {suggestedQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSuggestedQuestion(question)}
                  className="text-left p-2 bg-white border border-slate-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-xs text-slate-700"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Input */}
      <div className="bg-white border-t border-slate-200 p-3 flex-shrink-0">
        <div className="max-w-4xl mx-auto">
          <div className="flex gap-3">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleSendMessage(inputValue);
                }
              }}
              placeholder="Skriv dit spørgsmål om bedøvelse..."
              className="flex-1 bg-slate-50 border-slate-300"
            />
            <Button
              onClick={() => handleSendMessage(inputValue)}
              disabled={!inputValue.trim()}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <Send className="size-4" />
            </Button>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Tryk Enter for at sende • Dette er kun til generel information
          </p>
        </div>
      </div>
    </div>
  );
}
