'use client';

import { AlertTriangle, RefreshCw, WifiOff, ServerCrash } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ErrorStateProps {
  type?: 'offline' | 'server' | 'partial';
  message?: string;
  onRetry?: () => void;
}

export default function ErrorState({ type = 'server', message, onRetry }: ErrorStateProps) {
  const isOffline = type === 'offline';
  const isPartial = type === 'partial';

  const Icon = isOffline ? WifiOff : isPartial ? AlertTriangle : ServerCrash;

  const defaultMessages = {
    offline: 'Connessione assente. Verifica la tua connessione internet.',
    server: 'Impossibile connettersi all\'API. Il server potrebbe essere offline.',
    partial: 'Si è verificato un errore nel caricamento dei dati.',
  };

  const displayMessage = message || defaultMessages[type];

  // Partial error - inline style
  if (isPartial) {
    return (
      <div className="flex items-center gap-3 p-4 rounded-lg border border-yellow-500/30 bg-yellow-500/10">
        <Icon className="w-5 h-5 text-yellow-400 flex-shrink-0" />
        <p className="text-sm text-yellow-400">{displayMessage}</p>
        {onRetry && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onRetry}
            className="ml-auto text-yellow-400 hover:text-yellow-300 hover:bg-yellow-500/10"
          >
            <RefreshCw className="w-4 h-4 mr-1" />
            Riprova
          </Button>
        )}
      </div>
    );
  }

  // Full page error
  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] text-center px-4">
      <div className="w-20 h-20 flex items-center justify-center bg-[#FF6B35]/10 rounded-full mb-6">
        <Icon className="w-10 h-10 text-[#FF6B35]" />
      </div>

      <h2 className="text-xl font-bold text-white mb-2">
        {isOffline ? 'Sei offline' : 'Errore di connessione'}
      </h2>

      <p className="text-gray-400 max-w-md mb-6">
        {displayMessage}
      </p>

      {onRetry && (
        <Button
          onClick={onRetry}
          className="bg-[#FF6B35] hover:bg-[#FF6B35]/90 text-white"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Riprova connessione
        </Button>
      )}

      <p className="text-sm text-gray-500 mt-8">
        Se il problema persiste, verifica che il backend sia in esecuzione su{' '}
        <code className="text-[#00A8E8]">localhost:8000</code>
      </p>
    </div>
  );
}
