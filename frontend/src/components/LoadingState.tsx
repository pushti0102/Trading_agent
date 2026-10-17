import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
  subMessage?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading simulation data...',
  subMessage = 'Fetching verified historical observations from backend engine',
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center bg-ivory rounded-xl border border-borderBase shadow-card min-h-[260px]">
      <Loader2 className="w-8 h-8 text-olive-500 animate-spin mb-3" />
      <p className="text-sm font-semibold text-charcoal">{message}</p>
      {subMessage && (
        <p className="text-xs text-charcoal-muted mt-1 max-w-sm">{subMessage}</p>
      )}
    </div>
  );
};
