import React from 'react';
import { AlertCircle } from 'lucide-react';

interface EmptyStateProps {
  title?: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No records found',
  description = 'No simulation or trade records match the active criteria.',
  action,
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center bg-ivory rounded-xl border border-borderBase shadow-card min-h-[220px]">
      <div className="w-10 h-10 rounded-full bg-cream-200 flex items-center justify-center text-charcoal-muted mb-3 border border-borderBase">
        <AlertCircle className="w-5 h-5 text-olive-600" />
      </div>
      <h3 className="text-sm font-semibold text-charcoal">{title}</h3>
      <p className="text-xs text-charcoal-muted mt-1 max-w-sm mb-4">{description}</p>
      {action && (
        <button
          onClick={action.onClick}
          className="px-4 py-1.5 text-xs font-medium text-ivory bg-olive-500 hover:bg-olive-600 rounded-lg shadow-sm transition-colors"
        >
          {action.label}
        </button>
      )}
    </div>
  );
};
