import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  trend?: 'up' | 'down' | 'neutral';
  trendText?: string;
  icon?: React.ReactNode;
  badge?: string;
  tooltip?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subValue,
  trend,
  trendText,
  icon,
  badge,
  tooltip,
}) => {
  return (
    <div
      className="bg-ivory border border-borderBase rounded-xl p-5 shadow-card hover:shadow-card-hover transition-all duration-200 flex flex-col justify-between"
      title={tooltip}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-charcoal-muted">
          {label}
        </span>
        {icon && <div className="text-olive-500 opacity-80">{icon}</div>}
        {badge && (
          <span className="text-[11px] font-medium px-2 py-0.5 rounded bg-cream-200 text-olive-800 border border-borderBase">
            {badge}
          </span>
        )}
      </div>

      <div className="flex items-baseline justify-between">
        <div className="text-2xl font-bold tracking-tight text-charcoal num-tabular">
          {value}
        </div>
      </div>

      {(subValue || trendText) && (
        <div className="mt-3 pt-3 border-t border-cream-200 flex items-center justify-between text-xs text-charcoal-muted">
          <span>{subValue}</span>
          {trendText && (
            <span
              className={`font-medium ${
                trend === 'up'
                  ? 'text-gain'
                  : trend === 'down'
                  ? 'text-loss'
                  : 'text-charcoal-muted'
              }`}
            >
              {trendText}
            </span>
          )}
        </div>
      )}
    </div>
  );
};
