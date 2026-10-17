import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'olive' | 'sage' | 'gain' | 'loss' | 'neutral' | 'warning';
  size?: 'sm' | 'md';
}

export const Badge: React.FC<BadgeProps> = ({ children, variant = 'olive', size = 'sm' }) => {
  const variantStyles = {
    olive: 'bg-olive-100 text-olive-800 border-olive-200',
    sage: 'bg-cream-200 text-olive-700 border-olive-200',
    gain: 'bg-gain-bg text-gain border-gain-border',
    loss: 'bg-loss-bg text-loss border-loss-border',
    neutral: 'bg-cream-200 text-charcoal-muted border-borderBase',
    warning: 'bg-amber-50 text-amber-800 border-amber-200',
  };

  const sizeStyles = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
  };

  return (
    <span
      className={`inline-flex items-center font-medium rounded border ${variantStyles[variant]} ${sizeStyles[size]} tracking-tight transition-colors`}
    >
      {children}
    </span>
  );
};
