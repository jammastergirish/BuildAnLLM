"use client";

import type { CSSProperties, ReactNode } from "react";

type RangeSliderProps = {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  onChange: (value: number) => void;
  showValue?: boolean;
  valueLabel?: string;
  className?: string;
  style?: CSSProperties;
};

export default function RangeSlider({
  label,
  value,
  min,
  max,
  step = 1,
  disabled = false,
  onChange,
  showValue = true,
  valueLabel,
  className,
  style,
}: RangeSliderProps) {
  const displayValue = valueLabel ?? `${value} / ${max}`;
  return (
    <div className={className ? `slider ${className}` : "slider"} style={style}>
      <label>{label}</label>
      <div className="slider-row">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          disabled={disabled}
        />
        {showValue && <span className="slider-value">{displayValue}</span>}
      </div>
    </div>
  );
}
