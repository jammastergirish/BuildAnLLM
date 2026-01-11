"use client";

import { JobStatus } from "../lib/api";

type TrainingControlsProps = {
  job: JobStatus | null;
  isRunning: boolean;
  isPaused: boolean;
  isCreating?: boolean;
  disabled?: boolean;
  disabledReason?: string;
  progress: number;
  startLabel: string;
  error?: string | null;
  onPrimary: () => void;
  onStep: () => void;
  onReset?: () => void;
};

export default function TrainingControls({
  job,
  isRunning,
  isPaused,
  isCreating = false,
  disabled = false,
  disabledReason,
  progress,
  startLabel,
  error,
  onPrimary,
  onStep,
  onReset,
}: TrainingControlsProps) {
  const primaryLabel = isCreating
    ? "Initializing..."
    : !job
    ? startLabel
    : isRunning
    ? "Pause"
    : isPaused
    ? "Resume"
    : "Start New";

  const primaryDisabled = isCreating || disabled;
  const stepDisabled = disabled || !job || isRunning;
  const resetDisabled = disabled || !job;

  return (
    <div className="card">
      <div className="inline-row" style={{ marginBottom: 12 }}>
        <button className="primary" onClick={onPrimary} disabled={primaryDisabled}>
          {primaryLabel}
        </button>
        <button className="secondary" onClick={onStep} disabled={stepDisabled}>
          Step
        </button>
        {onReset && (
          <button
            className="secondary"
            onClick={onReset}
            disabled={resetDisabled}
            style={{ marginLeft: "auto" }}
            title="Stop training and reset all state"
          >
            Stop / Reset
          </button>
        )}
      </div>

      {disabled && disabledReason && <p className="badge demo-badge">{disabledReason}</p>}
      {job && (
        <>
          <div className="flex-between" style={{ marginBottom: 8 }}>
            <span className="badge">{job.status}</span>
            <span className="badge">
              {job.iter} / {job.max_iters}
            </span>
          </div>
          <div className="progress">
            <span style={{ width: `${progress * 100}%` }} />
          </div>
        </>
      )}

      {error && <p style={{ color: "#b42318" }}>{error}</p>}
    </div>
  );
}
