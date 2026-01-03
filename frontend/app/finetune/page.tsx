"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import CodePanel from "../../components/CodePanel";
import Heatmap from "../../components/Heatmap";
import LineChart from "../../components/LineChart";
import LogBox from "../../components/LogBox";
import RangeSlider from "../../components/RangeSlider";
import SideNav from "../../components/SideNav";
import StatCard from "../../components/StatCard";
import TokenSegments from "../../components/TokenSegments";
import TrainingControls from "../../components/TrainingControls";
import ModelConfigSummary from "../../components/ModelConfigSummary";
import { fetchJson, makeFormData, Checkpoint, CodeSnippet, JobStatus } from "../../lib/api";
import { useSse } from "../../lib/useSse";
import { useScrollSpy } from "../../lib/useScrollSpy";
import MarkdownBlock from "../../components/MarkdownBlock";
import { finetuneEquations, loraEquations } from "../../lib/equations";
import { formatDuration, formatTimestamp, formatCheckpointTimestamp } from "../../lib/time";
import { useDemoMode } from "../../lib/demo";

type MetricsPayload = {
  loss?: number;
  running_loss?: number;
  grad_norm?: number;
  aux_loss?: number;
  iter?: number;
  max_iters?: number;
  elapsed_time?: number;
  iter_per_sec?: number;
  eta_seconds?: number;
};

const finetuneSections = [
  { id: "checkpoint", label: "Checkpoint" },
  { id: "method", label: "Method" },
  { id: "training-data", label: "Training data" },
  { id: "hyperparameters", label: "Hyperparameters" },
  { id: "understand", label: "Understand" },
  { id: "train", label: "Train" },
  { id: "metrics", label: "Metrics" },
  { id: "inspect-batch", label: "Inspect batch" },
  { id: "eval-history", label: "Eval history" },
  { id: "logs", label: "Logs" },
];

export default function FinetunePage() {
  const isDemo = useDemoMode();
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [checkpointConfig, setCheckpointConfig] = useState<Record<string, number | string | boolean> | null>(null);
  const [method, setMethod] = useState<"full" | "lora">("full");
  const [loraConfig, setLoraConfig] = useState({
    lora_rank: 8,
    lora_alpha: 8,
    lora_dropout: 0,
    lora_target_modules: "all",
  });
  const [dataFile, setDataFile] = useState<File | null>(null);
  const [csvPreview, setCsvPreview] = useState<string[][]>([]);
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [maxLength, setMaxLength] = useState(512);
  const [autoStart, setAutoStart] = useState(true);
  const [trainingParams, setTrainingParams] = useState({
    batch_size: 4,
    epochs: 3,
    max_steps_per_epoch: 200,
    learning_rate: 0.00001,
    weight_decay: 0.01,
    eval_interval: 50,
    eval_iters: 50,
    save_interval: 500,
  });
  const [job, setJob] = useState<JobStatus | null>(null);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<MetricsPayload[]>([]);
  const [evalHistory, setEvalHistory] = useState<Record<string, number>[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [snippets, setSnippets] = useState<CodeSnippet[]>([]);
  const [snippetsLoading, setSnippetsLoading] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inspectSample, setInspectSample] = useState(0);
  const [inspectData, setInspectData] = useState<{
    token_labels: string[];
    prompt_tokens: string[];
    response_tokens: string[];
  } | null>(null);
  const [attention, setAttention] = useState<number[][]>([]);
  const [attnLayer, setAttnLayer] = useState(0);
  const [attnHead, setAttnHead] = useState(0);
  const inspectThrottleRef = useRef(0);
  const inspectMaxTokens = 128;

  useEffect(() => {
    fetchJson<{ checkpoints: Checkpoint[] }>("/api/checkpoints")
      .then((data) => setCheckpoints(data.checkpoints))
      .catch((err) => setError((err as Error).message));
  }, []);

  useEffect(() => {
    if (!selectedCheckpoint) {
      setCheckpointConfig(null);
      return;
    }
    fetchJson<{ cfg: Record<string, number | string | boolean> }>(
      `/api/checkpoints/${encodeURIComponent(selectedCheckpoint)}`
    )
      .then((data) => setCheckpointConfig(data.cfg))
      .catch((err) => setError((err as Error).message));
  }, [selectedCheckpoint]);

  useEffect(() => {
    if (!dataFile) {
      setCsvPreview([]);
      setCsvHeaders([]);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || "");
      const lines = text.trim().split(/\r?\n/).slice(0, 6);
      const rows = lines.map((line) => line.split(","));
      setCsvHeaders(rows[0] || []);
      setCsvPreview(rows.slice(1));
    };
    reader.readAsText(dataFile);
  }, [dataFile]);

  const availableCheckpoints = useMemo(
    () =>
      checkpoints
        .filter((ckpt) => !ckpt.is_finetuned)
        .slice()
        .sort((a, b) => b.mtime - a.mtime),
    [checkpoints]
  );

  useEffect(() => {
    if (selectedCheckpoint || availableCheckpoints.length === 0) {
      return;
    }
    setSelectedCheckpoint(availableCheckpoints[0].id);
  }, [availableCheckpoints, selectedCheckpoint]);

  const ssePath = job ? `/api/finetune/jobs/${job.job_id}/events` : undefined;
  const { lastEvent, error: sseError } = useSse(ssePath, Boolean(job));
  const withTimestamp = (message: string) => `[${formatTimestamp()}] ${message}`;
  const { activeSection, setActiveSection } = useScrollSpy(finetuneSections);

  useEffect(() => {
    if (!lastEvent) {
      return;
    }
    if (lastEvent.type === "status") {
      const payload = lastEvent.payload as JobStatus;
      setJob((prev) => (prev ? { ...prev, ...payload } : payload));
    }
    if (lastEvent.type === "metrics") {
      const payload = lastEvent.payload as MetricsPayload;
      setMetrics(payload);
      setMetricsHistory((prev) => [...prev.slice(-199), payload]);
      setJob((prev) =>
        prev
          ? {
              ...prev,
              iter: payload.iter ?? prev.iter,
              max_iters: payload.max_iters ?? prev.max_iters,
            }
          : prev
      );
    }
    if (lastEvent.type === "eval") {
      const payload = lastEvent.payload as Record<string, number>;
      setEvalHistory((prev) => [...prev.slice(-199), payload]);
    }
    if (lastEvent.type === "checkpoint") {
      const payload = lastEvent.payload as { iter: number };
      setLogs((prev) => [withTimestamp(`Checkpoint saved at ${payload.iter}`), ...prev].slice(0, 200));
    }
    if (lastEvent.type === "eval") {
      const payload = lastEvent.payload as { iter?: number; train_loss?: number; val_loss?: number };
      const iter = payload.iter ?? "?";
      const train = payload.train_loss?.toFixed?.(4) ?? "-";
      const val = payload.val_loss?.toFixed?.(4) ?? "-";
      setLogs((prev) => [withTimestamp(`Eval @ ${iter}: train ${train}, val ${val}`), ...prev].slice(0, 200));
    }
    if (lastEvent.type === "log") {
      const payload = lastEvent.payload as { message?: string };
      const message = payload?.message;
      if (message) {
        setLogs((prev) => [withTimestamp(message), ...prev].slice(0, 200));
      }
    }
    if (lastEvent.type === "done") {
      const payload = lastEvent.payload as JobStatus;
      setJob((prev) => (prev ? { ...prev, ...payload } : payload));
    }
    if (lastEvent.type === "error") {
      const payload = lastEvent.payload as { message?: string };
      setError(payload?.message || "Fine-tuning error");
    }
  }, [lastEvent]);

  useEffect(() => {
    if (sseError) {
      setError(sseError);
    }
  }, [sseError]);


  const isRunning = job?.status === "running";
  const isPaused = job?.status === "paused";
  const isInactive = !job || ["completed", "error", "canceled"].includes(job.status);
  const progress = job && job.max_iters ? Math.min(job.iter / job.max_iters, 1) : 0;
  const elapsedTime = metrics?.elapsed_time;
  const etaSeconds = metrics?.eta_seconds;
  const elapsedDisplay = elapsedTime !== undefined ? formatDuration(elapsedTime) : job ? "Calculating..." : "-";
  const etaDisplay =
    etaSeconds !== undefined && etaSeconds !== null ? formatDuration(etaSeconds) : job ? "Calculating..." : "-";
  const layersCount = checkpointConfig ? Number(checkpointConfig.n_layers || 0) : 0;
  const headsCount = checkpointConfig ? Number(checkpointConfig.n_heads || 0) : 0;
  const maxSampleIndex = Math.max(0, trainingParams.batch_size - 1);
  const maxLayerIndex = Math.max(0, layersCount - 1);
  const maxHeadIndex = Math.max(0, headsCount - 1);

  const createJob = async () => {
    setError(null);
    setIsCreating(true);
    try {
      if (!selectedCheckpoint) {
        throw new Error("Select a checkpoint before starting.");
      }
      const payload = {
        checkpoint_id: selectedCheckpoint,
        max_length: maxLength,
        use_lora: method === "lora",
        ...loraConfig,
        training: trainingParams,
        auto_start: autoStart,
        mask_prompt: true,
      };
      const form = makeFormData(payload, dataFile || undefined, "data_file");
      const data = await fetchJson<JobStatus>("/api/finetune/jobs", {
        method: "POST",
        body: form,
      });
      setJob(data);
      setMetrics(null);
      setLogs([]);
      setMetricsHistory([]);
      setEvalHistory([]);
      setInspectData(null);
      setAttention([]);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsCreating(false);
    }
  };

  const stepJob = async () => {
    if (!job) return;
    setError(null);
    try {
      await fetchJson(`/api/finetune/jobs/${job.job_id}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ include_batch: false }),
      });
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const pauseJob = async () => {
    if (!job) return;
    await fetchJson(`/api/finetune/jobs/${job.job_id}/pause`, { method: "POST" });
  };

  const resumeJob = async () => {
    if (!job) return;
    await fetchJson(`/api/finetune/jobs/${job.job_id}/resume`, { method: "POST" });
  };

  const handlePrimaryAction = async () => {
    if (!job || isInactive) {
      await createJob();
      return;
    }
    if (job.status === "running") {
      await pauseJob();
      return;
    }
    await resumeJob();
  };

  const loadSnippets = async () => {
    setError(null);
    try {
      const data = await fetchJson<{ snippets: CodeSnippet[] }>("/api/docs/finetuning-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_lora: method === "lora" }),
      });
      setSnippets(data.snippets);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const inspectBatch = async (sampleIndex?: number, silent = false) => {
    if (!job) return;
    if (!silent) {
      setError(null);
    }
    try {
      const index = sampleIndex ?? inspectSample;
      const data = await fetchJson<{
        token_labels: string[];
        prompt_tokens: string[];
        response_tokens: string[];
      }>(`/api/finetune/jobs/${job.job_id}/inspect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_index: index,
          max_tokens: inspectMaxTokens,
        }),
      });
      setInspectData(data);
      setAttention([]);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const loadAttention = async (sampleIndex?: number, silent = false) => {
    if (!job) return;
    if (!silent) {
      setError(null);
    }
    try {
      const index = isRunning ? 0 : sampleIndex ?? inspectSample;
      const data = await fetchJson<{ attention: number[][] }>(
        `/api/finetune/jobs/${job.job_id}/attention`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_index: index,
            max_tokens: inspectMaxTokens,
            layer: attnLayer,
            head: attnHead,
          }),
        }
      );
      setAttention(data.attention);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const refreshInspect = async (sampleIndex?: number, silent = false) => {
    await inspectBatch(sampleIndex, silent);
    await loadAttention(sampleIndex, silent);
  };

  const handleSampleChange = (value: number) => {
    setInspectSample(value);
    if (!job || isRunning) {
      return;
    }
    refreshInspect(value);
  };

  useEffect(() => {
    if (inspectSample > maxSampleIndex) {
      setInspectSample(maxSampleIndex);
    }
  }, [inspectSample, maxSampleIndex]);

  useEffect(() => {
    let active = true;
    setSnippetsLoading(true);
    const timeout = setTimeout(() => {
      loadSnippets().finally(() => {
        if (active) {
          setSnippetsLoading(false);
        }
      });
    }, 400);
    return () => {
      active = false;
      clearTimeout(timeout);
    };
  }, [method]);

  useEffect(() => {
    if (job?.status === "running") {
      setInspectSample(0);
    }
  }, [job?.status]);

  useEffect(() => {
    if (!job || job.status !== "running") {
      return;
    }
    if (!lastEvent || lastEvent.type !== "metrics") {
      return;
    }
    const now = Date.now();
    if (now - inspectThrottleRef.current < 1000) {
      return;
    }
    inspectThrottleRef.current = now;
    refreshInspect(0, true);
  }, [lastEvent, job]);

  useEffect(() => {
    if (!job || isRunning || !inspectData) {
      return;
    }
    loadAttention(undefined, true);
  }, [attnLayer, attnHead, job, isRunning, inspectData]);

  return (
    <div className="page-with-nav">
      <SideNav
        sections={finetuneSections}
        activeId={activeSection}
        onNavigate={setActiveSection}
        ariaLabel="Fine-tuning sections"
      />
      <div className="page-content">
        <section id="checkpoint" className="section scroll-section">
        <div className="section-title">
          <h2>Select Model</h2>
          <p>Pick a pre-trained checkpoint to fine-tune.</p>
        </div>
        <div className="card">
          <select
            value={selectedCheckpoint}
            onChange={(event) => setSelectedCheckpoint(event.target.value)}
          >
            <option value="">Select checkpoint</option>
            {availableCheckpoints.map((ckpt) => (
              <option key={ckpt.id} value={ckpt.id}>
                {formatCheckpointTimestamp(new Date(ckpt.mtime * 1000))} · {ckpt.name}
              </option>
            ))}
          </select>
          {checkpointConfig && (
            <ModelConfigSummary
              config={checkpointConfig}
              summaryItems={[
                { label: "d_model", value: checkpointConfig.d_model || "-" },
                { label: "n_layers", value: checkpointConfig.n_layers || "-" },
                { label: "n_heads", value: checkpointConfig.n_heads || "-" },
              ]}
            />
          )}
        </div>
      </section>

      <section id="method" className="section scroll-section">
        <div className="section-title">
          <h2>Method</h2>
          <p>Full parameter or LoRA.</p>
        </div>
        <div className="card">
          <div className="inline-row">
            <button
              className={method === "full" ? "primary" : "secondary"}
              onClick={() => setMethod("full")}
            >
              Full Parameter
            </button>
            <button
              className={method === "lora" ? "primary" : "secondary"}
              onClick={() => setMethod("lora")}
            >
              LoRA
            </button>
          </div>

          {method === "lora" && (
            <div className="grid-3" style={{ marginTop: 16 }}>
              <div>
                <label>Rank</label>
                <input
                  type="number"
                  value={loraConfig.lora_rank}
                  onChange={(event) =>
                    setLoraConfig((prev) => ({ ...prev, lora_rank: Number(event.target.value) }))
                  }
                />
              </div>
              <div>
                <label>Alpha</label>
                <input
                  type="number"
                  value={loraConfig.lora_alpha}
                  onChange={(event) =>
                    setLoraConfig((prev) => ({ ...prev, lora_alpha: Number(event.target.value) }))
                  }
                />
              </div>
              <div>
                <label>Dropout</label>
                <input
                  type="number"
                  step="0.01"
                  value={loraConfig.lora_dropout}
                  onChange={(event) =>
                    setLoraConfig((prev) => ({ ...prev, lora_dropout: Number(event.target.value) }))
                  }
                />
              </div>
              <div>
                <label>Target Modules</label>
                <select
                  value={loraConfig.lora_target_modules}
                  onChange={(event) =>
                    setLoraConfig((prev) => ({ ...prev, lora_target_modules: event.target.value }))
                  }
                >
                  <option value="all">all</option>
                  <option value="attention">attention</option>
                  <option value="mlp">mlp</option>
                </select>
              </div>
            </div>
          )}
          {method === "lora" && (
            <div style={{ marginTop: 12 }} className="badge">
              LoRA enabled · rank {loraConfig.lora_rank} · alpha {loraConfig.lora_alpha}
            </div>
          )}
        </div>
      </section>

      <section id="training-data" className="section scroll-section">
        <div className="section-title">
          <h2>Training Data</h2>
          <p>Upload a CSV or fall back to finetuning.csv.</p>
        </div>
        <div className="card">
          <input
            type="file"
            accept=".csv"
            onChange={(event) => setDataFile(event.target.files?.[0] || null)}
          />
          {csvPreview.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <table className="table">
                <thead>
                  <tr>
                    {csvHeaders.map((header, idx) => (
                      <th key={`${header}-${idx}`}>{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {csvPreview.map((row, rowIdx) => (
                    <tr key={`row-${rowIdx}`}>
                      {row.map((cell, cellIdx) => (
                        <td key={`cell-${rowIdx}-${cellIdx}`}>{cell}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>

      <section id="hyperparameters" className="section scroll-section">
        <div className="section-title">
          <h2>Hyperparameters</h2>
          <p>Optimize fine-tuning behavior.</p>
        </div>
        <div className="card">
          <details className="expander">
            <summary>Core Settings</summary>
            <div className="expander-content">
              <div className="grid-3">
                <div>
                  <label>Batch Size</label>
                  <input
                    type="number"
                    value={trainingParams.batch_size}
                    onChange={(event) =>
                      setTrainingParams((prev) => ({ ...prev, batch_size: Number(event.target.value) }))
                    }
                  />
                </div>
                <div>
                  <label>Epochs</label>
                  <input
                    type="number"
                    value={trainingParams.epochs}
                    onChange={(event) =>
                      setTrainingParams((prev) => ({ ...prev, epochs: Number(event.target.value) }))
                    }
                  />
                </div>
                <div>
                  <label>Max Steps/Epoch</label>
                  <input
                    type="number"
                    value={trainingParams.max_steps_per_epoch}
                    onChange={(event) =>
                      setTrainingParams((prev) => ({ ...prev, max_steps_per_epoch: Number(event.target.value) }))
                    }
                  />
                </div>
                <div>
                  <label>Max Length</label>
                  <input
                    type="number"
                    value={maxLength}
                    onChange={(event) => setMaxLength(Number(event.target.value))}
                  />
                </div>
              </div>
            </div>
          </details>

          <details className="expander">
            <summary>Optimization</summary>
            <div className="expander-content">
              <div className="grid-3">
                <div>
                  <label>Learning Rate</label>
                  <input
                    type="number"
                    step="0.000001"
                    value={trainingParams.learning_rate}
                    onChange={(event) =>
                      setTrainingParams((prev) => ({ ...prev, learning_rate: Number(event.target.value) }))
                    }
                  />
                </div>
                <div>
                  <label>Weight Decay</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={trainingParams.weight_decay}
                    onChange={(event) =>
                      setTrainingParams((prev) => ({ ...prev, weight_decay: Number(event.target.value) }))
                    }
                  />
                </div>
              </div>
            </div>
          </details>

          <details className="expander">
            <summary>Evaluation & Checkpointing</summary>
            <div className="expander-content">
              <div className="grid-3">
                <div>
                  <label>Eval Interval</label>
                  <input
                    type="number"
                    value={trainingParams.eval_interval}
                    onChange={(event) =>
                      setTrainingParams((prev) => ({ ...prev, eval_interval: Number(event.target.value) }))
                    }
                  />
                </div>
                <div>
                  <label>Save Interval</label>
                  <input
                    type="number"
                    value={trainingParams.save_interval}
                    onChange={(event) =>
                      setTrainingParams((prev) => ({ ...prev, save_interval: Number(event.target.value) }))
                    }
                  />
                </div>
                <div>
                  <label>Auto Start</label>
                  <select value={autoStart ? "yes" : "no"} onChange={(event) => setAutoStart(event.target.value === "yes")}>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </div>
              </div>
            </div>
          </details>
          <div className="grid-3" style={{ marginTop: 16 }}>
            <StatCard label="Batch Size" value={trainingParams.batch_size} />
            <StatCard label="Learning Rate" value={trainingParams.learning_rate} />
            <StatCard label="Max Length" value={maxLength} />
          </div>
        </div>
      </section>

      <section id="understand" className="section scroll-section">
        <div className="section-title">
          <h2>Understand</h2>
          <p>Masked loss and optional LoRA math.</p>
        </div>
        <div className="card">
          <details className="expander">
            <summary>Equations</summary>
            <div className="expander-content">
              <MarkdownBlock content={finetuneEquations} />
              {method === "lora" && <MarkdownBlock content={loraEquations} />}
            </div>
          </details>
          <details className="expander">
            <summary>Code Snippets</summary>
            <div className="expander-content">
              {snippetsLoading ? (
                <p>Loading code snippets...</p>
              ) : snippets.length === 0 ? (
                <p>No snippets available yet.</p>
              ) : (
                snippets.map((snippet) => <CodePanel key={snippet.title} snippet={snippet} />)
              )}
            </div>
          </details>
        </div>
      </section>

      <section id="train" className="section scroll-section">
        <div className="section-title">
          <h2>Train</h2>
          <p>Train your model.</p>
        </div>
        <TrainingControls
          job={job}
          isRunning={isRunning}
          isPaused={isPaused}
          isCreating={isCreating}
          disabled={isDemo}
          disabledReason={isDemo ? "Demo mode: fine-tuning disabled." : undefined}
          progress={progress}
          startLabel="Start Fine-Tuning"
          error={error}
          onPrimary={handlePrimaryAction}
          onStep={stepJob}
        />
      </section>

      <section id="metrics" className="section scroll-section">
        <div className="section-title">
          <h2>Live Metrics</h2>
          <p>Loss and gradients.</p>
        </div>
        <div className="card">
          <div className="grid-3" style={{ marginBottom: 12 }}>
            <StatCard label="Progress" value={`${(progress * 100).toFixed(1)}%`} />
            <StatCard label="Elapsed" value={elapsedDisplay} />
            <StatCard label="ETA" value={etaDisplay} />
          </div>
          <div className="grid-3" style={{ marginBottom: 12 }}>
            <StatCard label="Loss" value={metrics?.loss?.toFixed?.(4) || "-"} />
            <StatCard label="Running Loss" value={metrics?.running_loss?.toFixed?.(4) || "-"} />
            <StatCard label="Grad Norm" value={metrics?.grad_norm?.toFixed?.(4) || "-"} />
          </div>
          <LineChart
            data={metricsHistory.map((row) => ({
              iter: row.iter ?? 0,
              loss: row.loss ?? 0,
              running_loss: row.running_loss ?? 0,
            }))}
            xKey="iter"
            lines={[
              { dataKey: "loss", name: "Loss", color: "var(--accent)" },
              { dataKey: "running_loss", name: "Running Loss", color: "var(--accent-2)" },
            ]}
          />
        </div>
      </section>

      <section id="inspect-batch" className="section scroll-section">
        <div className="section-title">
          <h2>Inspect Batch</h2>
          <p>Prompt vs response tokens and attention patterns.</p>
        </div>
        <div className="card">
          <RangeSlider
            label="Sample"
            min={0}
            max={maxSampleIndex}
            value={inspectSample}
            onChange={handleSampleChange}
            disabled={!job || isRunning}
            style={{ marginBottom: 12 }}
          />
          {isRunning && <p>Live batch inspection updates while training is running.</p>}

          {inspectData ? (
            <div className="grid-2">
              <div>
                <label>Prompt Tokens</label>
                <div className="card" style={{ boxShadow: "none", background: "var(--card-muted)" }}>
                  <TokenSegments tokens={inspectData.prompt_tokens} tone="prompt" />
                </div>
              </div>
              <div>
                <label>Response Tokens</label>
                <div className="card" style={{ boxShadow: "none", background: "var(--card-muted)" }}>
                  <TokenSegments tokens={inspectData.response_tokens} tone="response" />
                </div>
              </div>
            </div>
          ) : (
            <p>Select a sample to inspect prompt/response tokens.</p>
          )}

          <div style={{ marginTop: 16 }}>
            <div className="grid-2">
              <RangeSlider
                label="Layer"
                min={0}
                max={maxLayerIndex}
                value={attnLayer}
                onChange={setAttnLayer}
                disabled={!job || isRunning || maxLayerIndex === 0}
              />
              <RangeSlider
                label="Head"
                min={0}
                max={maxHeadIndex}
                value={attnHead}
                onChange={setAttnHead}
                disabled={!job || isRunning || maxHeadIndex === 0}
              />
            </div>
            <Heatmap matrix={attention} labels={inspectData?.token_labels || []} />
          </div>
        </div>
      </section>

      <section id="eval-history" className="section scroll-section">
        <div className="section-title">
          <h2>Eval History</h2>
          <p>Train vs validation loss.</p>
        </div>
        <div className="card">
          <LineChart
            data={evalHistory.map((row) => ({
              iter: row.iter || 0,
              train: row.train_loss || 0,
              val: row.val_loss || 0,
            }))}
            xKey="iter"
            lines={[
              { dataKey: "train", name: "Train Loss", color: "var(--accent)" },
              { dataKey: "val", name: "Val Loss", color: "#fbbf24" },
            ]}
          />
        </div>
      </section>

      <section id="logs" className="section scroll-section">
        <div className="section-title">
          <h2>Logs</h2>
          <p>Checkpoint events.</p>
        </div>
        <div className="card">
          <LogBox logs={logs} />
        </div>
      </section>
      </div>
    </div>
  );
}
