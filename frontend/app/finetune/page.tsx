"use client";

import { useEffect, useMemo, useState } from "react";
import CodePanel from "../../components/CodePanel";
import LineChart from "../../components/LineChart";
import StatCard from "../../components/StatCard";
import { fetchJson, makeFormData, Checkpoint, CodeSnippet, JobStatus } from "../../lib/api";
import { useSse } from "../../lib/useSse";
import MarkdownBlock from "../../components/MarkdownBlock";
import { finetuneEquations, loraEquations } from "../../lib/equations";

export default function FinetunePage() {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [method, setMethod] = useState<"full" | "lora">("full");
  const [loraConfig, setLoraConfig] = useState({
    lora_rank: 8,
    lora_alpha: 8,
    lora_dropout: 0,
    lora_target_modules: "all",
  });
  const [dataFile, setDataFile] = useState<File | null>(null);
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
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<Record<string, number>[]>([]);
  const [evalHistory, setEvalHistory] = useState<Record<string, number>[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [snippets, setSnippets] = useState<CodeSnippet[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJson<{ checkpoints: Checkpoint[] }>("/api/checkpoints")
      .then((data) => setCheckpoints(data.checkpoints))
      .catch((err) => setError((err as Error).message));
  }, []);

  const availableCheckpoints = useMemo(
    () => checkpoints.filter((ckpt) => !ckpt.is_finetuned),
    [checkpoints]
  );

  const ssePath = job ? `/api/finetune/jobs/${job.job_id}/events` : undefined;
  const { lastEvent } = useSse(ssePath, Boolean(job));

  useEffect(() => {
    if (!lastEvent) {
      return;
    }
    if (lastEvent.type === "status") {
      setJob((prev) => ({
        ...(prev || {}),
        ...(lastEvent.payload as Record<string, unknown>),
      }));
    }
    if (lastEvent.type === "metrics") {
      const payload = lastEvent.payload as Record<string, number>;
      setMetrics(payload);
      setMetricsHistory((prev) => [...prev.slice(-199), payload]);
    }
    if (lastEvent.type === "eval") {
      const payload = lastEvent.payload as Record<string, number>;
      setEvalHistory((prev) => [...prev.slice(-199), payload]);
    }
    if (lastEvent.type === "checkpoint") {
      const payload = lastEvent.payload as { iter: number };
      setLogs((prev) => [`Checkpoint saved at ${payload.iter}`, ...prev].slice(0, 50));
    }
    if (lastEvent.type === "error") {
      const payload = lastEvent.payload as { message?: string };
      setError(payload?.message || "Fine-tuning error");
    }
  }, [lastEvent]);

  const progress = job ? Math.min(job.iter / job.max_iters, 1) : 0;

  const createJob = async () => {
    setError(null);
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
      setLogs([]);
      setMetricsHistory([]);
      setEvalHistory([]);
    } catch (err) {
      setError((err as Error).message);
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

  return (
    <>
      <section className="section">
        <div className="section-title">
          <h2>Select Checkpoint</h2>
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
                {ckpt.name}
              </option>
            ))}
          </select>
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Fine-Tuning Method</h2>
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
                <label>LoRA Rank</label>
                <input
                  type="number"
                  value={loraConfig.lora_rank}
                  onChange={(event) =>
                    setLoraConfig((prev) => ({ ...prev, lora_rank: Number(event.target.value) }))
                  }
                />
              </div>
              <div>
                <label>LoRA Alpha</label>
                <input
                  type="number"
                  value={loraConfig.lora_alpha}
                  onChange={(event) =>
                    setLoraConfig((prev) => ({ ...prev, lora_alpha: Number(event.target.value) }))
                  }
                />
              </div>
              <div>
                <label>LoRA Dropout</label>
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
        </div>
      </section>

      <section className="section">
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
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Hyperparameters</h2>
          <p>Optimize fine-tuning behavior.</p>
        </div>
        <div className="card">
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
              <label>Max Length</label>
              <input
                type="number"
                value={maxLength}
                onChange={(event) => setMaxLength(Number(event.target.value))}
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
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Understand Fine-Tuning</h2>
          <p>Masked loss and optional LoRA math.</p>
        </div>
        <div className="card">
          <MarkdownBlock content={finetuneEquations} />
          {method === "lora" && <MarkdownBlock content={loraEquations} />}
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Control</h2>
          <p>Start, pause, resume, or step.</p>
        </div>
        <div className="card">
          <div className="inline-row" style={{ marginBottom: 12 }}>
            <button className="primary" onClick={createJob}>Start Fine-Tuning</button>
            <button className="secondary" onClick={pauseJob} disabled={!job}>Pause</button>
            <button className="secondary" onClick={resumeJob} disabled={!job}>Resume</button>
            <button className="secondary" onClick={stepJob} disabled={!job}>Step</button>
            <button className="secondary" onClick={loadSnippets}>Load Code Snippets</button>
          </div>
          {job && (
            <>
              <div className="flex-between" style={{ marginBottom: 8 }}>
                <span className="badge">{job.status}</span>
                <span className="badge">{job.iter} / {job.max_iters}</span>
              </div>
              <div className="progress">
                <span style={{ width: `${progress * 100}%` }} />
              </div>
            </>
          )}
          {error && <p style={{ color: "#b42318" }}>{error}</p>}
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Metrics</h2>
          <p>Loss and gradients.</p>
        </div>
        <div className="card">
          <div className="grid-3" style={{ marginBottom: 12 }}>
            <StatCard label="Loss" value={metrics?.loss?.toFixed?.(4) || "-"} />
            <StatCard label="Running Loss" value={metrics?.running_loss?.toFixed?.(4) || "-"} />
            <StatCard label="Grad Norm" value={metrics?.grad_norm?.toFixed?.(4) || "-"} />
          </div>
          <LineChart
            data={metricsHistory.map((row) => ({
              iter: row.iter || 0,
              loss: row.loss || 0,
              running_loss: row.running_loss || 0,
            }))}
            xKey="iter"
            lines={[
              { dataKey: "loss", name: "Loss", color: "#d24b1a" },
              { dataKey: "running_loss", name: "Running Loss", color: "#0f4c5c" },
            ]}
          />
        </div>
      </section>

      <section className="section">
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
              { dataKey: "train", name: "Train Loss", color: "#1b6ca8" },
              { dataKey: "val", name: "Val Loss", color: "#a00f24" },
            ]}
          />
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Logs</h2>
          <p>Checkpoint events.</p>
        </div>
        <div className="card">
          <div className="log-box">
            {logs.length === 0 ? "No logs yet." : logs.join("\n")}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Code Snippets</h2>
          <p>Inspect fine-tuning code paths.</p>
        </div>
        <div>
          {snippets.length === 0 ? (
            <div className="card">Load code snippets to inspect fine-tuning modules.</div>
          ) : (
            snippets.map((snippet) => <CodePanel key={snippet.title} snippet={snippet} />)
          )}
        </div>
      </section>
    </>
  );
}
