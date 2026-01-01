"use client";

import { useEffect, useMemo, useState } from "react";
import CodePanel from "../../components/CodePanel";
import GraphvizDiagram from "../../components/GraphvizDiagram";
import MarkdownBlock from "../../components/MarkdownBlock";
import LineChart from "../../components/LineChart";
import StatCard from "../../components/StatCard";
import { fetchJson, makeFormData, CodeSnippet, JobStatus } from "../../lib/api";
import { useSse } from "../../lib/useSse";
import {
  defaultModelConfig,
  applyPreset,
  applySizePreset,
  estimateParams,
  ModelConfig,
  ModelSize,
} from "../../lib/modelConfig";
import { generateGraphvizArchitecture } from "../../lib/graphviz";
import { modelEquations } from "../../lib/equations";

const tokenizerOptions = ["character", "bpe-simple", "bpe-tiktoken", "sentencepiece"];

export default function PretrainPage() {
  const [modelConfig, setModelConfig] = useState<ModelConfig>(defaultModelConfig);
  const [modelSize, setModelSize] = useState<ModelSize>("small");
  const [useEinops, setUseEinops] = useState(true);
  const [tokenizerType, setTokenizerType] = useState("bpe-tiktoken");
  const [trainingFile, setTrainingFile] = useState<File | null>(null);
  const [autoStart, setAutoStart] = useState(true);
  const [trainingParams, setTrainingParams] = useState({
    batch_size: 32,
    epochs: 10,
    max_steps_per_epoch: 500,
    learning_rate: 0.001,
    weight_decay: 0.01,
    eval_interval: 500,
    eval_iters: 10,
    save_interval: 1000,
  });
  const [job, setJob] = useState<JobStatus | null>(null);
  const [metrics, setMetrics] = useState<Record<string, number> | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<Record<string, number>[]>([]);
  const [evalHistory, setEvalHistory] = useState<Record<string, number>[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [snippets, setSnippets] = useState<CodeSnippet[]>([]);
  const [error, setError] = useState<string | null>(null);

  const ssePath = job ? `/api/pretrain/jobs/${job.job_id}/events` : undefined;
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
      setError(payload?.message || "Training error");
    }
  }, [lastEvent]);

  const progress = job ? Math.min(job.iter / job.max_iters, 1) : 0;
  const estimatedParams = useMemo(
    () => estimateParams({ ...modelConfig, use_einops: useEinops }),
    [modelConfig, useEinops]
  );
  const diagramDot = useMemo(() => generateGraphvizArchitecture(modelConfig), [modelConfig]);

  const handlePreset = (preset: string) => {
    setModelConfig((prev) => applyPreset(prev, preset));
  };

  const handleSize = (size: ModelSize) => {
    setModelSize(size);
    setModelConfig((prev) => applySizePreset(prev, size));
  };

  const loadSnippets = async () => {
    setError(null);
    try {
      const data = await fetchJson<{ snippets: CodeSnippet[] }>("/api/docs/model-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_config: { ...modelConfig, use_einops: useEinops } }),
      });
      setSnippets(data.snippets);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const createJob = async () => {
    setError(null);
    try {
      const payload = {
        model_config: { ...modelConfig, use_einops: useEinops },
        tokenizer_type: tokenizerType,
        use_einops: useEinops,
        training: trainingParams,
        auto_start: autoStart,
      };
      const form = makeFormData(payload, trainingFile || undefined, "training_file");
      const data = await fetchJson<JobStatus>("/api/pretrain/jobs", {
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
      await fetchJson(`/api/pretrain/jobs/${job.job_id}/step`, {
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
    await fetchJson(`/api/pretrain/jobs/${job.job_id}/pause`, { method: "POST" });
  };

  const resumeJob = async () => {
    if (!job) return;
    await fetchJson(`/api/pretrain/jobs/${job.job_id}/resume`, { method: "POST" });
  };

  return (
    <>
      <section className="section">
        <div className="section-title">
          <h2>Upload Training Data</h2>
          <p>Provide a text file or let the backend use training.txt.</p>
        </div>
        <div className="card">
          <input
            type="file"
            accept=".txt"
            onChange={(event) => setTrainingFile(event.target.files?.[0] || null)}
          />
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Model Architecture</h2>
          <p>Presets, components, and dimensions.</p>
        </div>
        <div className="card">
          <div className="inline-row" style={{ marginBottom: 12 }}>
            <button className="secondary" onClick={() => handlePreset("gpt")}>GPT-2</button>
            <button className="secondary" onClick={() => handlePreset("llama")}>LLaMA 4</button>
            <button className="secondary" onClick={() => handlePreset("olmo")}>OLMo 3</button>
            <button className="secondary" onClick={() => handlePreset("deepseek")}>DeepSeek V2</button>
            <button className="secondary" onClick={() => handlePreset("mixtral")}>Mixtral</button>
          </div>

          <div className="inline-row" style={{ marginBottom: 16 }}>
            {["small", "medium", "full"].map((size) => (
              <button
                key={size}
                className={modelSize === size ? "primary" : "secondary"}
                onClick={() => handleSize(size as ModelSize)}
              >
                {size}
              </button>
            ))}
          </div>

          <div className="grid-3">
            <div>
              <label>Positional Encoding</label>
              <select
                value={modelConfig.positional_encoding}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    positional_encoding: event.target.value as ModelConfig["positional_encoding"],
                  }))
                }
              >
                <option value="learned">Learned</option>
                <option value="rope">RoPE</option>
                <option value="alibi">ALiBi</option>
                <option value="none">None</option>
              </select>
            </div>
            <div>
              <label>Normalization</label>
              <select
                value={modelConfig.normalization}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    normalization: event.target.value as ModelConfig["normalization"],
                  }))
                }
              >
                <option value="layernorm">LayerNorm</option>
                <option value="rmsnorm">RMSNorm</option>
              </select>
            </div>
            <div>
              <label>Activation</label>
              <select
                value={modelConfig.activation}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    activation: event.target.value as ModelConfig["activation"],
                  }))
                }
              >
                <option value="gelu">GELU</option>
                <option value="swiglu">SwiGLU</option>
              </select>
            </div>
            {modelConfig.positional_encoding === "rope" && (
              <div>
                <label>RoPE Theta</label>
                <input
                  type="number"
                  value={modelConfig.rope_theta || 10000}
                  onChange={(event) =>
                    setModelConfig((prev) => ({
                      ...prev,
                      rope_theta: Number(event.target.value),
                    }))
                  }
                />
              </div>
            )}
            <div>
              <label>d_model</label>
              <input
                type="number"
                value={modelConfig.d_model}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    d_model: Number(event.target.value),
                  }))
                }
              />
            </div>
            <div>
              <label>n_heads</label>
              <input
                type="number"
                value={modelConfig.n_heads}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    n_heads: Number(event.target.value),
                  }))
                }
              />
            </div>
            <div>
              <label>n_kv_heads</label>
              <input
                type="number"
                value={modelConfig.n_kv_heads || modelConfig.n_heads}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    n_kv_heads: Number(event.target.value),
                  }))
                }
              />
            </div>
            <div>
              <label>n_layers</label>
              <input
                type="number"
                value={modelConfig.n_layers}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    n_layers: Number(event.target.value),
                  }))
                }
              />
            </div>
            <div>
              <label>n_ctx</label>
              <input
                type="number"
                value={modelConfig.n_ctx}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    n_ctx: Number(event.target.value),
                  }))
                }
              />
            </div>
            <div>
              <label>d_head</label>
              <input
                type="number"
                value={modelConfig.d_head}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    d_head: Number(event.target.value),
                  }))
                }
              />
            </div>
            <div>
              <label>d_mlp</label>
              <input
                type="number"
                value={modelConfig.d_mlp}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    d_mlp: Number(event.target.value),
                  }))
                }
              />
            </div>
            <div>
              <label>Use MoE</label>
              <select
                value={modelConfig.use_moe ? "yes" : "no"}
                onChange={(event) =>
                  setModelConfig((prev) => ({
                    ...prev,
                    use_moe: event.target.value === "yes",
                  }))
                }
              >
                <option value="no">No</option>
                <option value="yes">Yes</option>
              </select>
            </div>
            <div>
              <label>Use Einops</label>
              <select value={useEinops ? "yes" : "no"} onChange={(event) => setUseEinops(event.target.value === "yes")}>
                <option value="yes">Yes</option>
                <option value="no">No</option>
              </select>
            </div>
          </div>

          {modelConfig.use_moe && (
            <div className="grid-3" style={{ marginTop: 16 }}>
              <div>
                <label>Experts</label>
                <input
                  type="number"
                  value={modelConfig.num_experts || 8}
                  onChange={(event) =>
                    setModelConfig((prev) => ({
                      ...prev,
                      num_experts: Number(event.target.value),
                    }))
                  }
                />
              </div>
              <div>
                <label>Experts Per Token</label>
                <input
                  type="number"
                  value={modelConfig.num_experts_per_tok || 2}
                  onChange={(event) =>
                    setModelConfig((prev) => ({
                      ...prev,
                      num_experts_per_tok: Number(event.target.value),
                    }))
                  }
                />
              </div>
              <div>
                <label>Shared Experts</label>
                <select
                  value={modelConfig.use_shared_experts ? "yes" : "no"}
                  onChange={(event) =>
                    setModelConfig((prev) => ({
                      ...prev,
                      use_shared_experts: event.target.value === "yes",
                    }))
                  }
                >
                  <option value="no">No</option>
                  <option value="yes">Yes</option>
                </select>
              </div>
              <div>
                <label>Shared Expert Count</label>
                <input
                  type="number"
                  value={modelConfig.num_shared_experts || 2}
                  onChange={(event) =>
                    setModelConfig((prev) => ({
                      ...prev,
                      num_shared_experts: Number(event.target.value),
                    }))
                  }
                />
              </div>
              <div>
                <label>Router Type</label>
                <select
                  value={modelConfig.router_type || "top_k"}
                  onChange={(event) =>
                    setModelConfig((prev) => ({
                      ...prev,
                      router_type: event.target.value as ModelConfig["router_type"],
                    }))
                  }
                >
                  <option value="top_k">top_k</option>
                  <option value="top_k_with_shared">top_k_with_shared</option>
                </select>
              </div>
              <div>
                <label>Load Balance Weight</label>
                <input
                  type="number"
                  step="0.001"
                  value={modelConfig.load_balancing_loss_weight || 0.01}
                  onChange={(event) =>
                    setModelConfig((prev) => ({
                      ...prev,
                      load_balancing_loss_weight: Number(event.target.value),
                    }))
                  }
                />
              </div>
              <div>
                <label>Expert Capacity Factor</label>
                <input
                  type="number"
                  step="0.05"
                  value={modelConfig.expert_capacity_factor || 1.25}
                  onChange={(event) =>
                    setModelConfig((prev) => ({
                      ...prev,
                      expert_capacity_factor: Number(event.target.value),
                    }))
                  }
                />
              </div>
            </div>
          )}

          <div className="grid-3" style={{ marginTop: 16 }}>
            <StatCard label="Est. Parameters" value={`${estimatedParams.toFixed(1)}M`} />
            <StatCard label="Batch Size" value={trainingParams.batch_size} />
            <StatCard label="Learning Rate" value={trainingParams.learning_rate} />
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Tokenizer</h2>
          <p>Choose how input text is tokenized.</p>
        </div>
        <div className="card">
          <select value={tokenizerType} onChange={(event) => setTokenizerType(event.target.value)}>
            {tokenizerOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Training Hyperparameters</h2>
          <p>Core settings for training.</p>
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
                step="0.0001"
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
          <h2>Understand the Model</h2>
          <p>Architecture diagram and math references.</p>
        </div>
        <div className="card">
          <h3>Architecture Diagram</h3>
          <GraphvizDiagram dot={diagramDot} />
          <div style={{ marginTop: 16 }}>
            <h3>Equations</h3>
            <MarkdownBlock content={modelEquations} />
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Training Control</h2>
          <p>Start, pause, or step through batches.</p>
        </div>
        <div className="card">
          <div className="inline-row" style={{ marginBottom: 12 }}>
            <button className="primary" onClick={createJob}>
              Start Training
            </button>
            <button className="secondary" onClick={pauseJob} disabled={!job}>
              Pause
            </button>
            <button className="secondary" onClick={resumeJob} disabled={!job}>
              Resume
            </button>
            <button className="secondary" onClick={stepJob} disabled={!job}>
              Step
            </button>
            <button className="secondary" onClick={loadSnippets}>
              Load Code Snippets
            </button>
          </div>

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
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Live Metrics</h2>
          <p>Training loss and evaluation checkpoints.</p>
        </div>
        <div className="card">
          <div className="grid-3" style={{ marginBottom: 12 }}>
            <StatCard label="Loss" value={metrics?.loss?.toFixed?.(4) || "-"} />
            <StatCard label="Running Loss" value={metrics?.running_loss?.toFixed?.(4) || "-"} />
            <StatCard label="Grad Norm" value={metrics?.grad_norm?.toFixed?.(4) || "-"} />
          </div>
          <LineChart
            data={metricsHistory.map((row) => ({
              iter: row.iter || row.iteration || row.iter_num || row.iter || 0,
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
          <p>Train and validation losses recorded at eval intervals.</p>
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
          <p>Checkpoint and status events.</p>
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
          <p>Backend code rendered for the selected configuration.</p>
        </div>
        <div>
          {snippets.length === 0 ? (
            <div className="card">Load code snippets to inspect model components.</div>
          ) : (
            snippets.map((snippet) => <CodePanel key={snippet.title} snippet={snippet} />)
          )}
        </div>
      </section>
    </>
  );
}
