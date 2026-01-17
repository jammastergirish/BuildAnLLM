"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import CodePanel from "../../components/CodePanel";
import GraphvizDiagram from "../../components/GraphvizDiagram";
import Heatmap from "../../components/Heatmap";
import LogBox from "../../components/LogBox";
import MarkdownBlock from "../../components/MarkdownBlock";
import LineChart from "../../components/LineChart";
import Modal from "../../components/Modal";
import RangeSlider from "../../components/RangeSlider";
import SideNav from "../../components/SideNav";
import StatCard from "../../components/StatCard";
import TokenRainbow from "../../components/TokenRainbow";
import TrainingControls from "../../components/TrainingControls";
import GradientChart from "../../components/GradientChart";
import LossLandscape from "../../components/LossLandscape";
import { fetchJson, makeFormData, CodeSnippet, JobStatus } from "../../lib/api";
import { useSse } from "../../lib/useSse";
import { useScrollSpy } from "../../lib/useScrollSpy";
import { formatDuration, formatTimestamp } from "../../lib/time";
import { useDemoMode } from "../../lib/demo";
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
const presetTokenizerMap: Record<string, string> = {
  gpt: "bpe-tiktoken",
  llama: "sentencepiece",
  olmo: "sentencepiece",
  deepseek: "sentencepiece",
  mixtral: "sentencepiece",
};

const pretrainSections = [
  { id: "training-data", label: "Training Data" },
  { id: "architecture", label: "Architecture" },
  { id: "tokenizer", label: "Tokenizer" },
  { id: "hyperparameters", label: "Hyperparameters" },
  { id: "understand-model", label: "Understand" },
  { id: "train-model", label: "Train" },
  { id: "live-metrics", label: "Metrics" },
  { id: "dynamics", label: "Dynamics" },
  { id: "inspect-batch", label: "Inspect batch" },
  { id: "eval-history", label: "Evaluation history" },
  { id: "logs", label: "Logs" },
];

// Type for pretraining data source from API
type DataSource = {
  name: string;
  filename: string;
  language: string;
  script: string;
  words: number;
  chars: number;
};

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
  layer_grads?: { layer: string; norm: number }[];
  trajectory?: { x: number; y: number };
};

type AxisDomainValue = number | "dataMax";
type AxisDomain = [AxisDomainValue, AxisDomainValue];

export default function PretrainPage() {
  const isDemo = useDemoMode();
  const [modelConfig, setModelConfig] = useState<ModelConfig>(defaultModelConfig);
  const [modelSize, setModelSize] = useState<ModelSize | null>("small");
  const [activePreset, setActivePreset] = useState<string | null>("gpt");
  const [useEinops, setUseEinops] = useState(true);
  const [tokenizerType, setTokenizerType] = useState("bpe-tiktoken");
  const [trainingFiles, setTrainingFiles] = useState<File[]>([]);
  const [uploadedFilesStats, setUploadedFilesStats] = useState<Map<string, { words: number; chars: number }>>(new Map());
  const [includedUploadedFiles, setIncludedUploadedFiles] = useState<Set<string>>(new Set());
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [selectedDataSources, setSelectedDataSources] = useState<Set<string>>(new Set());
  const [sortColumn, setSortColumn] = useState<keyof DataSource | "name" | null>(null);
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
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
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<MetricsPayload[]>([]);
  const [activeLayerGrads, setActiveLayerGrads] = useState<{ layer: string; norm: number }[]>([]);
  const [trajectoryHistory, setTrajectoryHistory] = useState<{ x: number; y: number; iter: number; loss?: number }[]>([]);
  const [evalHistory, setEvalHistory] = useState<Record<string, number>[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [snippets, setSnippets] = useState<CodeSnippet[]>([]);
  const [snippetsLoading, setSnippetsLoading] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inspectSample, setInspectSample] = useState(0);
  const [inspectData, setInspectData] = useState<{
    token_labels: string[];
    target_token: string;
    top_predictions: { token: string; prob: number }[];
    actual_rank?: number | null;
    actual_prob?: number | null;
  } | null>(null);
  const [attention, setAttention] = useState<number[][]>([]);
  const [attnLayer, setAttnLayer] = useState(0);
  const [attnHead, setAttnHead] = useState(0);
  const inspectThrottleRef = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inspectMaxTokens = 128;

  // Text preview modal state
  const [previewModalOpen, setPreviewModalOpen] = useState(false);
  const [previewContent, setPreviewContent] = useState<{ name: string; content: string } | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const openPreview = async (name: string, file?: File) => {
    setPreviewLoading(true);
    setPreviewModalOpen(true);
    try {
      if (file) {
        // For uploaded files, read from the File object directly
        const content = await file.text();
        setPreviewContent({ name, content });
      } else {
        // For built-in sources, fetch from backend
        const data = await fetchJson<{ name: string; content: string }>(
          `/api/pretrain/data-sources/${encodeURIComponent(name)}/content`
        );
        setPreviewContent(data);
      }
    } catch (err) {
      setPreviewContent({ name, content: `Error loading content: ${(err as Error).message}` });
    } finally {
      setPreviewLoading(false);
    }
  };

  const closePreview = () => {
    setPreviewModalOpen(false);
    setPreviewContent(null);
  };

  const ssePath = job ? `/api/pretrain/jobs/${job.job_id}/events` : undefined;
  const { lastEvent, error: sseError } = useSse(ssePath, Boolean(job));
  const withTimestamp = (message: string) => `[${formatTimestamp()}] ${message}`;
  const { activeSection, setActiveSection } = useScrollSpy(pretrainSections);

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
      setMetricsHistory((prev) => [...prev, payload]);
      
      if (payload.layer_grads) {
        setActiveLayerGrads(payload.layer_grads);
      }
      
      if (payload.trajectory) {
        setTrajectoryHistory((prev) => [
            ...prev, 
            { ...payload.trajectory!, iter: payload.iter ?? 0, loss: payload.loss }
        ].slice(-500)); // Keep last 500 points
      }

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
      setError(payload?.message || "Training error");
    }
  }, [lastEvent]);

  useEffect(() => {
    if (sseError) {
      setError(sseError);
    }
  }, [sseError]);


  const estimatedParams = useMemo(
    () => estimateParams({ ...modelConfig, use_einops: useEinops }),
    [modelConfig, useEinops]
  );
  const diagramDot = useMemo(() => generateGraphvizArchitecture(modelConfig), [modelConfig]);
  const attentionType =
    modelConfig.n_kv_heads === modelConfig.n_heads
      ? "mha"
      : modelConfig.n_kv_heads === 1
      ? "mqa"
      : "gqa";
  const isRunning = job?.status === "running";
  const isPaused = job?.status === "paused";
  const isInactive = !job || ["completed", "error", "canceled"].includes(job.status);
  const progress = job && job.max_iters ? Math.min(job.iter / job.max_iters, 1) : 0;
  const elapsedTime = metrics?.elapsed_time;
  const etaSeconds = metrics?.eta_seconds;
  const elapsedDisplay = elapsedTime !== undefined ? formatDuration(elapsedTime) : job ? "Calculating..." : "-";
  const etaDisplay =
    etaSeconds !== undefined && etaSeconds !== null ? formatDuration(etaSeconds) : job ? "Calculating..." : "-";
  const lossXMax = job?.max_iters ?? metrics?.max_iters ?? 0;
  const lossXDomain: AxisDomain = lossXMax > 0 ? [0, lossXMax] : [0, 1];
  const lossYDomain: AxisDomain = metricsHistory.length > 0 ? [0, "dataMax"] : [0, 1];
  const maxSampleIndex = Math.max(0, trainingParams.batch_size - 1);
  const maxLayerIndex = Math.max(0, modelConfig.n_layers - 1);
  const maxHeadIndex = Math.max(0, modelConfig.n_heads - 1);

  // Compute total stats for selected data sources
  const totalDataStats = useMemo(() => {
    let words = 0;
    let chars = 0;
    for (const name of selectedDataSources) {
      const source = dataSources.find((s) => s.name === name);
      if (source) {
        words += source.words;
        chars += source.chars;
      }
    }
    for (const fileName of includedUploadedFiles) {
      const stats = uploadedFilesStats.get(fileName);
      if (stats) {
        words += stats.words;
        chars += stats.chars;
      }
    }
    return { words, chars };
  }, [selectedDataSources, dataSources, includedUploadedFiles, uploadedFilesStats]);

  // Read uploaded files stats
  useEffect(() => {
    if (trainingFiles.length === 0) {
      setUploadedFilesStats(new Map());
      setIncludedUploadedFiles(new Set());
      return;
    }
    const newStats = new Map<string, { words: number; chars: number }>();
    let remaining = trainingFiles.length;
    for (const file of trainingFiles) {
      const reader = new FileReader();
      reader.onload = () => {
        const text = String(reader.result || "");
        const chars = text.length;
        const words = text.split(/\s+/).filter((w) => w.length > 0).length;
        newStats.set(file.name, { words, chars });
        remaining--;
        if (remaining === 0) {
          setUploadedFilesStats(new Map(newStats));
        }
      };
      reader.readAsText(file);
    }
    // Auto-include newly added files
    setIncludedUploadedFiles(new Set(trainingFiles.map((f) => f.name)));
  }, [trainingFiles]);

  // Fetch pretraining data sources from API on mount
  useEffect(() => {
    fetchJson<{ sources: DataSource[] }>("/api/pretrain/data-sources")
      .then((data) => {
        setDataSources(data.sources);
      })
      .catch((err) => setError((err as Error).message));
  }, []);

  const markConfigManual = () => {
    setActivePreset(null);
    setModelSize(null);
  };

  const updateModelConfig = (updater: (prev: ModelConfig) => ModelConfig) => {
    setModelConfig((prev) => updater(prev));
    markConfigManual();
  };

  const handlePreset = (preset: string) => {
    setActivePreset(preset);
    setModelSize("small");
    setTokenizerType(presetTokenizerMap[preset] ?? tokenizerType);
    setModelConfig((prev) => applyPreset(applySizePreset(prev, "small"), preset));
  };

  const handleSize = (size: ModelSize) => {
    setModelSize(size);
    setModelConfig((prev) => applySizePreset(prev, size));
  };

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
  }, [modelConfig, useEinops]);

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
    setIsCreating(true);
    try {
      // Collect all selected training text paths from dataSources
      const trainingTextPaths = Array.from(selectedDataSources)
        .map((name) => dataSources.find((s) => s.name === name)?.filename)
        .filter((f): f is string => Boolean(f));

      // Default to orwell if nothing selected and no file uploaded
      if (trainingTextPaths.length === 0 && includedUploadedFiles.size === 0) {
        const orwell = dataSources.find((s) => s.name === "George Orwell");
        if (orwell) {
          trainingTextPaths.push(orwell.filename);
        }
      }

      const payload = {
        model_config: { ...modelConfig, use_einops: useEinops },
        tokenizer_type: tokenizerType,
        use_einops: useEinops,
        training: trainingParams,
        training_text_paths: trainingTextPaths.length > 0 ? trainingTextPaths : undefined,
        auto_start: autoStart,
      };

      // Combine all included uploaded files into one blob if any are selected
      let fileToUpload: File | undefined;
      const includedFiles = trainingFiles.filter((f) => includedUploadedFiles.has(f.name));
      if (includedFiles.length > 0) {
        const texts = await Promise.all(
          includedFiles.map((f) => f.text())
        );
        const combinedText = texts.join("\n\n");
        fileToUpload = new File([combinedText], "combined_upload.txt", { type: "text/plain" });
      }

      const form = makeFormData(payload, fileToUpload, "training_file");
      const data = await fetchJson<JobStatus>("/api/pretrain/jobs", {
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
      setActiveLayerGrads([]);
      setTrajectoryHistory([]);
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

  const resetJob = async () => {
    // Cancel the job if it exists and is running/paused
    if (job && !["completed", "error", "canceled"].includes(job.status)) {
      try {
        await fetchJson(`/api/pretrain/jobs/${job.job_id}/cancel`, { method: "POST" });
      } catch {
        // Ignore cancel errors - we're resetting anyway
      }
    }

    // Reset all training state
    setJob(null);
    setMetrics(null);
    setMetricsHistory([]);
    setEvalHistory([]);
    setLogs([]);
    setInspectData(null);
    setAttention([]);
    setActiveLayerGrads([]);
    setTrajectoryHistory([]);
    setError(null);
    setIsCreating(false);
    setInspectSample(0);
    setAttnLayer(0);
    setAttnHead(0);
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
        target_token: string;
        top_predictions: { token: string; prob: number }[];
        actual_rank?: number | null;
        actual_prob?: number | null;
      }>(`/api/pretrain/jobs/${job.job_id}/inspect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sample_index: index,
          max_tokens: inspectMaxTokens,
          top_k: 10,
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
        `/api/pretrain/jobs/${job.job_id}/attention`,
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
    <>
    <div className="page-with-nav">
      <SideNav
        sections={pretrainSections}
        activeId={activeSection}
        onNavigate={setActiveSection}
        ariaLabel="Pretraining sections"
      />
      <div className="page-content">
      <section id="training-data" className="section scroll-section">
        <div className="section-title">
          <h2>Training Data</h2>
          <p>Select text corpora to concatenate for training.</p>
        </div>
        <div className="card">
          <div style={{ marginBottom: 16 }}>
            <span className="row-label-title" style={{ display: "block", marginBottom: 8 }}>Upload Custom Files</span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt"
              multiple
              onChange={(event) => {
                const files = Array.from(event.target.files || []);
                if (files.length > 0) {
                  setTrainingFiles((prev) => [...prev, ...files]);
                }
              }}
            />
          </div>

          <div style={{ marginBottom: 16 }}>
            <span className="row-label-title" style={{ display: "block", marginBottom: 12 }}>Select Sources</span>
            <table className="data-table" style={{ width: "100%", tableLayout: "fixed" }}>
              <thead>
                <tr>
                  <th style={{ width: 28, textAlign: "left" }}>
                    <label className="checkbox" title="Select All / None">
                      <input
                        type="checkbox"
                        checked={
                          (dataSources.length > 0 || trainingFiles.length > 0) &&
                          selectedDataSources.size === dataSources.length &&
                          includedUploadedFiles.size === trainingFiles.length
                        }
                        ref={(input) => {
                          if (input) {
                            const totalItems = dataSources.length + trainingFiles.length;
                            const selectedItems = selectedDataSources.size + includedUploadedFiles.size;
                            input.indeterminate = selectedItems > 0 && selectedItems < totalItems;
                          }
                        }}
                        onChange={(event) => {
                          if (event.target.checked) {
                            // Select all
                            setSelectedDataSources(new Set(dataSources.map((s) => s.name)));
                            setIncludedUploadedFiles(new Set(trainingFiles.map((f) => f.name)));
                          } else {
                            // Deselect all
                            setSelectedDataSources(new Set());
                            setIncludedUploadedFiles(new Set());
                          }
                        }}
                        aria-label="Select all sources"
                      />
                      <span className="checkbox-box" aria-hidden="true" />
                    </label>
                  </th>
                  <th
                    style={{ width: "40%", cursor: "pointer" }}
                    onClick={() => {
                      if (sortColumn === "name") {
                        setSortDir(sortDir === "asc" ? "desc" : "asc");
                      } else {
                        setSortColumn("name");
                        setSortDir("asc");
                      }
                    }}
                  >
                    Author/Text {sortColumn === "name" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th
                    style={{ width: "12%", cursor: "pointer" }}
                    onClick={() => {
                      if (sortColumn === "language") {
                        setSortDir(sortDir === "asc" ? "desc" : "asc");
                      } else {
                        setSortColumn("language");
                        setSortDir("asc");
                      }
                    }}
                  >
                    Language {sortColumn === "language" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th
                    style={{ width: "10%", cursor: "pointer" }}
                    onClick={() => {
                      if (sortColumn === "script") {
                        setSortDir(sortDir === "asc" ? "desc" : "asc");
                      } else {
                        setSortColumn("script");
                        setSortDir("asc");
                      }
                    }}
                  >
                    Script {sortColumn === "script" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th
                    style={{ width: "14%", textAlign: "right", cursor: "pointer" }}
                    onClick={() => {
                      if (sortColumn === "words") {
                        setSortDir(sortDir === "asc" ? "desc" : "asc");
                      } else {
                        setSortColumn("words");
                        setSortDir("desc");
                      }
                    }}
                  >
                    Words {sortColumn === "words" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                  <th
                    style={{ width: "14%", textAlign: "right", cursor: "pointer" }}
                    onClick={() => {
                      if (sortColumn === "chars") {
                        setSortDir(sortDir === "asc" ? "desc" : "asc");
                      } else {
                        setSortColumn("chars");
                        setSortDir("desc");
                      }
                    }}
                  >
                    Characters {sortColumn === "chars" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                  </th>
                </tr>
              </thead>
              <tbody>
                {trainingFiles.map((file) => {
                  const stats = uploadedFilesStats.get(file.name);
                  return (
                    <tr key={`uploaded-${file.name}`}>
                      <td>
                        <label className="checkbox">
                          <input
                            type="checkbox"
                            checked={includedUploadedFiles.has(file.name)}
                            onChange={(event) => {
                              setIncludedUploadedFiles((prev) => {
                                const next = new Set(prev);
                                if (event.target.checked) {
                                  next.add(file.name);
                                } else {
                                  next.delete(file.name);
                                }
                                return next;
                              });
                            }}
                          />
                          <span className="checkbox-box" aria-hidden="true" />
                        </label>
                      </td>
                      <td>
                        <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <span
                            className="text-link"
                            style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                            onClick={() => openPreview(file.name, file)}
                            role="button"
                            tabIndex={0}
                            onKeyDown={(e) => e.key === "Enter" && openPreview(file.name, file)}
                          >
                            📄 {file.name}
                          </span>
                          <button
                            type="button"
                            className="secondary"
                            style={{ padding: "2px 6px", fontSize: "0.75rem", flexShrink: 0 }}
                            onClick={() => {
                              setTrainingFiles((prev) => prev.filter((f) => f.name !== file.name));
                              setIncludedUploadedFiles((prev) => {
                                const next = new Set(prev);
                                next.delete(file.name);
                                return next;
                              });
                            }}
                            title="Remove"
                          >
                            ×
                          </button>
                        </span>
                      </td>
                      <td style={{ opacity: 0.6 }}>—</td>
                      <td style={{ opacity: 0.6 }}>—</td>
                      <td style={{ textAlign: "right" }}>
                        {stats?.words.toLocaleString() ?? "—"}
                      </td>
                      <td style={{ textAlign: "right" }}>
                        {stats?.chars.toLocaleString() ?? "—"}
                      </td>
                    </tr>
                  );
                })}
                {[...dataSources]
                  .sort((a, b) => {
                    if (!sortColumn) return 0;
                    const aVal = a[sortColumn];
                    const bVal = b[sortColumn];
                    if (typeof aVal === "number" && typeof bVal === "number") {
                      return sortDir === "asc" ? aVal - bVal : bVal - aVal;
                    }
                    return sortDir === "asc"
                      ? String(aVal).localeCompare(String(bVal))
                      : String(bVal).localeCompare(String(aVal));
                  })
                  .map((source) => (
                  <tr key={source.name}>
                    <td>
                      <label className="checkbox">
                        <input
                          type="checkbox"
                          checked={selectedDataSources.has(source.name)}
                          onChange={(event) => {
                            setSelectedDataSources((prev) => {
                              const next = new Set(prev);
                              if (event.target.checked) {
                                next.add(source.name);
                              } else {
                                next.delete(source.name);
                              }
                              return next;
                            });
                          }}
                        />
                        <span className="checkbox-box" aria-hidden="true" />
                      </label>
                    </td>
                    <td>
                      <span
                        className="text-link"
                        onClick={() => openPreview(source.name)}
                        role="button"
                        tabIndex={0}
                        onKeyDown={(e) => e.key === "Enter" && openPreview(source.name)}
                      >
                        {source.name}
                      </span>
                    </td>
                    <td>{source.language}</td>
                    <td>{source.script}</td>
                    <td style={{ textAlign: "right" }}>{source.words.toLocaleString()}</td>
                    <td style={{ textAlign: "right" }}>{source.chars.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {(totalDataStats.words > 0 || totalDataStats.chars > 0) && (
            <div style={{ paddingTop: 16, borderTop: "1px solid var(--border-subtle)" }}>
              <div style={{ display: "flex", gap: 24 }}>
                <div>
                  <span style={{ opacity: 0.6 }}>Total Words</span>
                  <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>
                    {totalDataStats.words.toLocaleString()}
                  </div>
                </div>
                <div>
                  <span style={{ opacity: 0.6 }}>Total Characters</span>
                  <div style={{ fontSize: "1.25rem", fontWeight: 600 }}>
                    {totalDataStats.chars.toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

      </section>

      <section id="architecture" className="section scroll-section">
        <div className="section-title">
          <h2>Architecture</h2>
          <p>Select a preset and/or go deeper with architecture settings.</p>
        </div>
        <div className="card">
          <div className="row-label" style={{ marginBottom: 12 }}>
            <span className="row-label-title">Architecture</span>
            <div className="inline-row">
              <button
                className={activePreset === "gpt" ? "primary" : "secondary"}
                onClick={() => handlePreset("gpt")}
              >
                GPT-2
              </button>
              <button
                className={activePreset === "llama" ? "primary" : "secondary"}
                onClick={() => handlePreset("llama")}
              >
                LLaMA 4
              </button>
              <button
                className={activePreset === "olmo" ? "primary" : "secondary"}
                onClick={() => handlePreset("olmo")}
              >
                OLMo 3
              </button>
              <button
                className={activePreset === "deepseek" ? "primary" : "secondary"}
                onClick={() => handlePreset("deepseek")}
              >
                DeepSeek V2
              </button>
              <button
                className={activePreset === "mixtral" ? "primary" : "secondary"}
                onClick={() => handlePreset("mixtral")}
              >
                Mixtral
              </button>
            </div>
          </div>

          <div className="row-label" style={{ marginBottom: 16 }}>
            <span className="row-label-title">Size</span>
            <div className="inline-row">
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
          </div>

          <details className="expander">
            <summary>Components</summary>
            <div className="expander-content">
              <div className="grid-3">
                <div>
                  <label>Positional Encoding</label>
                  <select
                    value={modelConfig.positional_encoding}
                    onChange={(event) =>
                      updateModelConfig((prev) => ({
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
                      updateModelConfig((prev) => ({
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
                      updateModelConfig((prev) => ({
                        ...prev,
                        activation: event.target.value as ModelConfig["activation"],
                      }))
                    }
                  >
                    <option value="gelu">GELU</option>
                    <option value="swiglu">SwiGLU</option>
                  </select>
                </div>
                <div>
                  <label>Attention Type</label>
                  <select
                    value={attentionType}
                    onChange={(event) => {
                      const next = event.target.value;
                      updateModelConfig((prev) => {
                        if (next === "mha") {
                          return { ...prev, n_kv_heads: prev.n_heads };
                        }
                        if (next === "mqa") {
                          return { ...prev, n_kv_heads: 1 };
                        }
                        const defaultKv = Math.max(1, Math.floor(prev.n_heads / 4));
                        return { ...prev, n_kv_heads: defaultKv };
                      });
                    }}
                  >
                    <option value="mha">Multi-Head (MHA)</option>
                    <option value="gqa">Grouped Query (GQA)</option>
                    <option value="mqa">Multi-Query (MQA)</option>
                  </select>
                </div>
                {modelConfig.positional_encoding === "rope" && (
                  <div>
                    <label>RoPE Theta</label>
                    <input
                      type="number"
                      value={modelConfig.rope_theta || 10000}
                      onChange={(event) =>
                        updateModelConfig((prev) => ({
                          ...prev,
                          rope_theta: Number(event.target.value),
                        }))
                      }
                    />
                  </div>
                )}
                <div>
                  <label>Use Einops</label>
                  <select
                    value={useEinops ? "yes" : "no"}
                    onChange={(event) => {
                      setUseEinops(event.target.value === "yes");
                      markConfigManual();
                    }}
                  >
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </select>
                </div>
              </div>
            </div>
          </details>

          <details className="expander">
            <summary>Dimensions</summary>
            <div className="expander-content">
              <div className="grid-3">
                <div>
                  <label>d_model</label>
                  <input
                    type="number"
                    value={modelConfig.d_model}
                    onChange={(event) =>
                      updateModelConfig((prev) => ({
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
                      updateModelConfig((prev) => ({
                        ...prev,
                        n_heads: Number(event.target.value),
                      }))
                    }
                  />
                </div>
                {attentionType === "gqa" && (
                  <div>
                    <label>n_kv_heads</label>
                    <input
                      type="number"
                      value={modelConfig.n_kv_heads || modelConfig.n_heads}
                      onChange={(event) =>
                        updateModelConfig((prev) => ({
                          ...prev,
                          n_kv_heads: Number(event.target.value),
                        }))
                      }
                    />
                  </div>
                )}
                <div>
                  <label>n_layers</label>
                  <input
                    type="number"
                    value={modelConfig.n_layers}
                    onChange={(event) =>
                      updateModelConfig((prev) => ({
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
                      updateModelConfig((prev) => ({
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
                      updateModelConfig((prev) => ({
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
                      updateModelConfig((prev) => ({
                        ...prev,
                        d_mlp: Number(event.target.value),
                      }))
                    }
                  />
                </div>
              </div>
            </div>
          </details>

          <details className="expander">
            <summary>Mixture of Experts</summary>
            <div className="expander-content">
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={Boolean(modelConfig.use_moe)}
                  onChange={(event) =>
                    updateModelConfig((prev) => ({
                      ...prev,
                      use_moe: event.target.checked,
                    }))
                  }
                />
                <span className="checkbox-box" aria-hidden="true" />
                <span className="checkbox-text">Use Mixture of Experts</span>
              </label>
              {modelConfig.use_moe && (
                <div className="grid-3" style={{ marginTop: 16 }}>
                  <div>
                    <label>Experts</label>
                    <input
                      type="number"
                      value={modelConfig.num_experts || 8}
                      onChange={(event) =>
                        updateModelConfig((prev) => ({
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
                        updateModelConfig((prev) => ({
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
                        updateModelConfig((prev) => ({
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
                        updateModelConfig((prev) => ({
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
                        updateModelConfig((prev) => ({
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
                        updateModelConfig((prev) => ({
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
                        updateModelConfig((prev) => ({
                          ...prev,
                          expert_capacity_factor: Number(event.target.value),
                        }))
                      }
                    />
                  </div>
                </div>
              )}
            </div>
          </details>

          <div className="grid-3" style={{ marginTop: 16 }}>
            <StatCard label="Parameters" value={`${estimatedParams.toFixed(1)}M`} />
            <StatCard label="Batch Size" value={trainingParams.batch_size} />
            <StatCard label="Learning Rate" value={trainingParams.learning_rate} />
          </div>
        </div>

      </section>

      <section id="tokenizer" className="section scroll-section">
        <div className="section-title">
          <h2>Tokenizer</h2>
          <p>Choose how input text is tokenized.</p>
        </div>
        <div className="card">
          <div className="inline-row">
            {tokenizerOptions.map((option) => (
              <button
                key={option}
                type="button"
                className={tokenizerType === option ? "primary" : "secondary"}
                aria-pressed={tokenizerType === option}
                onClick={() => setTokenizerType(option)}
              >
                {option}
              </button>
            ))}
          </div>
        </div>

      </section>

      <section id="hyperparameters" className="section scroll-section">
        <div className="section-title">
          <h2>Hyperparameters</h2>
          <p>Decide on core settings for training.</p>
        </div>
        <div className="card">
          <details className="expander">
            <summary>Core</summary>
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
        </div>

      </section>

      <section id="understand-model" className="section scroll-section">
        <div className="section-title">
          <h2>Understand</h2>
          <p>Better understand the model architecture, code, and math.</p>
        </div>
        <div className="card">
          <details className="expander">
            <summary>Architecture Diagram</summary>
            <div className="expander-content">
              <GraphvizDiagram dot={diagramDot} />
            </div>
          </details>
          <details className="expander">
            <summary>Equations</summary>
            <div className="expander-content">
              <MarkdownBlock content={modelEquations} />
            </div>
          </details>
          <details className="expander">
            <summary>Code</summary>
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

      <section id="train-model" className="section scroll-section">
        <div className="section-title">
          <h2>Train</h2>
          <p>Train your model.</p>
        </div>
        <TrainingControls
          job={job}
          isRunning={isRunning}
          isPaused={isPaused}
          isCreating={isCreating}
          disabled={isDemo || (!selectedDataSources.size && !includedUploadedFiles.size)}
          disabledReason={
            isDemo
              ? "Demo mode: pre-training disabled."
              : !selectedDataSources.size && !includedUploadedFiles.size
              ? "Select at least one data source in Training Data above."
              : undefined
          }
          progress={progress}
          startLabel="Start Training"
          error={error}
          onPrimary={handlePrimaryAction}
          onStep={stepJob}
          onReset={resetJob}
        />

      </section>

      <section id="live-metrics" className="section scroll-section">
        <div className="section-title">
          <h2>Metrics</h2>
          <p>Training loss and evaluation checkpoints.</p>
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
            xLabel="Iteration"
            yLabel="Loss"
            xDomain={lossXDomain}
            yDomain={lossYDomain}
            lines={[
              { dataKey: "loss", name: "Loss", color: "var(--accent)" },
              { dataKey: "running_loss", name: "Running Loss", color: "var(--accent-2)" },
            ]}
          />
          
          <div style={{ marginTop: 20, borderTop: "1px solid var(--stroke-subtle)", paddingTop: 16 }}>
            <details>
                <summary style={{ cursor: "pointer", color: "var(--ink-2)", fontSize: "0.9rem" }}>
                    What am I looking at?
                </summary>
                <div style={{ marginTop: 12, fontSize: "0.9rem", color: "var(--ink-1)", lineHeight: 1.6 }}>
                    <p style={{ marginBottom: 8 }}><strong>Progress:</strong> How far along the training is (based on max iterations).</p>
                    <p style={{ marginBottom: 8 }}><strong>Loss (Cross Entropy):</strong> The main objective function. Lower is better. It measures how "surprised" the model is by the real next token.</p>
                    <p style={{ marginBottom: 8 }}><strong>Running Loss:</strong> A smoothed average of the loss. This helps you see trends better when the raw loss is jumping around.</p>
                     <p style={{ marginBottom: 8 }}><strong>Grad Norm:</strong> The size of the update step. If this explodes (goes huge), training might crash. If it goes to zero, the model has stopped learning.</p>
                </div>
            </details>
          </div>
        </div>


      </section>

      <section id="dynamics" className="section scroll-section">
        <div className="section-title">
            <h2>Dynamics</h2>
            <p>Visualize gradient norms and optimization trajectory.</p>
        </div>
        <div className="card">
            <div className="grid-2">
                <div>
                    <div className="row-label" style={{ marginBottom: 8 }}>
                        <span className="row-label-title">Gradient Norms per Layer</span>
                    </div>
                    {activeLayerGrads.length > 0 ? (
                        <GradientChart data={activeLayerGrads} />
                    ) : (
                        <div style={{ padding: 20, textAlign: "center", opacity: 0.6 }}>
                            Start training to see gradients
                        </div>
                    )}
                </div>
                <div>
                     <div className="row-label" style={{ marginBottom: 8 }}>
                        <span className="row-label-title">Loss Landscape Trajectory (2D Projection)</span>
                    </div>
                    {trajectoryHistory.length > 0 ? (
                        <LossLandscape data={trajectoryHistory} />
                    ) : (
                         <div style={{ padding: 20, textAlign: "center", opacity: 0.6 }}>
                            Start training to see trajectory
                        </div>
                    )}
                </div>
            </div>
            
            <div style={{ marginTop: 20, borderTop: "1px solid var(--stroke-subtle)", paddingTop: 16 }}>
                <details>
                    <summary style={{ cursor: "pointer", color: "var(--ink-2)", fontSize: "0.9rem" }}>
                        What am I looking at?
                    </summary>
                    <div style={{ marginTop: 12, fontSize: "0.9rem", color: "var(--ink-1)", lineHeight: 1.6 }}>
                        <p style={{ marginBottom: 8 }}><strong>Gradient Norms per Layer:</strong> This shows the magnitude (L2 norm) of the gradients for each part of the model. It tells you "how hard" each layer is trying to change.</p>
                        <ul style={{ paddingLeft: 20, marginBottom: 16 }}>
                             <li><strong>Layers 0-N:</strong> The main transformer blocks.</li>
                             <li><strong>Embedding:</strong> The input token embeddings.</li>
                             <li><strong>Head:</strong> The final output layer projecting to vocabulary.</li>
                             <li><strong>Norm:</strong> The Layer Normalization parameters (scale/bias) that help stabilize training.</li>
                        </ul>
                        
                        <p style={{ marginBottom: 8 }}><strong>Loss Landscape Trajectory:</strong> Neural networks live in a massive multi-dimensional space (millions of parameters). We can't see that, so we cheat:</p>
                        <ul style={{ paddingLeft: 20 }}>
                             <li>We pick two <strong>random, fixed directions</strong> (random vectors) at the start.</li>
                             <li>These become our X and Y axes (unitless dimensions).</li>
                             <li>As the model trains, we project its current weights onto this 2D plane.</li>
                             <li>The resulting line shows the path the model is taking through the "loss landscape" as it tries to find a minimum (a valley of low loss).</li>
                        </ul>
                    </div>
                </details>
            </div>
        </div>

      </section>

      <section id="inspect-batch" className="section scroll-section">
        <div className="section-title">
          <h2>Inspect Batch</h2>
          <p>Peek at tokens, next-token predictions, and attention.</p>
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
                <label>Input Tokens</label>
                <div className="card" style={{ boxShadow: "none", background: "var(--card-muted)" }}>
                  <TokenRainbow tokens={inspectData.token_labels} />
                </div>
              </div>
              <div>
                <label>Target (Next Token)</label>
                <div className="card" style={{ boxShadow: "none", background: "var(--card-muted)" }}>
                  <div className="inspect-target">{inspectData.target_token || "-"}</div>
                  {inspectData.actual_rank !== null && inspectData.actual_rank !== undefined && (
                    <div className="inspect-row inspect-rank" style={{ marginTop: 10 }}>
                      <span className="inspect-rank-label">Rank #{inspectData.actual_rank}</span>
                      <span className="inspect-value inspect-rank-value">
                        {((inspectData.actual_prob || 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                  )}
                  <div className="inspect-preds">
                    <div className="inspect-subtitle">Top 10 predictions</div>
                    {inspectData.top_predictions?.map((pred, idx) => (
                      <div key={`${pred.token}-${idx}`} className="inspect-row">
                        <span className="inspect-token">{pred.token}</span>
                        <span className="inspect-value">{(pred.prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <p>Select a sample to inspect tokens and predictions.</p>
          )}

          <div style={{ marginTop: 16 }}>
            <div className="grid-2">
              <RangeSlider
                label="Layer"
                min={0}
                max={maxLayerIndex}
                value={attnLayer}
                onChange={setAttnLayer}
                disabled={!job || isRunning}
              />
              <RangeSlider
                label="Head"
                min={0}
                max={maxHeadIndex}
                value={attnHead}
                onChange={setAttnHead}
                disabled={!job || isRunning}
              />
            </div>

              <Heatmap matrix={attention} labels={inspectData?.token_labels || []} />

          </div>
          
           <div style={{ marginTop: 20, borderTop: "1px solid var(--stroke-subtle)", paddingTop: 16 }}>
                <details>
                    <summary style={{ cursor: "pointer", color: "var(--ink-2)", fontSize: "0.9rem" }}>
                        What am I looking at?
                    </summary>
                     <div style={{ marginTop: 12, fontSize: "0.9rem", color: "var(--ink-1)", lineHeight: 1.6 }}>
                        <p style={{ marginBottom: 8 }}><strong>Input Tokens:</strong> The unique integer IDs (and colored labels) for each part of the text. Models don't see words; they see tokens.</p>
                        <p style={{ marginBottom: 8 }}><strong>Target:</strong> The actual next token in the training data that the model is trying to predict.</p>
                        <p style={{ marginBottom: 8 }}><strong>Predictions:</strong> The top 10 tokens the model <em>thought</em> should come next, and their probabilities.</p>
                        <p style={{ marginBottom: 8 }}><strong>Attention Heatmap:</strong> This grid shows how much the model attends to past tokens when processing the current token.</p>
                        <ul style={{ paddingLeft: 20 }}>
                             <li><strong>Vertical Axis (Query):</strong> The token currently being processed.</li>
                             <li><strong>Horizontal Axis (Key):</strong> The past tokens it is "looking back" at.</li>
                             <li><strong>Bright Cells:</strong> Strong attention relationships.</li>
                        </ul>
                    </div>
                </details>
            </div>
        </div>

      </section>

      <section id="eval-history" className="section scroll-section">
        <div className="section-title">
          <h2>Evaluation</h2>
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
              { dataKey: "train", name: "Train Loss", color: "var(--accent)" },
              { dataKey: "val", name: "Val Loss", color: "#fbbf24" },
            ]}
          />
          
         <div style={{ marginTop: 20, borderTop: "1px solid var(--stroke-subtle)", paddingTop: 16 }}>
            <details>
                <summary style={{ cursor: "pointer", color: "var(--ink-2)", fontSize: "0.9rem" }}>
                    What am I looking at?
                </summary>
                <div style={{ marginTop: 12, fontSize: "0.9rem", color: "var(--ink-1)", lineHeight: 1.6 }}>
                    <p style={{ marginBottom: 8 }}><strong>Train Loss:</strong> The error on the data the model is actively learning from.</p>
                    <p style={{ marginBottom: 8 }}><strong>Val (Validation) Loss:</strong> The error on data the model has NEVER seen before.</p>
                    <p style={{ marginBottom: 8 }}><strong>Why compare them?</strong></p>
                    <ul style={{ paddingLeft: 20 }}>
                            <li>If <strong>Val Loss</strong> starts going UP while <strong>Train Loss</strong> goes DOWN, the model is significantly <strong>overfitting</strong> (memorizing, not learning).</li>
                            <li>Ideally, both go down together.</li>
                    </ul>
                </div>
            </details>
        </div>
        </div>

      </section>

      <section id="logs" className="section scroll-section">
        <div className="section-title">
          <h2>Logs</h2>
          {/* <p>Checkpoint and status events.</p> */}
        </div>
        <div className="card">
          <LogBox logs={logs} />
        </div>
      </section>
      </div>
    </div>

      <Modal isOpen={previewModalOpen} onClose={closePreview} title={previewContent?.name || "Preview"}>
        {previewLoading ? (
          <p>Loading...</p>
        ) : previewContent ? (
          <div className="text-preview">{previewContent.content}</div>
        ) : null}
      </Modal>
    </>
  );
}
