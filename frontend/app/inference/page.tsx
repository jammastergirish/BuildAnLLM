"use client";

import { useEffect, useState } from "react";
import CodePanel from "../../components/CodePanel";
import Heatmap from "../../components/Heatmap";
import LineChart from "../../components/LineChart";
import StatCard from "../../components/StatCard";
import { Checkpoint, CodeSnippet, fetchJson } from "../../lib/api";
import MarkdownBlock from "../../components/MarkdownBlock";
import { inferenceEquations } from "../../lib/equations";

type SessionInfo = {
  session_id: string;
  tokenizer_type: string;
  param_count_m: number;
  cfg: Record<string, number | string>;
};

type DiagnosticsMeta = {
  diagnostic_id: string;
  token_ids: number[];
  token_labels: string[];
};

export default function InferencePage() {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [prompt, setPrompt] = useState("First Citizen:");
  const [maxNewTokens, setMaxNewTokens] = useState(200);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState<number | "">("");
  const [topP, setTopP] = useState(0.9);
  const [generatedText, setGeneratedText] = useState("");
  const [diagnostics, setDiagnostics] = useState<DiagnosticsMeta | null>(null);
  const [attentionMatrix, setAttentionMatrix] = useState<number[][]>([]);
  const [layerNorms, setLayerNorms] = useState<Array<{ layer: number; avg_norm: number }>>([]);
  const [logitLens, setLogitLens] = useState<Array<{ layer: number; predictions: { rank: number; token: string; prob: number }[] }>>([]);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [selectedPosition, setSelectedPosition] = useState(0);
  const [snippets, setSnippets] = useState<CodeSnippet[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJson<{ checkpoints: Checkpoint[] }>("/api/checkpoints")
      .then((data) => setCheckpoints(data.checkpoints))
      .catch((err) => setError((err as Error).message));
  }, []);

  const loadSession = async () => {
    setError(null);
    try {
      if (!selectedCheckpoint) {
        throw new Error("Select a checkpoint first.");
      }
      const data = await fetchJson<SessionInfo>("/api/inference/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint_id: selectedCheckpoint }),
      });
      setSession(data);
      setGeneratedText("");
      setDiagnostics(null);
      setAttentionMatrix([]);
      setLayerNorms([]);
      setLogitLens([]);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const generate = async () => {
    if (!session) return;
    setError(null);
    try {
      const data = await fetchJson<{ generated_text: string }>(
        `/api/inference/sessions/${session.session_id}/generate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt,
            max_new_tokens: maxNewTokens,
            temperature,
            top_k: topK === "" ? null : topK,
            top_p: topP,
          }),
        }
      );
      setGeneratedText(data.generated_text);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const runDiagnostics = async () => {
    if (!session) return;
    setError(null);
    try {
      const data = await fetchJson<DiagnosticsMeta>(
        `/api/inference/sessions/${session.session_id}/diagnostics`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        }
      );
      setDiagnostics(data);
      setSelectedLayer(0);
      setSelectedHead(0);
      setSelectedPosition(Math.max(0, data.token_ids.length - 2));
    } catch (err) {
      setError((err as Error).message);
    }
  };

  useEffect(() => {
    if (!diagnostics) return;
    fetchJson<{ attention: number[][] }>(
      `/api/inference/diagnostics/${diagnostics.diagnostic_id}/attention?layer=${selectedLayer}&head=${selectedHead}`
    )
      .then((data) => setAttentionMatrix(data.attention))
      .catch((err) => setError((err as Error).message));
  }, [diagnostics, selectedLayer, selectedHead]);

  useEffect(() => {
    if (!diagnostics) return;
    fetchJson<{ layers: Array<{ layer: number; predictions: { rank: number; token: string; prob: number }[] }> }>(
      `/api/inference/diagnostics/${diagnostics.diagnostic_id}/logit-lens?position=${selectedPosition}`
    )
      .then((data) => setLogitLens(data.layers))
      .catch((err) => setError((err as Error).message));
  }, [diagnostics, selectedPosition]);

  useEffect(() => {
    if (!diagnostics) return;
    fetchJson<{ layers: Array<{ layer: number; avg_norm: number }> }>(
      `/api/inference/diagnostics/${diagnostics.diagnostic_id}/layer-norms`
    )
      .then((data) => setLayerNorms(data.layers))
      .catch((err) => setError((err as Error).message));
  }, [diagnostics]);

  const loadSnippets = async () => {
    setError(null);
    try {
      const data = await fetchJson<{ snippets: CodeSnippet[] }>("/api/docs/inference-code");
      setSnippets(data.snippets);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const layersCount = session ? Number(session.cfg?.n_layers || 0) : 0;
  const headsCount = session ? Number(session.cfg?.n_heads || 0) : 0;

  return (
    <>
      <section className="section">
        <div className="section-title">
          <h2>Select Model</h2>
          <p>Load a checkpoint for inference.</p>
        </div>
        <div className="card">
          <div className="inline-row" style={{ alignItems: "center" }}>
            <select value={selectedCheckpoint} onChange={(event) => setSelectedCheckpoint(event.target.value)}>
              <option value="">Select checkpoint</option>
              {checkpoints.map((ckpt) => (
                <option key={ckpt.id} value={ckpt.id}>
                  {ckpt.name}
                </option>
              ))}
            </select>
            <button className="primary" onClick={loadSession}>Load Model</button>
          </div>

          {session && (
            <div className="grid-3" style={{ marginTop: 16 }}>
              <StatCard label="Tokenizer" value={session.tokenizer_type} />
              <StatCard label="Parameters" value={`${session.param_count_m}M`} />
              <StatCard label="d_model" value={session.cfg?.d_model || "-"} />
              <StatCard label="n_layers" value={session.cfg?.n_layers || "-"} />
              <StatCard label="n_heads" value={session.cfg?.n_heads || "-"} />
              <StatCard label="n_ctx" value={session.cfg?.n_ctx || "-"} />
            </div>
          )}
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Generation Settings</h2>
          <p>Sampling controls for text generation.</p>
        </div>
        <div className="card">
          <div className="grid-2">
            <div>
              <label>Prompt</label>
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
            </div>
            <div className="grid-3">
              <div>
                <label>Max New Tokens</label>
                <input type="number" value={maxNewTokens} onChange={(event) => setMaxNewTokens(Number(event.target.value))} />
              </div>
              <div>
                <label>Temperature</label>
                <input type="number" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
              </div>
              <div>
                <label>Top-k</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(event) => setTopK(event.target.value === "" ? "" : Number(event.target.value))}
                />
              </div>
              <div>
                <label>Top-p</label>
                <input type="number" step="0.05" value={topP} onChange={(event) => setTopP(Number(event.target.value))} />
              </div>
            </div>
          </div>
          <div className="inline-row" style={{ marginTop: 12 }}>
            <button className="primary" onClick={generate} disabled={!session}>Generate</button>
            <button className="secondary" onClick={runDiagnostics} disabled={!session}>Run Diagnostics</button>
            <button className="secondary" onClick={loadSnippets}>Load Code Snippets</button>
          </div>
          {error && <p style={{ color: "#b42318" }}>{error}</p>}
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Understand Text Generation</h2>
          <p>Sampling equations and references.</p>
        </div>
        <div className="card">
          <MarkdownBlock content={inferenceEquations} />
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Generated Text</h2>
          <p>Model output based on your prompt.</p>
        </div>
        <div className="card">
          <textarea value={generatedText} readOnly style={{ minHeight: 220 }} />
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Model Internals</h2>
          <p>Attention, logit lens, and layer norms.</p>
        </div>
        <div className="card">
          {diagnostics ? (
            <>
              <div className="grid-3">
                <div>
                  <label>Layer</label>
                  <input
                    type="number"
                    min={0}
                    max={Math.max(0, layersCount - 1)}
                    value={selectedLayer}
                    onChange={(event) => setSelectedLayer(Number(event.target.value))}
                  />
                </div>
                <div>
                  <label>Head</label>
                  <input
                    type="number"
                    min={0}
                    max={Math.max(0, headsCount - 1)}
                    value={selectedHead}
                    onChange={(event) => setSelectedHead(Number(event.target.value))}
                  />
                </div>
                <div>
                  <label>Logit Lens Position</label>
                  <input
                    type="number"
                    min={0}
                    max={Math.max(0, diagnostics.token_ids.length - 1)}
                    value={selectedPosition}
                    onChange={(event) => setSelectedPosition(Number(event.target.value))}
                  />
                </div>
              </div>

              <div style={{ marginTop: 16 }}>
                <h3>Attention Heatmap</h3>
                {attentionMatrix.length > 0 ? (
                  <Heatmap matrix={attentionMatrix} labels={diagnostics.token_labels} />
                ) : (
                  <p>Loading attention map...</p>
                )}
              </div>

              <div style={{ marginTop: 24 }}>
                <h3>Logit Lens</h3>
                <div className="card" style={{ boxShadow: "none", background: "rgba(255,255,255,0.55)" }}>
                  {logitLens.length === 0 ? (
                    <p>Loading logit lens...</p>
                  ) : (
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Layer</th>
                          <th>Top Predictions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {logitLens.map((row) => (
                          <tr key={row.layer}>
                            <td>{row.layer}</td>
                            <td>
                              {row.predictions.map((pred) => (
                                <span key={`${row.layer}-${pred.rank}`} className="badge" style={{ marginRight: 8 }}>
                                  {pred.rank}. {pred.token} ({(pred.prob * 100).toFixed(1)}%)
                                </span>
                              ))}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>

              <div style={{ marginTop: 24 }}>
                <h3>Layer Norms</h3>
                <LineChart
                  data={layerNorms.map((row) => ({ layer: row.layer, norm: row.avg_norm }))}
                  xKey="layer"
                  lines={[{ dataKey: "norm", name: "Avg Norm", color: "#0f4c5c" }]}
                />
              </div>
            </>
          ) : (
            <p>Run diagnostics to inspect attention, logit lens, and layer norms.</p>
          )}
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Code Snippets</h2>
          <p>Sampling and inference code.</p>
        </div>
        <div>
          {snippets.length === 0 ? (
            <div className="card">Load code snippets to inspect inference modules.</div>
          ) : (
            snippets.map((snippet) => <CodePanel key={snippet.title} snippet={snippet} />)
          )}
        </div>
      </section>
    </>
  );
}
