"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import LineChart from "../../components/LineChart";
import MarkdownBlock from "../../components/MarkdownBlock";
import ModelConfigSummary from "../../components/ModelConfigSummary";
import RangeSlider from "../../components/RangeSlider";
import SideNav from "../../components/SideNav";
import StatCard from "../../components/StatCard";
import { fetchJson, Checkpoint } from "../../lib/api";
import { useDemoMode } from "../../lib/demo";
import { grpoEquations } from "../../lib/equations";
import { formatCheckpointTimestamp } from "../../lib/time";
import { useScrollSpy } from "../../lib/useScrollSpy";

type Answer = {
  text: string;
  tokens: string[];
  n_tokens: number;
  reward: number;
  format_ok: boolean;
  correct: boolean;
  advantage: number;
  mean_logp?: number;
  kl?: number;
};

type Group = {
  answers: Answer[];
  summary: {
    question: string;
    a: number;
    b: number;
    target: number;
    mean: number;
    std: number;
    format_rate: number;
    correct_rate: number;
    gen_secs: number;
  };
};

type RoundResult = {
  groups: Group[];
  update: { loss: number; kl: number; grad_norm: number } | null;
  step: number;
  trained: boolean;
  format_rate: number;
  correct_rate: number;
  mean_reward: number;
};

type Session = {
  session_id: string;
  tokenizer_type: string;
  param_count_m: number;
  cfg: Record<string, number | string | boolean>;
  is_finetuned: boolean;
  example_prompt: string;
  example_answer: string;
};

type EvalResult = {
  which: "policy" | "ref";
  held_out: boolean;
  format_rate: number;
  correct_rate: number;
  n: number;
};

type UpdatePoint = {
  step: number;
  mean_reward: number;
  format_rate: number;
  correct_rate: number;
  kl: number;
};

const rlSections = [
  { id: "checkpoint", label: "Checkpoint" },
  { id: "settings", label: "Settings" },
  { id: "understand", label: "Understand" },
  { id: "train", label: "Train" },
  { id: "rounds", label: "Latest round" },
  { id: "progress", label: "Progress" },
  { id: "held-out", label: "Held-out check" },
];

const STAGE_LABEL: Record<string, string> = {
  pretrain: "Pre-trained",
  sft: "Fine-tuned (SFT)",
  rl: "RL",
};

const noteStyle = {
  marginTop: 12,
  padding: "8px 12px",
  border: "1px solid var(--stroke-strong)",
  borderRadius: 8,
};

const errorStyle = { ...noteStyle, borderColor: "rgba(239, 68, 68, 0.7)", color: "#fca5a5" };

function advantageColor(adv: number) {
  const strength = Math.min(Math.abs(adv) / 2, 1);
  const alpha = 0.12 + 0.5 * strength;
  return adv >= 0
    ? `rgba(34, 197, 94, ${alpha})`
    : `rgba(239, 68, 68, ${alpha})`;
}

function pct(x: number | undefined) {
  return x === undefined ? "-" : `${(x * 100).toFixed(1)}%`;
}

function trailingMean(values: number[], i: number, window: number) {
  const slice = values.slice(Math.max(0, i - window + 1), i + 1);
  return slice.reduce((s, v) => s + v, 0) / slice.length;
}

export default function RLPage() {
  const isDemo = useDemoMode();
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [learningRate, setLearningRate] = useState(3e-5);
  const [beta, setBeta] = useState(0.02);
  const [digits, setDigits] = useState(3);
  const [problems, setProblems] = useState(8);
  const [group, setGroup] = useState(8);
  const [temperature, setTemperature] = useState(1.0);
  const [maxNew, setMaxNew] = useState(16);
  const [nUpdates, setNUpdates] = useState(25);
  const [fixedA, setFixedA] = useState("");
  const [fixedB, setFixedB] = useState("");

  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<{
    done: number;
    total: number;
    label: string;
  } | null>(null);
  const stopRef = useRef(false);
  const [latest, setLatest] = useState<RoundResult | null>(null);
  const [history, setHistory] = useState<UpdatePoint[]>([]);
  const [evals, setEvals] = useState<EvalResult[]>([]);
  const [savedPath, setSavedPath] = useState<string | null>(null);

  const { activeSection, setActiveSection } = useScrollSpy(rlSections);

  // Scores taken under other settings are not comparable, so drop them.
  useEffect(() => {
    setEvals([]);
  }, [digits, temperature, maxNew]);

  useEffect(() => {
    fetchJson<{ checkpoints: Checkpoint[] }>("/api/checkpoints")
      .then((data) => setCheckpoints(data.checkpoints))
      .catch((err) => setError((err as Error).message));
  }, []);

  // Fine-tuned checkpoints first: the RL tab expects a model that has already
  // learned the answer format.
  const sortedCheckpoints = useMemo(() => {
    const rank = (c: Checkpoint) =>
      c.stage === "sft" ? 0 : c.stage === "rl" ? 1 : 2;
    return checkpoints
      .slice()
      .sort((a, b) => rank(a) - rank(b) || b.mtime - a.mtime);
  }, [checkpoints]);

  const post = <T,>(path: string, body: object = {}) =>
    fetchJson<T>(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

  const readError = (err: unknown) => {
    const message = (err as Error).message;
    try {
      return JSON.parse(message).detail ?? message;
    } catch {
      return message;
    }
  };

  const loadModel = async () => {
    if (!selectedCheckpoint) return;
    setLoading(true);
    setError(null);
    setSession(null);
    setLatest(null);
    setHistory([]);
    setEvals([]);
    setSavedPath(null);
    try {
      const data = await post<Session>("/api/rl/sessions", {
        checkpoint_id: selectedCheckpoint,
        learning_rate: learningRate,
        beta,
      });
      setSession(data);
    } catch (err) {
      setError(readError(err));
    } finally {
      setLoading(false);
    }
  };

  const roundBody = (train: boolean) => {
    const body: Record<string, unknown> = {
      train,
      digits,
      group,
      problems,
      max_new_tokens: maxNew,
      temperature,
      beta,
    };
    if (!train && fixedA !== "" && fixedB !== "") {
      body.a = Number(fixedA);
      body.b = Number(fixedB);
    }
    return body;
  };

  const runRounds = async (train: boolean, count: number) => {
    if (!session) return;
    setBusy(true);
    setError(null);
    stopRef.current = false;
    try {
      for (let i = 0; i < count; i += 1) {
        if (stopRef.current) break;
        setProgress({
          done: i,
          total: count,
          label: train ? `update ${i + 1} of ${count}` : "sampling",
        });
        const result = await post<RoundResult>(
          `/api/rl/sessions/${session.session_id}/round`,
          roundBody(train),
        );
        setLatest(result);
        if (result.trained && result.update) {
          const point: UpdatePoint = {
            step: result.step,
            mean_reward: result.mean_reward,
            format_rate: result.format_rate,
            correct_rate: result.correct_rate,
            kl: result.update.kl,
          };
          setHistory((prev) => [...prev, point]);
        }
      }
    } catch (err) {
      setError(readError(err));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const resetWeights = async () => {
    if (!session) return;
    setBusy(true);
    try {
      await post(`/api/rl/sessions/${session.session_id}/reset`);
      setHistory([]);
      setLatest(null);
      setEvals((prev) => prev.filter((e) => e.which === "ref"));
    } catch (err) {
      setError(readError(err));
    } finally {
      setBusy(false);
    }
  };

  const saveCheckpoint = async () => {
    if (!session) return;
    setBusy(true);
    try {
      const data = await post<{ path: string }>(
        `/api/rl/sessions/${session.session_id}/save`,
      );
      setSavedPath(data.path);
      const list = await fetchJson<{ checkpoints: Checkpoint[] }>(
        "/api/checkpoints",
      );
      setCheckpoints(list.checkpoints);
    } catch (err) {
      setError(readError(err));
    } finally {
      setBusy(false);
    }
  };

  const runHeldOut = async () => {
    if (!session) return;
    setBusy(true);
    setError(null);
    // The frozen starting model never changes, so measure it only once.
    const haveRef = (heldOut: boolean) =>
      evals.some((e) => e.which === "ref" && e.held_out === heldOut);
    const jobs: { which: "policy" | "ref"; held_out: boolean }[] = [];
    for (const heldOut of [true, false]) {
      if (!haveRef(heldOut)) jobs.push({ which: "ref", held_out: heldOut });
      jobs.push({ which: "policy", held_out: heldOut });
    }
    const results = evals.filter((e) => e.which === "ref");
    try {
      for (let i = 0; i < jobs.length; i += 1) {
        const job = jobs[i];
        setProgress({
          done: i,
          total: jobs.length,
          label: `${job.which === "ref" ? "starting model" : "current model"}, ${job.held_out ? "unseen" : "training"} problems`,
        });
        const r = await post<EvalResult>(
          `/api/rl/sessions/${session.session_id}/evaluate`,
          {
            digits,
            samples: 4,
            temperature,
            max_new_tokens: maxNew,
            ...job,
          },
        );
        results.push(r);
      }
      setEvals(results);
    } catch (err) {
      setError(readError(err));
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const findEval = (which: "policy" | "ref", heldOut: boolean) =>
    evals.find((e) => e.which === which && e.held_out === heldOut);

  const chartData = useMemo(() => {
    const correct = history.map((h) => h.correct_rate);
    const fmt = history.map((h) => h.format_rate);
    return history.map((h, i) => ({
      step: h.step,
      correct: correct[i],
      correct_avg: trailingMean(correct, i, 10),
      formatted_avg: trailingMean(fmt, i, 10),
      kl: h.kl,
    }));
  }, [history]);

  const disabled = isDemo || !session || busy;

  return (
    <div className="page-with-nav">
      <SideNav
        sections={rlSections}
        activeId={activeSection}
        onNavigate={setActiveSection}
        ariaLabel="Reinforcement learning sections"
      />
      <div className="page-content">
        <section id="checkpoint" className="section scroll-section">
          <div className="section-title">
            <h2>Select Model</h2>
            <p>
              Pick a checkpoint to improve with reinforcement learning (GRPO).
            </p>
          </div>
          <div className="card">
            <p style={{ marginTop: 0 }}>
              RL needs a model that already answers in the right format and is
              right some of the time. GRPO learns only from differences between
              answers to the same question: if every answer is wrong, or every
              answer is right, there is nothing to learn from. So first go to{" "}
              <Link href="/finetune" className="text-link">
                Fine-Tuning
              </Link>
              , pick a checkpoint trained with the GPT-2 (tiktoken) tokenizer,
              and choose the <b>Arithmetic</b> dataset. Then load the result
              here.
            </p>
            <div className="inline-row">
              <select
                value={selectedCheckpoint}
                onChange={(event) => setSelectedCheckpoint(event.target.value)}
                disabled={busy || loading}
              >
                <option value="">Select checkpoint</option>
                {sortedCheckpoints.map((ckpt) => (
                  <option key={ckpt.id} value={ckpt.id}>
                    {
                      STAGE_LABEL[
                        ckpt.stage ?? (ckpt.is_finetuned ? "sft" : "pretrain")
                      ]
                    }{" "}
                    · {formatCheckpointTimestamp(new Date(ckpt.mtime * 1000))} ·{" "}
                    {ckpt.run_id}/{ckpt.name}
                  </option>
                ))}
              </select>
              <button
                className="primary"
                onClick={loadModel}
                disabled={isDemo || !selectedCheckpoint || loading || busy}
              >
                {loading ? "Loading..." : session ? "Reload" : "Load"}
              </button>
            </div>
            {isDemo && (
              <p className="badge demo-badge">
                Demo mode: RL training disabled.
              </p>
            )}
            {error && (
              <p
                
                style={errorStyle}
              >
                {error}
              </p>
            )}
            {session && (
              <div style={{ marginTop: 16 }}>
                <ModelConfigSummary
                  config={session.cfg}
                  summaryItems={[
                    { label: "params", value: `${session.param_count_m}M` },
                    { label: "tokenizer", value: session.tokenizer_type },
                    { label: "n_layers", value: session.cfg.n_layers || "-" },
                  ]}
                />
                <p style={{ marginBottom: 0 }}>
                  Every question looks like this, and a correct answer like
                  this:
                </p>
                <pre className="code-block">
                  {session.example_prompt + session.example_answer}
                </pre>
                {!session.is_finetuned && (
                  <p style={noteStyle}>
                    This checkpoint has not been fine-tuned. It has probably
                    never seen the answer format, so every answer will score the
                    same and GRPO will have nothing to learn from.
                  </p>
                )}
              </div>
            )}
          </div>
        </section>

        <section id="settings" className="section scroll-section">
          <div className="section-title">
            <h2>Settings</h2>
            <p>What it trains on, and how hard it pushes.</p>
          </div>
          <div className="card">
            <div className="grid-2">
              <RangeSlider
                label="Digits per number"
                value={digits}
                min={1}
                max={4}
                onChange={setDigits}
                valueLabel={`up to ${digits}`}
              />
              <RangeSlider
                label="Problems per update"
                value={problems}
                min={1}
                max={16}
                onChange={setProblems}
                valueLabel={String(problems)}
              />
              <RangeSlider
                label="Answers per problem (group size G)"
                value={group}
                min={2}
                max={16}
                onChange={setGroup}
                valueLabel={String(group)}
              />
              <RangeSlider
                label="Temperature"
                value={temperature}
                min={0.3}
                max={1.6}
                step={0.05}
                onChange={setTemperature}
                valueLabel={temperature.toFixed(2)}
              />
              <RangeSlider
                label={
                  <>
                    KL penalty (<span style={{ textTransform: "none" }}>β</span>)
                  </>
                }
                value={beta}
                min={0}
                max={0.2}
                step={0.005}
                onChange={setBeta}
                valueLabel={beta.toFixed(3)}
              />
              <RangeSlider
                label="Max new tokens"
                value={maxNew}
                min={8}
                max={48}
                step={2}
                onChange={setMaxNew}
                valueLabel={String(maxNew)}
              />
              <div>
                <label>Learning rate (applies when you load)</label>
                <select
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                >
                  <option value={1e-5}>1e-5</option>
                  <option value={3e-5}>3e-5 (default)</option>
                  <option value={1e-4}>1e-4</option>
                  <option value={3e-4}>3e-4 (watch it collapse)</option>
                </select>
              </div>
            </div>
            <p style={{ marginBottom: 0 }}>
              <b>Problems per update</b> matters more than it looks. With one
              problem per update, the gradient is mostly noise: the optimiser
              takes a full-sized step in a random direction, and the model
              drifts without improving. Averaging over eight problems lets the
              signal through. Try 1 and watch the drift climb while accuracy
              goes nowhere.
            </p>
          </div>
        </section>

        <section id="understand" className="section scroll-section">
          <div className="section-title">
            <h2>Understand</h2>
            <p>The group is its own yardstick.</p>
          </div>
          <div className="card">
            <details className="expander">
              <summary>How a round works</summary>
              <div className="expander-content">
                <ol>
                  <li>
                    Draw fresh random problems. Training never repeats a fixed
                    question.
                  </li>
                  <li>
                    Ask the model each question G times, sampling at the
                    temperature above.
                  </li>
                  <li>
                    Score each answer with a program, not a learned model: 0.2
                    if the right number appears anywhere, 0.3 for using the{" "}
                    <code>&lt;answer&gt;</code> tags, and 1.0 more if the number
                    in the tags is right.
                  </li>
                  <li>
                    Compare each score with its group&apos;s mean. Above the
                    mean is a positive advantage.
                  </li>
                  <li>
                    Push up every token of the above-average answers and push
                    down every token of the below-average ones, with a penalty
                    for drifting from the starting model.
                  </li>
                </ol>
              </div>
            </details>
            <details className="expander">
              <summary>Equations</summary>
              <div className="expander-content">
                <MarkdownBlock content={grpoEquations} />
              </div>
            </details>
          </div>
        </section>

        <section id="train" className="section scroll-section">
          <div className="section-title">
            <h2>Train</h2>
            <p>One update at a time, so you can watch each one.</p>
          </div>
          <div className="card">
            <div className="inline-row" style={{ marginBottom: 12 }}>
              <button
                className="secondary"
                onClick={() => runRounds(false, 1)}
                disabled={disabled}
              >
                Sample only
              </button>
              <button
                className="primary"
                onClick={() => runRounds(true, 1)}
                disabled={disabled}
              >
                Train 1 update
              </button>
              <button
                className="primary"
                onClick={() => runRounds(true, nUpdates)}
                disabled={disabled}
              >
                Train {nUpdates} updates
              </button>
              {busy && (
                <button
                  className="secondary"
                  onClick={() => (stopRef.current = true)}
                >
                  Stop
                </button>
              )}
              <button
                className="secondary"
                onClick={resetWeights}
                disabled={disabled}
              >
                Reset weights
              </button>
              <button
                className="secondary"
                onClick={saveCheckpoint}
                disabled={disabled || history.length === 0}
              >
                Save checkpoint
              </button>
            </div>
            <RangeSlider
              label="Updates per run"
              value={nUpdates}
              min={5}
              max={200}
              step={5}
              onChange={setNUpdates}
              valueLabel={String(nUpdates)}
            />
            <div className="grid-2" style={{ marginTop: 12 }}>
              <div>
                <label>Try one question (Sample only): a</label>
                <input
                  type="number"
                  value={fixedA}
                  onChange={(e) => setFixedA(e.target.value)}
                  placeholder="428"
                />
              </div>
              <div>
                <label>b</label>
                <input
                  type="number"
                  value={fixedB}
                  onChange={(e) => setFixedB(e.target.value)}
                  placeholder="357"
                />
              </div>
            </div>
            <p style={{ marginBottom: 0 }}>
              <b>Sample only</b> asks and scores, then stops: the weights do not
              change. A fixed question is used only here. Training always draws
              fresh problems, because repeating one question teaches an answer,
              not a skill.
            </p>
            {progress && (
              <div style={{ marginTop: 12 }}>
                <div className="flex-between" style={{ marginBottom: 6 }}>
                  <span className="badge">{progress.label}</span>
                  <span className="badge">
                    {progress.done} / {progress.total}
                  </span>
                </div>
                <div className="progress">
                  <span
                    style={{
                      width: `${(progress.done / progress.total) * 100}%`,
                    }}
                  />
                </div>
              </div>
            )}
            {savedPath && (
              <p style={noteStyle}>
                Saved to {savedPath}. It now appears on the Inference tab.
              </p>
            )}
          </div>
        </section>

        <section id="rounds" className="section scroll-section">
          <div className="section-title">
            <h2>Latest Round</h2>
            <p>Every answer, every token, and which way it gets pushed.</p>
          </div>
          <div className="card">
            {!latest ? (
              <p style={{ margin: 0 }}>
                Press Sample only or Train to see a round.
              </p>
            ) : (
              <>
                <div className="grid-3" style={{ marginBottom: 12 }}>
                  <StatCard
                    label="Mean reward"
                    value={latest.mean_reward.toFixed(3)}
                  />
                  <StatCard label="Formatted" value={pct(latest.format_rate)} />
                  <StatCard label="Correct" value={pct(latest.correct_rate)} />
                </div>
                {latest.update && (
                  <div className="grid-3" style={{ marginBottom: 12 }}>
                    <StatCard
                      label="Loss"
                      value={latest.update.loss.toFixed(4)}
                    />
                    <StatCard
                      label="Drift from start (KL)"
                      value={latest.update.kl.toFixed(4)}
                    />
                    <StatCard
                      label="Grad norm"
                      value={latest.update.grad_norm.toFixed(3)}
                    />
                  </div>
                )}
                {latest.groups.map((g, gi) => (
                  <details key={gi} className="expander" open={gi === 0}>
                    <summary>
                      {g.summary.question.replace(/\n/g, " ")} →{" "}
                      {g.summary.target} ·{" "}
                      {g.answers.filter((a) => a.correct).length}/
                      {g.answers.length} correct · group mean{" "}
                      {g.summary.mean.toFixed(2)}
                    </summary>
                    <div className="expander-content">
                      {g.summary.std === 0 && (
                        <p style={noteStyle}>
                          Every answer scored the same, so every advantage is
                          zero. This group teaches nothing.
                        </p>
                      )}
                      {g.answers.map((a, ai) => (
                        <div
                          key={ai}
                          className="flex-between"
                          style={{
                            gap: 12,
                            padding: "4px 0",
                            borderBottom: "1px solid var(--stroke)",
                          }}
                        >
                          <div
                            style={{
                              fontFamily:
                                "var(--font-mono), IBM Plex Mono, SFMono-Regular, Menlo, monospace",
                              lineHeight: 1.9,
                            }}
                          >
                            {a.tokens.map((t, ti) => (
                              <span
                                key={ti}
                                title={`token ${ti}: ${JSON.stringify(t)}`}
                                style={{
                                  background: advantageColor(a.advantage),
                                  padding: "2px 4px",
                                  marginRight: 2,
                                  borderRadius: 4,
                                  whiteSpace: "pre",
                                }}
                              >
                                {t}
                              </span>
                            ))}
                          </div>
                          <span
                            className="badge"
                            style={{ whiteSpace: "nowrap" }}
                          >
                            {a.correct ? "✓" : a.format_ok ? "✗" : "no tags"} ·
                            r={a.reward.toFixed(1)} · A=
                            {a.advantage >= 0 ? "+" : ""}
                            {a.advantage.toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </details>
                ))}
                <p style={{ marginBottom: 0 }}>
                  <b>The colour is the advantage.</b> Every token in one answer
                  has the same colour: green answers beat their group and all
                  their tokens get pushed up; red ones get pushed down,
                  including tokens that were fine. GRPO never asks which token
                  made an answer right. Across many rounds, tokens that keep
                  showing up in good answers are pushed up more often than down.
                </p>
              </>
            )}
          </div>
        </section>

        <section id="progress" className="section scroll-section">
          <div className="section-title">
            <h2>Progress</h2>
            <p>On the training problems, update by update.</p>
          </div>
          <div className="card">
            <div className="grid-3" style={{ marginBottom: 12 }}>
              <StatCard
                label="Updates applied"
                value={
                  history.length > 0 ? history[history.length - 1].step : 0
                }
              />
              <StatCard
                label="Correct (last 10)"
                value={
                  chartData.length
                    ? pct(chartData[chartData.length - 1].correct_avg)
                    : "-"
                }
              />
              <StatCard
                label="Drift from start (KL)"
                value={
                  history.length
                    ? history[history.length - 1].kl.toFixed(4)
                    : "-"
                }
              />
            </div>
            <LineChart
              data={chartData}
              xKey="step"
              xLabel="update"
              yDomain={[0, 1]}
              lines={[
                {
                  dataKey: "correct",
                  name: "Correct",
                  color: "rgba(249, 115, 22, 0.35)",
                },
                {
                  dataKey: "correct_avg",
                  name: "Correct (avg of 10)",
                  color: "var(--accent)",
                },
                {
                  dataKey: "formatted_avg",
                  name: "Formatted (avg of 10)",
                  color: "#60a5fa",
                },
              ]}
            />
            <LineChart
              data={chartData}
              xKey="step"
              xLabel="update"
              lines={[
                {
                  dataKey: "kl",
                  name: "Drift from start (KL)",
                  color: "#a78bfa",
                },
              ]}
            />
            <p style={{ marginBottom: 0 }}>
              These are the problems the model is training on, so they are not a
              clean measure of learning. Every update draws new ones, and some
              are harder than others, so the faint line jumps around. Use the
              held-out check below to see whether the model really improved.
            </p>
          </div>
        </section>

        <section id="held-out" className="section scroll-section">
          <div className="section-title">
            <h2>Held-out Check</h2>
            <p>Problems neither SFT nor RL ever saw.</p>
          </div>
          <div className="card">
            <p style={{ marginTop: 0 }}>
              About one problem in ten is set aside by a fixed rule, and neither
              the arithmetic dataset nor RL training ever uses it. One press
              scores 64 of those, and 64 problems from the training pool for
              contrast, each answered 4 times. It scores both the starting
              model, which is still in memory as the frozen copy the KL penalty
              compares against, and the model as it is now.
            </p>
            <button
              className="primary"
              onClick={runHeldOut}
              disabled={disabled}
            >
              Run held-out check
            </button>
            {evals.length > 0 && (
              <table className="table" style={{ marginTop: 16 }}>
                <thead>
                  <tr>
                    <th>Problems</th>
                    <th>Starting model</th>
                    <th>Current model</th>
                  </tr>
                </thead>
                <tbody>
                  {[true, false].map((heldOut) => {
                    const ref = findEval("ref", heldOut);
                    const cur = findEval("policy", heldOut);
                    return (
                      <tr key={String(heldOut)}>
                        <td>
                          {heldOut
                            ? "Unseen (held out)"
                            : "From the training pool"}
                        </td>
                        <td>
                          {ref ? `${pct(ref.correct_rate)} correct` : "-"}
                        </td>
                        <td>
                          {cur ? `${pct(cur.correct_rate)} correct` : "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
            <p style={{ marginBottom: 0 }}>
              Gains on unseen problems mean the model got better at adding.
              Gains only on training-pool problems could just be memorising.
              Each number comes from 256 answers, so differences of a few
              percentage points are noise.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
