"use client";

import { useEffect, useState, useMemo } from "react";
import SideNav from "../../components/SideNav";
import { Checkpoint, fetchJson } from "../../lib/api";
import { formatCheckpointTimestamp } from "../../lib/time";
import { Check, Loader2, AlertCircle } from "lucide-react";
import { useScrollSpy } from "../../lib/useScrollSpy";
import { useDemoMode } from "../../lib/demo";

// Types
type EvalTaskInfo = {
  name: string;
  description: string;
  metrics: string[];
};

type EvalTaskMap = Record<string, EvalTaskInfo>;

type JobStatus = {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed" | "error";
  tasks: string[];
  checkpoint_id: string;
  created_at: number;
  progress?: number;
  error?: string;
  results?: Record<string, Record<string, number>>; // Dynamic metrics
};

const SECTIONS = [
  { id: "select-model", label: "Model & Tasks" },
  { id: "run-eval", label: "Run Evaluation" },
  { id: "results", label: "Results" },
];

export default function EvaluationsPage() {
  const isDemo = useDemoMode();
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");
  const [taskMap, setTaskMap] = useState<EvalTaskMap>({});
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set(["hellaswag"]));
  const [activeJobs, setActiveJobs] = useState<JobStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [completedResults, setCompletedResults] = useState<Record<string, Record<string, Record<string, number>>>>({});
  
  const { activeSection, setActiveSection } = useScrollSpy(SECTIONS);

  // Initial Data Load
  useEffect(() => {
    // Fetch checkpoints
    fetchJson<{ checkpoints: Checkpoint[] }>("/api/checkpoints")
      .then((data) => setCheckpoints(data.checkpoints))
      .catch((err) => console.error("Failed to load checkpoints:", err));

    // Fetch tasks
    fetchJson<{ tasks: EvalTaskMap }>("/api/evaluations/tasks")
      .then((data) => {
        setTaskMap(data.tasks);
      })
      .catch((err) => console.error("Failed to load tasks:", err));

    // Fetch jobs
    loadJobs();
  }, []);

  const sortedCheckpoints = useMemo(
    () => [...checkpoints].sort((a, b) => b.mtime - a.mtime),
    [checkpoints]
  );
  
  // Auto-select first checkpoint
  useEffect(() => {
    if (!selectedCheckpoint && sortedCheckpoints.length > 0) {
      setSelectedCheckpoint(sortedCheckpoints[0].id);
    }
  }, [sortedCheckpoints, selectedCheckpoint]);

  const loadJobs = async () => {
    try {
      const data = await fetchJson<{ jobs: JobStatus[] }>("/api/evaluations/jobs");
      setActiveJobs(data.jobs);
      
      // Check for completed jobs without results loaded
      data.jobs.forEach(job => {
        if (job.status === "completed" && !completedResults[job.job_id]) {
          loadResults(job.job_id);
        }
      });
    } catch (err) {
      console.error("Failed to load jobs:", err);
    }
  };

  const loadResults = async (jobId: string) => {
    try {
      const results = await fetchJson<Record<string, Record<string, number>>>(`/api/evaluations/results/${jobId}`);
      setCompletedResults(prev => ({ ...prev, [jobId]: results }));
    } catch (err) {
      console.error(`Failed to load results for job ${jobId}:`, err);
    }
  };

  // Poll for updates if there are running jobs
  useEffect(() => {
    if (activeJobs.some(j => j.status === "running" || j.status === "pending")) {
      const interval = setInterval(loadJobs, 2000);
      return () => clearInterval(interval);
    }
  }, [activeJobs]);

  const toggleTask = (task: string) => {
    const newSelected = new Set(selectedTasks);
    if (newSelected.has(task)) {
      newSelected.delete(task);
    } else {
      newSelected.add(task);
    }
    setSelectedTasks(newSelected);
  };

  const runEvaluation = async () => {
    if (!selectedCheckpoint || selectedTasks.size === 0) return;
    
    setIsRunning(true);
    setError(null);
    try {
      await fetchJson("/api/evaluations/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          checkpoint_id: selectedCheckpoint,
          tasks: Array.from(selectedTasks),
          limit: isDemo ? 10 : undefined // Limit for demo or speed
        }),
      });
      loadJobs();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="page-with-nav">
      <SideNav
        sections={SECTIONS}
        activeId={activeSection}
        onNavigate={setActiveSection}
        ariaLabel="Evaluations sections"
      />
      <div className="page-content">
        
        {/* Model & Task Selection */}
        <section id="select-model" className="section scroll-section">
          <div className="section-title">
            <h2>Model & Tasks</h2>
            <p>Configure your evaluation run.</p>
          </div>
          <div className="card">
            <div>
              <div style={{ marginBottom: "24px" }}>
                <label>Checkpoint</label>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                  disabled={isRunning}
                  style={{ width: "100%", maxWidth: "400px" }}
                >
                  <option value="">Select a checkpoint...</option>
                  {sortedCheckpoints.map((ckpt) => (
                    <option key={ckpt.id} value={ckpt.id}>
                      {formatCheckpointTimestamp(new Date(ckpt.mtime * 1000))} · {ckpt.name}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label>Tasks</label>
                <div style={{ 
                  display: "grid", 
                  gridTemplateColumns: "repeat(auto-fill, minmax(250px, 1fr))", 
                  gap: "12px",
                  maxHeight: "400px",
                  overflowY: "auto",
                  border: "1px solid var(--border)",
                  borderRadius: "6px",
                  padding: "16px",
                  marginTop: "8px"
                }}>
                  {Object.entries(taskMap).map(([taskId, info]) => (
                    <div key={taskId} className="tooltip-container" style={{ position: "relative" }}>
                      <label 
                        className="checkbox-label" 
                        style={{ fontSize: "14px", alignItems: "flex-start", cursor: "pointer" }}
                        title={info.description} // basic browser tooltip fallback
                      >
                        <input
                          type="checkbox"
                          checked={selectedTasks.has(taskId)}
                          onChange={() => toggleTask(taskId)}
                          disabled={isRunning}
                          style={{ marginTop: "4px" }}
                        />
                        <div style={{ marginLeft: "8px" }}>
                          <div style={{ fontWeight: 600, fontSize: "14px", color: "var(--ink-1)" }}>{info.name}</div>
                          <div style={{ fontSize: "13px", color: "var(--muted)", lineHeight: "1.4", marginTop: "4px", opacity: 0.9 }}>
                             {info.description}
                          </div>
                        </div>
                      </label>
                    </div>
                  ))}
                  {Object.keys(taskMap).length === 0 && (
                      <div className="text-muted">Loading tasks...</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Run Control */}
        <section id="run-eval" className="section scroll-section">
          <div className="section-title">
            <h2>Run Evaluation</h2>
            <p>Start the evaluation process.</p>
          </div>
          <div className="card">
            <button
              className="primary"
              onClick={runEvaluation}
              disabled={isRunning || !selectedCheckpoint || selectedTasks.size === 0 || activeJobs.some(j => j.checkpoint_id === selectedCheckpoint && (j.status === "running" || j.status === "pending"))}
            >
              {isRunning ? (
                <>
                  <Loader2 className="spin" size={16} style={{ marginRight: 8 }} />
                  Starting...
                </>
              ) : activeJobs.some(j => j.checkpoint_id === selectedCheckpoint && (j.status === "running" || j.status === "pending")) ? (
                "Evaluation Running..."
              ) : (
                "Start Evaluation"
              )}
            </button>
            {error && (
              <div style={{ marginTop: 12, color: "var(--error)", display: "flex", alignItems: "center", gap: 8 }}>
                <AlertCircle size={16} />
                {error}
              </div>
            )}
          </div>
        </section>

        {/* Results & History */}
        <section id="results" className="section scroll-section">
          <div className="section-title">
            <h2>Results</h2>
            <p>History of evaluation runs.</p>
          </div>
          
          <div className="stack">
            {activeJobs.length === 0 ? (
              <div className="card muted">No evaluations run yet.</div>
            ) : (
              activeJobs.map(job => (
                <div key={job.job_id} className="card">
                  <div className="inline-row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
                    <div>
                      <strong>
                        {checkpoints.find(c => c.id === job.checkpoint_id)?.name || job.checkpoint_id}
                      </strong>
                      <div className="text-muted" style={{ fontSize: "12px" }}>
                        {new Date(job.created_at * 1000).toLocaleString()}
                      </div>
                    </div>
                    <div>
                      <span className={`badge ${job.status === "completed" ? "success" : job.status === "failed" || job.status === "error" ? "error" : "warning"}`}>
                        {job.status}
                      </span>
                    </div>
                  </div>

                  <div style={{ marginBottom: 12 }}>
                    <span className="text-muted">Tasks: </span>
                    {job.tasks.map(slug => taskMap[slug]?.name || slug).join(", ")}
                  </div>

                  {job.status === "running" && (
                    <div style={{ marginTop: 8, marginBottom: 12 }}>
                      <div className="inline-row" style={{ justifyContent: "space-between", fontSize: "12px", marginBottom: 4 }}>
                        <span className="text-muted">Progress</span>
                        <span>{job.progress?.toFixed(1) || 0}%</span>
                      </div>
                      <div style={{ height: 6, background: "var(--border)", borderRadius: 3, overflow: "hidden" }}>
                        <div 
                          style={{ 
                            height: "100%", 
                            background: "var(--accent)", 
                            width: `${Math.max(5, job.progress || 0)}%`, // Min 5% to show something
                            transition: "width 0.5s ease" 
                          }} 
                        />
                      </div>
                    </div>
                  )}

                  {(job.status === "failed" || job.status === "error") && job.error && (
                    <div className="code-block error">
                      {job.error}
                    </div>
                  )}

                  {job.status === "completed" && completedResults[job.job_id] && (
                    <table className="table" style={{ marginTop: 12 }}>
                      <thead>
                        <tr>
                          <th>Task</th>
                          <th>Metrics</th>
                          <th>Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(completedResults[job.job_id]).map(([task, metrics]) => (
                           Object.entries(metrics).map(([metricKey, value], idx) => (
                             <tr key={`${task}-${metricKey}`}>
                               {idx === 0 && <td rowSpan={Object.keys(metrics).length} style={{ verticalAlign: "top", borderBottom: Object.keys(metrics).length > 1 ? "1px solid var(--border)" : undefined }}>
                                 {taskMap[task]?.name || task}
                               </td>}
                               <td>{metricKey}</td>
                               <td>
                                 {typeof value === 'number' 
                                   ? (metricKey.includes('stderr') ? `±${value.toFixed(4)}` : `${(value * 100).toFixed(2)}%` )
                                   : String(value)
                                 }
                               </td>
                             </tr>
                           ))
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              ))
            )}
          </div>
        </section>

      </div>
    </div>
  );
}
