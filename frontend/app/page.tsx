import fs from "fs";
import path from "path";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import StatCard from "../components/StatCard";
import { API_BASE_URL } from "../lib/env";

async function getSystemInfo() {
  const res = await fetch(`${API_BASE_URL}/api/system/info`, { cache: "no-store" });
  if (!res.ok) {
    return null;
  }
  return res.json();
}

function loadReadme(): string {
  const readmePath = path.resolve(process.cwd(), "..", "README.md");
  if (!fs.existsSync(readmePath)) {
    return "";
  }
  const content = fs.readFileSync(readmePath, "utf-8");
  return content.replace(/<img[^>]*>/g, "");
}

export default async function OverviewPage() {
  const systemInfo = await getSystemInfo();
  const readme = loadReadme();

  return (
    <>
      <section className="section">
        <div className="section-title">
          <h2>Device</h2>
          <p>Runtime and hardware details reported by the backend.</p>
        </div>
        <div className="card">
          {systemInfo ? (
            <div className="grid-3">
              <StatCard label="CPU" value={systemInfo.cpu} />
              <StatCard label="RAM" value={`${systemInfo.ram_gb} GB`} />
              <StatCard label="GPU" value={systemInfo.gpu} />
              <StatCard label="MPS" value={systemInfo.mps} />
              <StatCard label="OS" value={systemInfo.os} />
              <StatCard label="Python" value={systemInfo.python} />
              <StatCard label="PyTorch" value={systemInfo.torch} />
            </div>
          ) : (
            <p>Unable to load system info.</p>
          )}
        </div>
      </section>

      <section className="section">
        <div className="section-title">
          <h2>Project README</h2>
          <p>Reference notes from the repository.</p>
        </div>
        <div className="card" style={{ padding: 24 }}>
          {readme ? (
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
              {readme}
            </ReactMarkdown>
          ) : (
            <p>README.md not found.</p>
          )}
        </div>
      </section>
    </>
  );
}
