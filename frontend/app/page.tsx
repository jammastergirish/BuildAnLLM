import fs from "fs";
import path from "path";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import SystemInfoPanel from "../components/SystemInfoPanel";

function loadReadme(): string {
  const readmePath = path.resolve(process.cwd(), "..", "README.md");
  if (!fs.existsSync(readmePath)) {
    return "";
  }
  const content = fs.readFileSync(readmePath, "utf-8");
  return content.replace(/<img[^>]*>/g, "");
}

export default async function OverviewPage() {
  const readme = loadReadme();

  return (
    <>
      <section className="section">
        <div className="section-title">
          <h2>Device</h2>
          {/* <p>Runtime and hardware details reported by the backend.</p> */}
        </div>
        <SystemInfoPanel />
      </section>

      <section className="section">
        <div className="section-title">
          <h2>README</h2>
          {/* <p>Reference notes from the repository.</p> */}
        </div>
        <div className="card" style={{ padding: 24 }}>
          {readme ? (
            <div className="readme-content">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex]}
                components={{
                  img: () => null,
                }}
              >
                {readme}
              </ReactMarkdown>
            </div>
          ) : (
            <p>README.md not found.</p>
          )}
        </div>
      </section>
    </>
  );
}
