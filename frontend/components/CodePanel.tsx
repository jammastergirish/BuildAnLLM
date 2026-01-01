import { CodeSnippet } from "../lib/api";

export default function CodePanel({ snippet }: { snippet: CodeSnippet }) {
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="flex-between" style={{ marginBottom: 8 }}>
        <strong>{snippet.title}</strong>
        <a href={snippet.github_url} className="badge" target="_blank" rel="noreferrer">
          view source
        </a>
      </div>
      <div className="badge" style={{ marginBottom: 12 }}>
        {snippet.file}:{snippet.start_line}-{snippet.end_line}
      </div>
      <pre className="code-block">{snippet.code}</pre>
    </div>
  );
}
