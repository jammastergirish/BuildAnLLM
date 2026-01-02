"use client";

import { useEffect, useState } from "react";
import TokenRainbow from "../../components/TokenRainbow";
import { fetchJson } from "../../lib/api";

export default function PlaygroundPage() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("gpt-4");
  const [inputText, setInputText] = useState(
    "Hello world! This is a test of the tokenizer.\n\nPython code:\ndef hello():\n    print('world')"
  );
  const [tokens, setTokens] = useState<number[]>([]);
  const [decodedTokens, setDecodedTokens] = useState<string[]>([]);
  const [stats, setStats] = useState({ token_count: 0, char_count: 0, chars_per_token: 0 });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJson<{ models: string[] }>("/api/tokenizers/tiktoken/models")
      .then((data) => {
        setModels(data.models);
        if (data.models.length > 0 && !data.models.includes(selectedModel)) {
          setSelectedModel(data.models[0]);
        }
      })
      .catch((err) => setError((err as Error).message));
  }, []);

  useEffect(() => {
    const encode = async () => {
      try {
        const data = await fetchJson<{
          tokens: number[];
          decoded_tokens: string[];
          token_count: number;
          char_count: number;
          chars_per_token: number;
        }>("/api/tokenizers/tiktoken/encode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: selectedModel, text: inputText }),
        });
        setTokens(data.tokens);
        setDecodedTokens(data.decoded_tokens);
        setStats({
          token_count: data.token_count,
          char_count: data.char_count,
          chars_per_token: data.chars_per_token,
        });
      } catch (err) {
        setError((err as Error).message);
      }
    };

    encode();
  }, [selectedModel, inputText]);

  return (
    <>
      <section className="section">
        <div className="section-title">
          <h2>Tokenizer Playground</h2>
          <p>Inspect how different models split input text.</p>
        </div>
        <div className="card">
          <div className="grid-2">
            <div>
              <label>Model</label>
              <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                {models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
              <div style={{ marginTop: 16 }}>
                <label>Input</label>
                <textarea value={inputText} onChange={(event) => setInputText(event.target.value)} />
              </div>
              <div className="grid-3" style={{ marginTop: 12 }}>
                <div className="stat">
                  <h4>Characters</h4>
                  <p>{stats.char_count}</p>
                </div>
                <div className="stat">
                  <h4>Tokens</h4>
                  <p>{stats.token_count}</p>
                </div>
                <div className="stat">
                  <h4>Characters per Token</h4>
                  <p>{stats.chars_per_token.toFixed(2)}</p>
                </div>
              </div>
            </div>
            <div>
              <label>Tokens</label>
              <div className="card" style={{ boxShadow: "none", background: "var(--card-muted)" }}>
                {decodedTokens.length > 0 ? (
                  <TokenRainbow tokens={decodedTokens} />
                ) : (
                  <p>No tokens yet.</p>
                )}
              </div>
              <div style={{ marginTop: 16 }}>
                <label>Raw IDs</label>
                <div className="log-box">{tokens.join(", ")}</div>
              </div>
            </div>
          </div>
          {error && <p style={{ color: "#b42318" }}>{error}</p>}
        </div>
      </section>
    </>
  );
}
