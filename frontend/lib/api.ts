import { API_BASE_URL } from "./env";

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed with ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function makeFormData(payload: object, file?: File, fileField?: string) {
  const form = new FormData();
  form.set("payload", JSON.stringify(payload));
  if (file && fileField) {
    form.set(fileField, file);
  }
  return form;
}

export type JobStatus = {
  job_id: string;
  kind: string;
  status: string;
  iter: number;
  max_iters: number;
  created_at: number;
  error?: string | null;
};

export type Checkpoint = {
  id: string;
  path: string;
  run_id: string;
  is_finetuned: boolean;
  name: string;
  iter: number | null;
  mtime: number;
  size_bytes: number;
};

export type CodeSnippet = {
  title: string;
  module: string;
  object: string;
  file: string;
  start_line: number;
  end_line: number;
  github_url: string;
  code: string;
};
