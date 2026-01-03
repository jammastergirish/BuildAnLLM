const defaultApiBase =
  process.env.NODE_ENV === "production" ? "" : "http://127.0.0.1:8000";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? defaultApiBase;
