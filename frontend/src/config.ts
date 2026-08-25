export const DEFAULT_API_BASE_URL = "http://localhost:8000";

const removeTrailingSlashes = (value: string): string => value.replace(/\/+$/, "");

export const resolveApiBaseUrl = (
  runtimeApiBaseUrl?: string,
  viteApiBaseUrl?: string,
): string => {
  const configuredUrl = runtimeApiBaseUrl?.trim() || viteApiBaseUrl?.trim();
  return removeTrailingSlashes(configuredUrl || DEFAULT_API_BASE_URL);
};

const runtimeApiBaseUrl = document.getElementById("chatbot-root")?.dataset.apiBaseUrl;

export const API_BASE_URL = resolveApiBaseUrl(
  runtimeApiBaseUrl,
  import.meta.env.VITE_API_BASE_URL,
);

export const CHATBOT_API_TIMEOUTS_MS = {
  CREATE_SESSION: 3000,
  DELETE_SESSION: 3000,
  GENERATE_MESSAGE: 300000,
};
