/** Local llama.cpp server provider over its OpenAI-compatible endpoint. */
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import type {
  ModelProvider,
  ModelDescriptor,
  ResolvedProviderConfig,
  ProviderAuthMethod,
} from "@pizza-bot/core";
import { createOpenAiChatModel, isChatModel } from "./openai.js";
import {
  catalogConnectionError,
  catalogHttpError,
} from "../catalog-error.js";

// llama.cpp's server (`llama-server`) serves the OpenAI-compatible routes under
// `/v1`, so discovery hits `/v1/models` and inference reuses the OpenAI client.
const DEFAULT_BASE_URL = "http://localhost:8080/v1";
const DEFAULT_CONTEXT_LENGTH = 32_768;
const DEFAULT_MAX_TOKENS = 8_192;
const FETCH_TIMEOUT_MS = 5_000;
// The `openai` SDK refuses to construct a client without a credential; a local
// llama-server authenticates nobody, so a placeholder bearer token stands in.
const KEYLESS_API_KEY = "none";

interface LlamaCppModelsResponse {
  data?: Array<{ id?: string }>;
}

const AUTH_SCHEMA: readonly ProviderAuthMethod[] = [
  {
    id: "local",
    label: "Local server",
    fields: [
      { key: "baseUrl", label: "Base URL", type: "text", required: false, default: DEFAULT_BASE_URL },
      { key: "apiKey", label: "API key", type: "password", required: false },
      {
        key: "contextLength",
        label: "Context window",
        type: "text",
        required: false,
        default: String(DEFAULT_CONTEXT_LENGTH),
      },
      {
        key: "maxTokens",
        label: "Output token budget",
        type: "text",
        required: false,
        default: String(DEFAULT_MAX_TOKENS),
      },
    ],
  },
];

export interface LlamaCppProviderOptions {
  baseUrl?: string;
  apiKey?: string;
  contextLength?: number;
  maxTokens?: number;
  fetch?: typeof fetch;
}

export class LlamaCppLangChainModelProvider implements ModelProvider {
  readonly id = "llamacpp";
  readonly authSchema = AUTH_SCHEMA;
  readonly availableWithoutConfig = true;
  private baseUrl: string;
  private apiKey: string | undefined;
  private contextLength: number;
  private maxTokens: number;
  private readonly fetchFn: typeof fetch;
  private readonly descriptors = new Map<string, ModelDescriptor>();

  constructor(opts: LlamaCppProviderOptions = {}) {
    this.baseUrl = normalizeBaseUrl(opts.baseUrl ?? process.env.LLAMACPP_BASE_URL ?? DEFAULT_BASE_URL);
    this.apiKey = opts.apiKey ?? process.env.LLAMACPP_API_KEY ?? undefined;
    this.contextLength = positiveInteger(opts.contextLength) ?? DEFAULT_CONTEXT_LENGTH;
    this.maxTokens = positiveInteger(opts.maxTokens) ?? DEFAULT_MAX_TOKENS;
    this.fetchFn = opts.fetch ?? fetch;
  }

  configure(cfg: ResolvedProviderConfig): void {
    this.baseUrl = normalizeBaseUrl(
      cfg.values.baseUrl?.trim() || process.env.LLAMACPP_BASE_URL || DEFAULT_BASE_URL,
    );
    this.apiKey = cfg.values.apiKey?.trim() || process.env.LLAMACPP_API_KEY || undefined;
    this.contextLength = positiveIntegerString(cfg.values.contextLength) ?? DEFAULT_CONTEXT_LENGTH;
    this.maxTokens = positiveIntegerString(cfg.values.maxTokens) ?? DEFAULT_MAX_TOKENS;
  }

  async listModels(): Promise<ModelDescriptor[]> {
    let response: Response;
    try {
      response = await this.fetchFn(`${this.baseUrl}/models`, {
        headers: this.apiKey ? { authorization: `Bearer ${this.apiKey}` } : {},
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      });
    } catch (cause) {
      throw catalogConnectionError("llama.cpp", cause);
    }
    if (!response.ok) throw catalogHttpError("llama.cpp", response.status);
    let body: LlamaCppModelsResponse;
    try {
      body = (await response.json()) as LlamaCppModelsResponse;
    } catch (cause) {
      throw catalogConnectionError("llama.cpp", cause);
    }
    const descriptors: ModelDescriptor[] = (body.data ?? []).flatMap((model) => {
      if (!model.id || !isChatModel(model.id)) return [];
      return [{
        id: model.id,
        provider: "llamacpp",
        displayName: `${model.id} (llama.cpp)`,
        contextWindow: this.contextLength,
        supportsTools: true,
        supportsVision: false,
      }];
    });
    this.descriptors.clear();
    for (const descriptor of descriptors) this.descriptors.set(descriptor.id, descriptor);
    return descriptors;
  }

  async buildModel(modelId: string): Promise<BaseChatModel> {
    if (!this.descriptors.has(modelId)) {
      await this.listModels().catch(() => []);
    }
    const descriptor = this.descriptors.get(modelId) ?? {
      id: modelId,
      provider: "llamacpp",
      displayName: `${modelId} (llama.cpp)`,
      contextWindow: this.contextLength,
    };
    const maxTokens = Math.min(
      this.maxTokens,
      descriptor.maxOutputTokens ?? Number.POSITIVE_INFINITY,
    );
    // llama-server implements Chat Completions, not the Responses API.
    return createOpenAiChatModel({
      modelId,
      apiKey: this.apiKey || KEYLESS_API_KEY,
      baseUrl: this.baseUrl,
      maxTokens,
      apiMode: "chat-completions",
      descriptor,
      fetch: this.fetchFn,
    });
  }
}

function normalizeBaseUrl(url: string): string {
  return url.replace(/\/$/, "");
}

function positiveInteger(value: number | undefined): number | undefined {
  if (value === undefined) return undefined;
  return Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function positiveIntegerString(value: string | undefined): number | undefined {
  if (!value?.trim()) return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : undefined;
}
