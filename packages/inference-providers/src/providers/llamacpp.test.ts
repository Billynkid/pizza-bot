import { afterEach, describe, expect, it, vi } from "vitest";
import { LlamaCppLangChainModelProvider } from "./llamacpp.js";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("llama.cpp model discovery", () => {
  it("reports an unreachable server as a retryable network failure", async () => {
    const cause = new TypeError("fetch failed");
    const provider = new LlamaCppLangChainModelProvider({
      fetch: vi.fn().mockRejectedValue(cause),
    });

    await expect(provider.listModels()).rejects.toMatchObject({
      name: "ModelCatalogError",
      code: "network",
      retryable: true,
      cause,
    });
  });

  it("reports server HTTP failures with endpoint retry semantics", async () => {
    const provider = new LlamaCppLangChainModelProvider({
      fetch: vi.fn().mockResolvedValue(new Response(null, { status: 503 })),
    });

    await expect(provider.listModels()).rejects.toMatchObject({
      name: "ModelCatalogError",
      code: "endpoint",
      retryable: true,
    });
  });

  it("maps the OpenAI-compatible model listing into descriptors, skipping non-chat models", async () => {
    const fetchFn = vi.fn(async () =>
      Response.json({ data: [{ id: "qwen2.5-coder-7b" }, { id: "text-embedding-3-small" }] }),
    );
    const provider = new LlamaCppLangChainModelProvider({
      contextLength: 16_384,
      fetch: fetchFn,
    });

    await expect(provider.listModels()).resolves.toEqual([{
      id: "qwen2.5-coder-7b",
      provider: "llamacpp",
      displayName: "qwen2.5-coder-7b (llama.cpp)",
      contextWindow: 16_384,
      supportsTools: true,
      supportsVision: false,
    }]);
  });

  it("queries the default local endpoint without an authorization header", async () => {
    const fetchFn = vi.fn(async () => Response.json({ data: [] }));
    const provider = new LlamaCppLangChainModelProvider({ fetch: fetchFn });

    await provider.listModels();

    expect(fetchFn).toHaveBeenCalledWith(
      "http://localhost:8080/v1/models",
      expect.objectContaining({ headers: {} }),
    );
  });

  it("sends a bearer token when an API key is configured", async () => {
    const fetchFn = vi.fn(async () => Response.json({ data: [] }));
    const provider = new LlamaCppLangChainModelProvider({ fetch: fetchFn });

    provider.configure({
      method: "local",
      values: { baseUrl: "http://gpu-box.example:9000/v1", apiKey: "secret" },
    });
    await provider.listModels();

    expect(fetchFn).toHaveBeenCalledWith(
      "http://gpu-box.example:9000/v1/models",
      expect.objectContaining({ headers: { authorization: "Bearer secret" } }),
    );
  });

  it("falls back to the environment base URL when config clears it", async () => {
    vi.stubEnv("LLAMACPP_BASE_URL", "http://llama-fallback.example/v1");
    const fetchFn = vi.fn(async () => Response.json({ data: [] }));
    const provider = new LlamaCppLangChainModelProvider({
      baseUrl: "http://llama-custom.example/v1",
      fetch: fetchFn,
    });

    provider.configure({ method: "local", values: {} });
    await provider.listModels();

    expect(fetchFn).toHaveBeenCalledWith(
      "http://llama-fallback.example/v1/models",
      expect.any(Object),
    );
  });

  it("builds a model whose profile reflects the configured context window", async () => {
    const fetchFn = vi.fn(async () => Response.json({ data: [{ id: "local-model" }] }));
    const provider = new LlamaCppLangChainModelProvider({
      contextLength: 8_192,
      fetch: fetchFn,
    });

    const model = await provider.buildModel("local-model");
    expect(model.profile.maxInputTokens).toBe(8_192);
  });
});
