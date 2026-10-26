// Reuse DSH's native DeepSeek protocol; private vLLM accepts Bearer authentication.
// The credential is resolved in memory and never enters plugin configuration.
import { pathToFileURL } from 'node:url';
import { liftToolResultImages } from './message_images.mjs';
const root = `${process.env.RAX_DSH_INSTALL}/node_modules/@deepseek-ai`;
const protocol = await import(pathToFileURL(`${root}/dsh-llm-deepseek/lib/index.js`).href);
export const Config = protocol.Config;
export const name = 'rax-private-deepseek';
export const inject = ['llm'];

export function apply(ctx, config) {
  const options = () => protocol.resolveAdapterOptions(protocol.plainOptions(config));
  const endpoint = options().baseURL.replace(/\/+$/, '');
  const fetch = globalThis.fetch;
  // vLLM's Anthropic schema still requires the legacy budget field. Preserve
  // native max effort and allow the full configured output window for thinking.
  globalThis.fetch = (input, init) => {
    const url = String(input?.url ?? input);
    if ((url === `${endpoint}/v1/messages` || url === `${endpoint}/messages`) && init?.body) {
      const body = JSON.parse(init.body);
      if (body.thinking?.type === 'enabled' && body.thinking.budget_tokens === undefined) {
        body.thinking.budget_tokens = body.max_tokens - 1;
      }
      liftToolResultImages(body);
      init = { ...init, body: JSON.stringify(body) };
    }
    return fetch(input, init);
  };
  protocol.registerDeepSeekProvider(ctx, 'rax-private', {
    options,
    providerName: 'Private DeepSeek',
    resolveAuth: async () => {
      const key = process.env.RAX_MODEL_KEY;
      if (!key) throw new Error('Private DeepSeek credential is missing');
      return { headers: { authorization: `Bearer ${key}` } };
    },
    discoverModels: async provider => options().models.map(model => protocol.catalogModelInfo(provider, model)),
  });
}
