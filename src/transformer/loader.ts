/** Loads the toy Transformer weights (public/models/toy-transformer.json) once. */
import { loadTransformer, type TransformerFile, type TransformerModel } from './model';

let pending: Promise<TransformerModel> | null = null;

export function loadToyTransformer(): Promise<TransformerModel> {
  pending ??= fetch(new URL('models/toy-transformer.json', document.baseURI))
    .then(async (response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status} while loading the model`);
      return loadTransformer((await response.json()) as TransformerFile);
    })
    .catch((error: unknown) => {
      pending = null; // allow a retry
      throw error;
    });
  return pending;
}
