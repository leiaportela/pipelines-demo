const cache = new Map();

export async function getJSON(url) {
  if (cache.has(url)) return cache.get(url);
  const p = fetch(url).then(async (r) => {
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || `${r.status} ${r.statusText}`);
    return body;
  });
  cache.set(url, p);
  p.catch(() => cache.delete(url));
  return p;
}
