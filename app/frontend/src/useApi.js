import { useEffect, useState } from "react";
import { getJSON } from "./api.js";

export function useApi(path, runId) {
  const [state, setState] = useState({ data: null, error: null });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    setState({ data: null, error: null });
    getJSON(`/api/${path}${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`)
      .then((data) => live && setState({ data, error: null }))
      .catch((e) => live && setState({ data: null, error: e.message }));
    return () => { live = false; };
  }, [path, runId, attempt]);
  return { ...state, retry: () => setAttempt((a) => a + 1) };
}
