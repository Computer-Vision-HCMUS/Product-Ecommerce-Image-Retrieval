import React, { useState } from "react";

const MODALITIES = [
  { key: "image", label: "Image", type: "file", accept: "image/*" },
  { key: "title", label: "Title / Caption text", type: "text" },
  { key: "caption", label: "Description", type: "text" },
  { key: "pv", label: "Table (PV attributes)", type: "textarea" },
  { key: "video", label: "Video", type: "file", accept: "video/*" },
  { key: "audio", label: "Audio", type: "file", accept: "audio/*" },
];

export default function App() {
  const [topK, setTopK] = useState(10);
  const [fields, setFields] = useState({ title: "", caption: "", pv: "" });
  const [files, setFiles] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState(null);

  const onFile = (key, file) => setFiles((prev) => ({ ...prev, [key]: file }));

  const onSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResults(null);
    const form = new FormData();
    form.append("top_k", String(topK));
    form.append("title", fields.title);
    form.append("caption", fields.caption);
    form.append("pv", fields.pv);
    if (files.image) form.append("image", files.image);
    if (files.video) form.append("video", files.video);
    if (files.audio) form.append("audio", files.audio);
    try {
      const response = await fetch("/api/search", { method: "POST", body: form });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${response.status}`);
      }
      setResults(await response.json());
    } catch (err) {
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header>
        <h1>SCALE Product Retrieval</h1>
        <p>Query with up to 5 modalities. Missing modalities use zero imputation at fusion time.</p>
      </header>

      <form className="panel" onSubmit={onSubmit}>
        <label>
          Top K
          <input type="number" min="1" max="50" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
        </label>

        {MODALITIES.map((item) =>
          item.type === "file" ? (
            <label key={item.key}>
              {item.label}
              <input type="file" accept={item.accept} onChange={(e) => onFile(item.key, e.target.files?.[0] || null)} />
            </label>
          ) : (
            <label key={item.key}>
              {item.label}
              {item.type === "textarea" ? (
                <textarea
                  rows="4"
                  value={fields[item.key]}
                  onChange={(e) => setFields((prev) => ({ ...prev, [item.key]: e.target.value }))}
                />
              ) : (
                <input
                  type="text"
                  value={fields[item.key]}
                  onChange={(e) => setFields((prev) => ({ ...prev, [item.key]: e.target.value }))}
                />
              )}
            </label>
          )
        )}

        <button type="submit" disabled={loading}>{loading ? "Searching..." : "Search"}</button>
      </form>

      {error && <div className="error">{error}</div>}

      {results && (
        <section className="panel">
          <h2>Results</h2>
          <p className="meta">Query modalities: {JSON.stringify(results.query_modalities)}</p>
          <div className="grid">
            {results.top_k.map((item) => (
              <article key={item.id} className="card">
                {item.image_path ? (
                  <img src={`/api/file?path=${encodeURIComponent(item.image_path)}`} alt={item.title} onError={(e) => { e.currentTarget.style.display = "none"; }} />
                ) : (
                  <div className="placeholder">No image</div>
                )}
                <div className="card-body">
                  <strong>{item.title || item.id}</strong>
                  <span>{item.label}</span>
                  <span>score: {item.score.toFixed(4)}</span>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
