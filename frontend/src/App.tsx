import type React from "react";
import { useCallback, useRef, useState } from "react";

// ─── Types ──────────────────────────────────────────────────────────────────
interface Damage {
	id: number;
	class: string;
	confidence: number;
	box: number[];
	severity: "Mild" | "Moderate" | "Severe";
	panel?: string;
	metrics: {
		pixel_area: number;
		cm2_area: number;
		length_cm?: number;
		depth_cm?: number;
		width_cm?: number;
		height_cm?: number;
		spread_area_cm2?: number;
	};
}

interface AnalysisResult {
	success: boolean;
	overall_severity: string;
	summary: { Mild: number; Moderate: number; Severe: number };
	calibration: { cm_per_pixel: number; reference_found: boolean };
	damages: Damage[];
	annotated_image: string;
	repair_guide: string;
}

// ─── Constants ───────────────────────────────────────────────────────────────
const API_KEY = "overbody_secure_key_2026";
const API_BASE = "http://localhost:8000";

const CLASS_COLORS: Record<string, string> = {
	scratch: "#06b6d4",
	dent: "#a855f7",
	crack: "#eab308",
	rust: "#f97316",
	glass_shatter: "#38bdf8",
	broken_lamp: "#f59e0b",
};

const LOADING_STAGES = [
	"Extracting visual edges and contours…",
	"Locating potential surface deformities…",
	"Measuring physical scale dimensions…",
	"Running AI severity classifier…",
	"Generating repair guidance with Gemini…",
];

// ─── Markdown renderer ───────────────────────────────────────────────────────
function renderMarkdown(text: string): React.ReactNode[] {
	if (!text) return [];
	const lines = text.split("\n");
	const out: React.ReactNode[] = [];
	let tableHeaders: string[] = [];
	let tableRows: string[][] = [];
	let inTable = false;

	const bold = (str: string, key?: string | number) => {
		const parts = str.split("**");
		return (
			<span key={key}>
				{/* biome-ignore lint/suspicious/noArrayIndexKey: rendering static split parts */}
				{parts.map((p, i) => (i % 2 === 1 ? <strong key={i}>{p}</strong> : p))}
			</span>
		);
	};

	const flushTable = (i: number) => {
		if (!tableHeaders.length) return;
		out.push(
			<table key={`tbl-${i}`}>
				<thead>
					<tr>
						{/* biome-ignore lint/suspicious/noArrayIndexKey: static headers */}
						{tableHeaders.map((h, hi) => (
							<th key={hi}>{h}</th>
						))}
					</tr>
				</thead>
				<tbody>
					{/* biome-ignore lint/suspicious/noArrayIndexKey: static rows */}
					{tableRows.map((row, ri) => (
						<tr key={ri}>
							{/* biome-ignore lint/suspicious/noArrayIndexKey: static cells */}
							{row.map((cell, ci) => (
								<td key={ci}>{bold(cell)}</td>
							))}
						</tr>
					))}
				</tbody>
			</table>,
		);
		tableHeaders = [];
		tableRows = [];
		inTable = false;
	};

	for (let i = 0; i < lines.length; i++) {
		const t = lines[i].trim();

		if (t.startsWith("|")) {
			const cells = t
				.split("|")
				.map((c) => c.trim())
				.filter(Boolean);
			if (cells.every((c) => /^[-:]+$/.test(c))) continue;
			if (!inTable) {
				inTable = true;
				tableHeaders = cells;
			} else tableRows.push(cells);
			continue;
		}
		if (inTable) {
			flushTable(i);
		}

		if (!t) continue;
		if (t.startsWith("#### ")) {
			out.push(<h4 key={i}>{t.slice(5)}</h4>);
		} else if (t.startsWith("### ")) {
			out.push(<h3 key={i}>{t.slice(4)}</h3>);
		} else if (t.startsWith("## ")) {
			out.push(<h2 key={i}>{t.slice(3)}</h2>);
		} else if (t.startsWith("# ")) {
			out.push(<h2 key={i}>{t.slice(2)}</h2>);
		} else if (t.startsWith("- ") || t.startsWith("* ")) {
			out.push(<li key={i}>{bold(t.slice(2))}</li>);
		} else {
			out.push(<p key={i}>{bold(t)}</p>);
		}
	}
	if (inTable) flushTable(lines.length);
	return out;
}

// ─── App ─────────────────────────────────────────────────────────────────────
export default function App() {
	const [file, setFile] = useState<File | null>(null);
	const [preview, setPreview] = useState<string | null>(null);
	const [loading, setLoading] = useState(false);
	const [stage, setStage] = useState("");
	const [result, setResult] = useState<AnalysisResult | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [tab, setTab] = useState<"annotated" | "split" | "restored">(
		"annotated",
	);
	const [splitPos, setSplitPos] = useState(50);
	const [hoverId, setHoverId] = useState<number | null>(null);
	const [imgDim, setImgDim] = useState({ w: 0, h: 0, nw: 1, nh: 1 });
	const imgRef = useRef<HTMLImageElement>(null);

	// Handle file selection
	const onFile = (f: File) => {
		setError(null);
		setResult(null);
		setFile(f);
		setPreview(URL.createObjectURL(f));
	};

	const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const f = e.target.files?.[0];
		if (f) onFile(f);
	};

	const onDrop = (e: React.DragEvent) => {
		e.preventDefault();
		const f = e.dataTransfer.files?.[0];
		if (f && f.type.startsWith("image/")) onFile(f);
	};

	// Run analysis
	const analyze = useCallback(async (f: File) => {
		setLoading(true);
		setError(null);
		let si = 0;
		setStage(LOADING_STAGES[0]);
		const iv = setInterval(() => {
			si++;
			if (si < LOADING_STAGES.length) setStage(LOADING_STAGES[si]);
		}, 1300);

		try {
			const fd = new FormData();
			fd.append("file", f);
			const res = await fetch(`${API_BASE}/api/v1/analyze`, {
				method: "POST",
				headers: { "X-API-Key": API_KEY },
				body: fd,
			});
			if (res.status === 429)
				throw new Error("Rate limit hit — wait a moment and try again.");
			if (!res.ok) {
				const d = await res.json().catch(() => ({}));
				throw new Error(d.detail || `Server error ${res.status}`);
			}
			const data: AnalysisResult = await res.json();
			setResult(data);
			setTab("annotated");
		} catch (e: unknown) {
			setError(
				e instanceof Error
					? e.message
					: "Unknown error. Is the backend running on port 8000?",
			);
		} finally {
			clearInterval(iv);
			setLoading(false);
		}
	}, []);

	const reset = () => {
		setFile(null);
		setPreview(null);
		setResult(null);
		setError(null);
		setTab("annotated");
		setSplitPos(50);
	};

	// Export report
	const exportReport = async () => {
		if (!result) return;
		try {
			const res = await fetch(`${API_BASE}/api/v1/export-report`, {
				method: "POST",
				headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
				body: JSON.stringify(result),
			});
			if (res.ok) {
				const html = await res.text();
				const w = window.open("", "_blank");
				if (w) {
					w.document.write(html);
					w.document.close();
				}
			}
		} catch {
			/* ignore */
		}
	};

	// Image loaded — record natural dims for bounding box scaling
	const onImgLoad = () => {
		const el = imgRef.current;
		if (!el) return;
		setImgDim({
			w: el.clientWidth,
			h: el.clientHeight,
			nw: el.naturalWidth || 1,
			nh: el.naturalHeight || 1,
		});
	};

	// Sample data
	const loadSample = () => {
		setLoading(true);
		setStage("Loading sample data…");
		setTimeout(() => {
			setResult({
				success: true,
				overall_severity: "Severe",
				summary: { Mild: 1, Moderate: 1, Severe: 1 },
				calibration: { cm_per_pixel: 0.04, reference_found: true },
				damages: [
					{
						id: 1,
						class: "scratch",
						confidence: 0.89,
						box: [80, 160, 260, 38],
						severity: "Moderate",
						panel: "Door Panel",
						metrics: { pixel_area: 9880, cm2_area: 15.8, length_cm: 10.5 },
					},
					{
						id: 2,
						class: "dent",
						confidence: 0.94,
						box: [380, 80, 160, 160],
						severity: "Severe",
						panel: "Rear Bumper / Trunk",
						metrics: {
							pixel_area: 25600,
							cm2_area: 41.0,
							depth_cm: 2.5,
							width_cm: 6.4,
							height_cm: 6.4,
						},
					},
					{
						id: 3,
						class: "rust",
						confidence: 0.76,
						box: [50, 280, 100, 72],
						severity: "Mild",
						panel: "Front Bumper / Fender",
						metrics: {
							pixel_area: 7200,
							cm2_area: 11.5,
							spread_area_cm2: 11.5,
						},
					},
				],
				annotated_image:
					"https://images.unsplash.com/photo-1619642751034-765dfdf7c58e?auto=format&fit=crop&q=80&w=800",
				repair_guide: `## Vehicle Overbody Repair Report (Sample Mode)

### 1. Overall Condition Assessment
- **Overall Severity Rating:** Severe Damage Detected (Urgent action advised)
- **Primary Finding:** Deep dent (2.5 cm) on Rear Bumper, moderate paint scratch on Door Panel, and mild rust near front wheel arch.

### 2. Action Plan per Finding

#### Finding #1 — Dent (Severe)
- **Steps:** Depth of 2.5 cm exceeds simple suction tools. Use professional Paintless Dent Repair (PDR) rods or weld-on puller pins, then apply Bondo filler, block sand, and repaint.

#### Finding #2 — Scratch (Moderate)
- **Steps:** Wet sand with 2000-grit paper, apply primer, spray color-matched base coat, finish with 2K clear coat, machine-buff to blend.

#### Finding #3 — Rust (Mild)
- **Steps:** Sand to bare metal, treat with phosphoric acid rust converter, prime with zinc-rich primer, repaint.

### 3. Recommended Tools & Materials
- **Tools:** DA sander, block sander, slide-hammer puller, PDR rod set.
- **Supplies:** Bondo body filler, 2K clear coat, color-matched paint, 2000-grit wet sandpaper, wax & grease remover.

### 4. Estimated Cost Range
| Damage | DIY Cost | Professional Cost |
| :--- | :--- | :--- |
| Severe Dent | $120 – $250 | $800 – $1,800 |
| Moderate Scratch | $60 – $110 | $400 – $650 |
| Mild Rust Spot | $45 – $90 | $350 – $600 |
`,
			});
			setPreview(
				"https://images.unsplash.com/photo-1619642751034-765dfdf7c58e?auto=format&fit=crop&q=80&w=800",
			);
			setLoading(false);
		}, 1200);
	};

	// Helpers
	const hasPanelDmg = (...kws: string[]) =>
		result?.damages.some((d) =>
			kws.some((k) => (d.panel ?? "").toLowerCase().includes(k.toLowerCase())),
		) ?? false;

	const scaleX = imgDim.w / imgDim.nw;
	const scaleY = imgDim.h / imgDim.nh;

	const sevClass = (s: string) =>
		s === "Severe"
			? "severe"
			: s === "Moderate"
				? "moderate"
				: s === "Mild"
					? "mild"
					: "good";

	// ─── Render ─────────────────────────────────────────────────────────────────
	return (
		<div className="app-shell">
			{/* ── Navbar ── */}
			<header className="navbar">
				<div className="navbar-brand">
					<div className="navbar-icon">
						<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								strokeWidth="2.5"
								d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
							/>
						</svg>
					</div>
					<div>
						<div className="navbar-title">Overbody Damage Detector</div>
						<div className="navbar-subtitle">AI Severity Advisor</div>
					</div>
				</div>
				<div className="navbar-actions">
					<button className="btn btn-ghost" onClick={loadSample}>
						Demo Sample
					</button>
					{result && (
						<button className="btn btn-secondary btn-sm" onClick={reset}>
							New Analysis
						</button>
					)}
				</div>
			</header>

			{/* ── Main ── */}
			<main className="main-content">
				{/* ─── LANDING ─── */}
				{!preview && !loading && (
					<div className="landing-container fade-up">
						<div className="landing-badge">
							<span className="landing-badge-dot" />
							Powered by OpenCV + Gemini AI
						</div>
						<h1 className="landing-h1">
							Detect & Assess
							<br />
							<span>Vehicle Surface Damage</span>
						</h1>
						<p className="landing-subtitle">
							Upload a photo of any vehicle panel. Our CV pipeline detects
							scratches, dents, and rust — then estimates physical dimensions
							and generates an AI-powered repair guide.
						</p>

						<div
							className="drop-zone"
							onDragOver={(e) => e.preventDefault()}
							onDrop={onDrop}
						>
							<input
								type="file"
								accept="image/png,image/jpeg,image/webp"
								onChange={onInputChange}
							/>
							<div className="drop-zone-icon">
								<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path
										strokeLinecap="round"
										strokeLinejoin="round"
										strokeWidth="1.8"
										d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
									/>
								</svg>
							</div>
							<div className="drop-zone-title">
								<span>Click to upload</span> or drag & drop
							</div>
							<div className="drop-zone-hint">PNG, JPG, WEBP supported</div>
						</div>

						<div className="landing-actions">
							<button className="btn btn-primary" onClick={loadSample}>
								<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path
										strokeLinecap="round"
										strokeLinejoin="round"
										strokeWidth="2"
										d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
									/>
									<path
										strokeLinecap="round"
										strokeLinejoin="round"
										strokeWidth="2"
										d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
									/>
								</svg>
								Try Sample Analysis
							</button>
						</div>
					</div>
				)}

				{/* ─── PREVIEW (image selected, not analyzed) ─── */}
				{preview && !result && !loading && (
					<div className="preview-card fade-up">
						<div className="preview-card-header">
							<h3>Selected Image</h3>
							<button className="btn btn-ghost btn-sm" onClick={reset}>
								Change
							</button>
						</div>
						<img className="preview-img" src={preview} alt="Preview" />
						<div className="preview-card-actions">
							<button className="btn btn-secondary btn-full" onClick={reset}>
								Cancel
							</button>
							<button
								className="btn btn-primary btn-full"
								onClick={() => file && analyze(file)}
							>
								<svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path
										strokeLinecap="round"
										strokeLinejoin="round"
										strokeWidth="2"
										d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
									/>
								</svg>
								Run Assessment
							</button>
						</div>
					</div>
				)}

				{/* ─── LOADING ─── */}
				{loading && (
					<div className="loading-state fade-up">
						<div className="spinner">
							<div className="spinner-track" />
							<div className="spinner-thumb" />
						</div>
						<div className="loading-title">Analyzing Vehicle Surface</div>
						<div className="loading-stage">{stage}</div>
					</div>
				)}

				{/* ─── ERROR ─── */}
				{error && (
					<div className="error-state fade-up">
						<svg
							className="error-icon"
							fill="none"
							stroke="currentColor"
							viewBox="0 0 24 24"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								strokeWidth="2"
								d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
							/>
						</svg>
						<p className="error-message">{error}</p>
						<button className="btn btn-secondary" onClick={reset}>
							Reset
						</button>
					</div>
				)}

				{/* ─── RESULTS DASHBOARD ─── */}
				{result && (
					<div className="dashboard fade-up">
						{/* ═══ LEFT COLUMN ════════════════════════════════════════════ */}
						<div className="dashboard-left">
							{/* ── Image Visualizer Card ── */}
							<div className="card">
								<div className="card-header">
									<div className="card-header-left">
										<div className="card-header-icon blue">
											<svg
												fill="none"
												stroke="currentColor"
												viewBox="0 0 24 24"
											>
												<path
													strokeLinecap="round"
													strokeLinejoin="round"
													strokeWidth="2"
													d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
												/>
												<path
													strokeLinecap="round"
													strokeLinejoin="round"
													strokeWidth="2"
													d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
												/>
											</svg>
										</div>
										<span className="card-title">Surface Visualizer</span>
										<span className="card-tag">
											{result.calibration.reference_found
												? "Calibrated"
												: "Approx."}
										</span>
									</div>
									<div className="tab-row">
										<button
											className={`tab-btn ${tab === "annotated" ? "active" : ""}`}
											onClick={() => setTab("annotated")}
										>
											Overlay
										</button>
										<button
											className={`tab-btn ${tab === "split" ? "active" : ""}`}
											onClick={() => setTab("split")}
										>
											Compare
										</button>
										<button
											className={`tab-btn green ${tab === "restored" ? "active" : ""}`}
											onClick={() => setTab("restored")}
										>
											Restored
										</button>
									</div>
								</div>

								<div className="card-body" style={{ paddingTop: 14 }}>
									{/* Split Compare View */}
									{tab === "split" && (
										<div
											className="split-container"
											style={
												{ "--split-x": `${splitPos}%` } as React.CSSProperties
											}
										>
											<div className="split-base">
												<img src={preview!} alt="Original" />
											</div>
											<div className="split-overlay">
												<img src={result.annotated_image} alt="Annotated" />
											</div>
											<div className="split-handle">
												<div className="split-handle-knob">
													<svg
														fill="none"
														stroke="currentColor"
														viewBox="0 0 24 24"
														aria-label="Split slider handle"
													>
														<path
															strokeLinecap="round"
															strokeLinejoin="round"
															strokeWidth="2.5"
															d="M8 9l4-4 4 4m0 6l-4 4-4-4"
														/>
													</svg>
												</div>
											</div>
											<input
												className="split-input"
												type="range"
												min={0}
												max={100}
												value={splitPos}
												onChange={(e) => setSplitPos(Number(e.target.value))}
											/>
										</div>
									)}

									{/* Overlay / Restored Views */}
									{tab !== "split" && (
										<div className="image-viewer" style={{ maxHeight: 380 }}>
											<div
												style={{
													position: "relative",
													display: "inline-block",
												}}
											>
												<img
													ref={imgRef}
													onLoad={onImgLoad}
													src={
														tab === "restored"
															? preview!
															: result.annotated_image
													}
													alt="Vehicle"
													style={{
														maxWidth: "100%",
														maxHeight: 360,
														display: "block",
														filter:
															tab === "restored"
																? "brightness(1.04) contrast(0.95) saturate(1.1)"
																: "none",
													}}
												/>

												{/* Bounding boxes */}
												{tab === "annotated" &&
													imgDim.w > 0 &&
													result.damages.map((d) => {
														const [bx, by, bw, bh] = d.box;
														const isH = hoverId === d.id;
														const cls =
															d.class in CLASS_COLORS ? d.class : "default";
														return (
															<div
																key={d.id}
																className={`bbox-overlay ${cls}${isH ? " hovered" : ""}`}
																style={{
																	left: bx * scaleX,
																	top: by * scaleY,
																	width: bw * scaleX,
																	height: bh * scaleY,
																	borderColor:
																		CLASS_COLORS[d.class] ?? "#22c55e",
																}}
																onMouseEnter={() => setHoverId(d.id)}
																onMouseLeave={() => setHoverId(null)}
															>
																{isH && (
																	<div className="bbox-tooltip">
																		{d.class.replace("_", " ").toUpperCase()} ·{" "}
																		{Math.round(d.confidence * 100)}%
																	</div>
																)}
															</div>
														);
													})}

												{/* Restored badge */}
												{tab === "restored" && (
													<div className="restoration-badge">
														✨ Restoration Simulation
													</div>
												)}
											</div>
										</div>
									)}

									<div className="viewer-tip">
										<span>
											{tab === "split"
												? "↔ Drag slider to compare original vs. annotated"
												: tab === "restored"
													? "✨ Damage overlays hidden — restoration preview active"
													: "🔍 Hover damage boxes to inspect metrics"}
										</span>
										<button className="btn btn-ghost btn-sm" onClick={reset}>
											New image
										</button>
									</div>
								</div>
							</div>

							{/* ── AI Repair Guidance Card ── */}
							<div className="card">
								<div className="card-header">
									<div className="card-header-left">
										<div className="card-header-icon purple">
											<svg
												fill="none"
												stroke="currentColor"
												viewBox="0 0 24 24"
												aria-label="Lightning bolt guidance icon"
											>
												<path
													strokeLinecap="round"
													strokeLinejoin="round"
													strokeWidth="2"
													d="M13 10V3L4 14h7v7l9-11h-7z"
												/>
											</svg>
										</div>
										<span className="card-title">AI Repair Guidance</span>
									</div>
									<button
										className="btn btn-ghost btn-sm"
										onClick={exportReport}
									>
										<svg
											fill="none"
											stroke="currentColor"
											viewBox="0 0 24 24"
											aria-label="Print icon"
										>
											<path
												strokeLinecap="round"
												strokeLinejoin="round"
												strokeWidth="2"
												d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"
											/>
										</svg>
										Print PDF
									</button>
								</div>
								<div className="card-body">
									<div className="repair-guide-content">
										{renderMarkdown(result.repair_guide)}
									</div>
								</div>
							</div>
						</div>

						{/* ═══ RIGHT COLUMN ═══════════════════════════════════════════ */}
						<div className="dashboard-right">
							{/* ── Severity Summary ── */}
							<div className="card">
								<div className="card-header">
									<div className="card-header-left">
										<div className="card-header-icon green">
											<svg
												fill="none"
												stroke="currentColor"
												viewBox="0 0 24 24"
												aria-label="Overview chart icon"
											>
												<path
													strokeLinecap="round"
													strokeLinejoin="round"
													strokeWidth="2"
													d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
												/>
											</svg>
										</div>
										<span className="card-title">Condition Overview</span>
									</div>
								</div>
								<div className="card-body">
									<div className="severity-header">
										<div>
											<div className="severity-label">Overall Rating</div>
											<div
												className={`severity-value ${sevClass(result.overall_severity)}`}
											>
												{result.overall_severity} Damage
											</div>
											<div className="severity-caption">
												Based on highest-severity finding
											</div>
										</div>
										<div className="severity-bars">
											<div
												className={`sev-bar ${result.overall_severity === "Mild" ? "mild-active" : ""}`}
											/>
											<div
												className={`sev-bar ${result.overall_severity === "Moderate" ? "moderate-active" : ""}`}
											/>
											<div
												className={`sev-bar ${result.overall_severity === "Severe" ? "severe-active" : ""}`}
											/>
										</div>
									</div>
									<div className="severity-stats">
										<div className="stat-box">
											<div className="stat-count mild">
												{result.summary.Mild}
											</div>
											<div className="stat-label">Mild</div>
										</div>
										<div className="stat-box">
											<div className="stat-count moderate">
												{result.summary.Moderate}
											</div>
											<div className="stat-label">Moderate</div>
										</div>
										<div className="stat-box">
											<div className="stat-count severe">
												{result.summary.Severe}
											</div>
											<div className="stat-label">Severe</div>
										</div>
									</div>
								</div>
							</div>

							{/* ── Spatial Diagnostic Map ── */}
							<div className="card">
								<div className="card-header">
									<div className="card-header-left">
										<div className="card-header-icon blue">
											<svg
												fill="none"
												stroke="currentColor"
												viewBox="0 0 24 24"
												aria-label="Map icon"
											>
												<path
													strokeLinecap="round"
													strokeLinejoin="round"
													strokeWidth="2"
													d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7"
												/>
											</svg>
										</div>
										<span className="card-title">Spatial Diagnostic Map</span>
									</div>
								</div>
								<div className="card-body">
									<div className="spatial-map">
										<div className="map-grid" />
										<svg
											className="car-svg"
											viewBox="0 0 200 80"
											fill="none"
											stroke="currentColor"
											strokeWidth="1.3"
											aria-label="Car blueprint outline"
										>
											<path d="M38,28 C45,16 68,12 100,12 C132,12 155,16 162,28 L178,28 C188,28 192,34 192,40 C192,46 188,52 178,52 L162,52 C155,64 132,68 100,68 C68,68 45,64 38,52 L22,52 C12,52 8,46 8,40 C8,34 12,28 22,28 Z" />
											<path
												d="M72,20 L128,20 C140,20 150,28 150,40 C150,52 140,60 128,60 L72,60 C60,60 50,52 50,40 C50,28 60,20 72,20 Z"
												strokeDasharray="3 3"
												strokeOpacity="0.5"
											/>
											<path
												d="M183,28 L192,32 L192,36 L182,32 Z"
												fill="rgba(255,255,255,0.08)"
											/>
											<path
												d="M183,52 L192,48 L192,44 L182,48 Z"
												fill="rgba(255,255,255,0.08)"
											/>
										</svg>
										<div className="panel-indicators">
											<div className="panel-indicator">
												<div className="panel-indicator-label">Front</div>
												<div
													className={`panel-dot ${hasPanelDmg("Front", "Fender") ? "active-red" : ""}`}
												/>
											</div>
											<div className="panel-indicator">
												<div className="panel-indicator-label">Hood</div>
												<div
													className={`panel-dot ${hasPanelDmg("Hood", "Roof") ? "active-yellow" : ""}`}
												/>
											</div>
											<div className="panel-indicator">
												<div className="panel-indicator-label">Door</div>
												<div
													className={`panel-dot ${hasPanelDmg("Door") ? "active-orange" : ""}`}
												/>
											</div>
											<div className="panel-indicator">
												<div className="panel-indicator-label">Rear</div>
												<div
													className={`panel-dot ${hasPanelDmg("Rear", "Trunk") ? "active-red" : ""}`}
												/>
											</div>
										</div>
									</div>
									<div className="map-caption">
										Pulsing beacons indicate affected panels
									</div>
								</div>
							</div>

							{/* ── Damage Findings List ── */}
							<div className="card">
								<div className="card-header">
									<div className="card-header-left">
										<div className="card-header-icon blue">
											<svg
												fill="none"
												stroke="currentColor"
												viewBox="0 0 24 24"
												aria-label="Findings alert icon"
											>
												<path
													strokeLinecap="round"
													strokeLinejoin="round"
													strokeWidth="2"
													d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
												/>
											</svg>
										</div>
										<span className="card-title">Detected Findings</span>
									</div>
									<span className="card-tag">
										{result.damages.length} found
									</span>
								</div>
								<div className="card-body">
									<div className="damage-list">
										{result.damages.map((d) => {
											const isH = hoverId === d.id;
											const label = d.class
												.replace("_", " ")
												.replace(/\b\w/g, (c) => c.toUpperCase());
											const color = CLASS_COLORS[d.class] ?? "#22c55e";
											const badgeCls =
												d.severity === "Severe"
													? "badge-severe"
													: d.severity === "Moderate"
														? "badge-moderate"
														: "badge-mild";
											return (
												<div
													key={d.id}
													className={`damage-item${isH ? " hovered" : ""}`}
													onMouseEnter={() => setHoverId(d.id)}
													onMouseLeave={() => setHoverId(null)}
												>
													<div className="damage-item-top">
														<div className="damage-item-name">
															<div
																className="damage-dot"
																style={{ background: color }}
															/>
															<span className="damage-name-text">{label}</span>
														</div>
														<span className={`badge ${badgeCls}`}>
															{d.severity}
														</span>
													</div>

													<div className="damage-panel-badge">
														<svg
															fill="none"
															stroke="currentColor"
															viewBox="0 0 24 24"
															aria-label="Location pin icon"
														>
															<path
																strokeLinecap="round"
																strokeLinejoin="round"
																strokeWidth="2"
																d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
															/>
														</svg>
														{d.panel}
													</div>

													<div className="damage-metrics">
														{d.metrics.length_cm != null && (
															<div className="metric-row">
																<span className="metric-label">Length</span>
																<span className="metric-value">
																	{d.metrics.length_cm} cm
																</span>
															</div>
														)}
														{d.metrics.depth_cm != null && (
															<div className="metric-row">
																<span className="metric-label">Depth</span>
																<span className="metric-value">
																	{d.metrics.depth_cm} cm
																</span>
															</div>
														)}
														{d.metrics.width_cm != null && (
															<div className="metric-row">
																<span className="metric-label">Width</span>
																<span className="metric-value">
																	{d.metrics.width_cm} cm
																</span>
															</div>
														)}
														{d.metrics.cm2_area != null && (
															<div className="metric-row">
																<span className="metric-label">Area</span>
																<span className="metric-value">
																	{d.metrics.cm2_area} cm²
																</span>
															</div>
														)}
														<div className="metric-row">
															<span className="metric-label">Confidence</span>
															<span className="metric-value blue">
																{Math.round(d.confidence * 100)}%
															</span>
														</div>
													</div>
												</div>
											);
										})}
									</div>
								</div>
							</div>
						</div>
					</div>
				)}
			</main>
		</div>
	);
}
