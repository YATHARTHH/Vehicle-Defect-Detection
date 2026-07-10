import React, { useState } from 'react';

interface Damage {
  id: number;
  class: string;
  confidence: number;
  box: number[];
  severity: string;
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

interface AnalysisResults {
  success: boolean;
  overall_severity: string;
  summary: {
    Mild: number;
    Moderate: number;
    Severe: number;
  };
  calibration: {
    cm_per_pixel: number;
    reference_found: boolean;
  };
  damages: Damage[];
  annotated_image: string;
  repair_guide: string;
}

const renderMarkdown = (text: string) => {
  if (!text) return null;
  const lines = text.split('\n');
  let inTable = false;
  let tableHeaders: string[] = [];
  
  return lines.map((line, idx) => {
    const trimmed = line.trim();
    if (trimmed.startsWith('# ')) {
      inTable = false;
      return <h2 key={idx} className="text-xl font-bold text-white mt-6 mb-3">{trimmed.substring(2)}</h2>;
    }
    if (trimmed.startsWith('## ')) {
      inTable = false;
      return <h2 key={idx} className="text-lg font-bold text-white mt-5 mb-2 border-b border-gray-800 pb-1">{trimmed.substring(3)}</h2>;
    }
    if (trimmed.startsWith('### ')) {
      inTable = false;
      return <h3 key={idx} className="text-md font-bold text-gray-200 mt-4 mb-2">{trimmed.substring(4)}</h3>;
    }
    if (trimmed.startsWith('#### ')) {
      inTable = false;
      return <h4 key={idx} className="text-sm font-semibold text-blue-400 mt-3 mb-1">{trimmed.substring(5)}</h4>;
    }
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      inTable = false;
      const content = trimmed.substring(2);
      const parts = content.split('**');
      const parsedContent = parts.map((part, pIdx) => 
        pIdx % 2 === 1 ? <strong key={pIdx} className="text-white font-semibold">{part}</strong> : part
      );
      return <li key={idx} className="ml-4 list-disc mb-2 text-gray-300 text-sm">{parsedContent}</li>;
    }
    if (trimmed.startsWith('|')) {
      const cells = trimmed.split('|').map(c => c.trim()).filter(c => c !== '');
      if (cells.every(c => c.startsWith(':') || c.startsWith('-') || c.endsWith('-'))) {
        return null;
      }
      
      if (!inTable) {
        inTable = true;
        tableHeaders = cells;
        return (
          <div key={idx} className="overflow-x-auto my-4 rounded-xl border border-gray-800">
            <table className="min-w-full divide-y divide-gray-800 bg-gray-950/40">
              <thead>
                <tr>
                  {tableHeaders.map((h, hIdx) => (
                    <th key={hIdx} className="px-4 py-3 text-left text-xs font-bold text-gray-300 uppercase tracking-wider bg-gray-900/60">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-900"></tbody>
            </table>
          </div>
        );
      }
      
      return (
        <tr key={idx} className="hover:bg-gray-900/40 transition">
          {cells.map((cell, cIdx) => {
            const parts = cell.split('**');
            const parsedCell = parts.map((part, pIdx) => 
              pIdx % 2 === 1 ? <strong key={pIdx} className="text-white">{part}</strong> : part
            );
            return <td key={cIdx} className="px-4 py-2.5 text-xs text-gray-300">{parsedCell}</td>;
          })}
        </tr>
      );
    }
    
    if (trimmed !== '') {
      inTable = false;
      const parts = trimmed.split('**');
      const parsedText = parts.map((part, pIdx) => 
        pIdx % 2 === 1 ? <strong key={pIdx} className="text-white font-semibold">{part}</strong> : part
      );
      return <p key={idx} className="text-sm text-gray-400 mb-3 leading-relaxed">{parsedText}</p>;
    }
    return null;
  });
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('');
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeImageTab, setActiveImageTab] = useState<'annotated' | 'original' | 'restored'>('annotated');
  const [hoveredDamageId, setHoveredDamageId] = useState<number | null>(null);

  // Default key matching the secure backend setting
  const SECURE_API_KEY = 'overbody_secure_key_2026';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResults(null);
    }
  };

  const uploadAndAnalyze = async (selectedFile: File) => {
    setLoading(true);
    setError(null);
    
    const stages = [
      'Extracting visual edges and contours...',
      'Locating potential surface deformities...',
      'Measuring physical scale dimensions (pixel-to-cm)...',
      'Running AI damage severity classifier...',
      'Generating part replacement suggestions with Gemini...'
    ];
    
    let stageIdx = 0;
    setLoadingStage(stages[0]);
    const stageInterval = setInterval(() => {
      stageIdx++;
      if (stageIdx < stages.length) {
        setLoadingStage(stages[stageIdx]);
      }
    }, 1200);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: {
          'X-API-Key': SECURE_API_KEY
        },
        body: formData,
      });

      if (response.status === 429) {
        throw new Error('API Rate limit exceeded. Please wait a moment before trying again.');
      }

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to analyze the image');
      }

      const data: AnalysisResults = await response.json();
      setResults(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Connecting to backend API failed. Make sure the FastAPI server is running on port 8000.');
    } finally {
      clearInterval(stageInterval);
      setLoading(false);
    }
  };

  const triggerUpload = () => {
    if (file) {
      uploadAndAnalyze(file);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    setActiveImageTab('annotated');
  };

  const handleExportPDF = async () => {
    if (!results) return;
    try {
      const response = await fetch('http://localhost:8000/api/export-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': SECURE_API_KEY
        },
        body: JSON.stringify(results)
      });
      if (response.ok) {
        const html = await response.text();
        const win = window.open('', '_blank');
        if (win) {
          win.document.write(html);
          win.document.close();
        }
      } else {
        alert("Failed to generate report export.");
      }
    } catch (err) {
      console.error(err);
      alert("Error contacting export service.");
    }
  };

  const handleUseSample = () => {
    setLoading(true);
    setLoadingStage('Loading sample data...');
    setTimeout(() => {
      setResults({
        success: true,
        overall_severity: 'Severe',
        summary: { Mild: 1, Moderate: 1, Severe: 1 },
        calibration: { cm_per_pixel: 0.04, reference_found: true },
        damages: [
          {
            id: 1,
            class: 'scratch',
            confidence: 0.89,
            box: [120, 200, 300, 30],
            severity: 'Moderate',
            metrics: { pixel_area: 9000, cm2_area: 14.4, length_cm: 12.0 }
          },
          {
            id: 2,
            class: 'dent',
            confidence: 0.94,
            box: [450, 150, 180, 180],
            severity: 'Severe',
            metrics: { pixel_area: 32400, cm2_area: 51.8, depth_cm: 2.8, width_cm: 7.2, height_cm: 7.2 }
          },
          {
            id: 3,
            class: 'rust',
            confidence: 0.76,
            box: [280, 420, 110, 80],
            severity: 'Mild',
            metrics: { pixel_area: 8800, cm2_area: 14.1, spread_area_cm2: 14.1 }
          }
        ],
        annotated_image: 'https://images.unsplash.com/photo-1619642751034-765dfdf7c58e?auto=format&fit=crop&q=80&w=800',
        repair_guide: `## Vehicle Overbody Repair Report (Sample Mode)

### 1. Overall Condition Assessment
- **Overall Severity Rating:** Severe Damage Detected (Urgent action advised)
- **Primary Finding:** Deep dent (2.8cm depth) on side panel, alongside moderate paint scratches and mild rust forming near wheel arch.

### 2. Action Plan per Finding

#### Finding #1: Dent (Severe Severity)
- **Steps:** The dent has a depth of 2.8cm which stretches beyond simple suction pulling. Requires inner panel access using professional Paintless Dent Repair (PDR) steel rods. Alternatively, file down paint, weld puller pins to draw out panel structure, apply Bondo body filler, sand flat with block sander, and prepare for multi-coat paint blend.

#### Finding #2: Scratch (Moderate Severity)
- **Steps:** Clean scratch length. Sand outer clear coat down with 2000-grit wet sandpaper. Apply a primer layer, spray color-matched base paint, and seal with a high-durability 2K gloss clear coat. Buffer to match surrounding panel sheen.

#### Finding #3: Rust (Mild Severity)
- **Steps:** Sand rust area down to bare steel. Apply phosphoric acid rust converter to neutralize remaining microscopic oxidation. Prime with zinc-rich primer before painting.

### 3. Recommended Tools & Parts
- **Tools:** Dual-Action sander, block sander, slide hammer pin welder, PDR rods.
- **Supplies:** Bondo body filler, 2K Clear coat spray, color-matched paint, prep-grease wipes, 2000-grit wet sand-paper.

### 4. Estimated Cost Range
| Damage Item | Est. DIY Cost | Est. Professional Cost |
| :--- | :--- | :--- |
| Severe Panel Dent | $120 - $250 | $800 - $1800 |
| Moderate Scratch | $60 - $110 | $400 - $650 |
| Mild Wheel-Arch Rust | $45 - $90 | $350 - $600 |
`
      });
      setPreviewUrl('https://images.unsplash.com/photo-1619642751034-765dfdf7c58e?auto=format&fit=crop&q=80&w=800');
      setLoading(false);
    }, 1500);
  };

  // Helper to check if a damage class is present
  const hasDamage = (className: string) => {
    return results?.damages.some(d => d.class === className) ?? false;
  };

  return (
    <div className="min-h-screen text-gray-100 flex flex-col pb-12">
      {/* Navbar */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="bg-blue-600 p-2 rounded-lg text-white shadow-lg shadow-blue-500/30">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path>
            </svg>
          </div>
          <div>
            <span className="font-extrabold text-xl tracking-tight text-white">Overbody Damage</span>
            <span className="text-xs block text-blue-400 font-semibold tracking-wider uppercase">Detection & Severity Advisor</span>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <button onClick={handleUseSample} className="text-sm font-semibold text-gray-300 hover:text-white transition bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded-lg">
            Demo Sample
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 mt-8 flex-grow">
        
        {/* Upload State / Landing Page */}
        {!previewUrl && (
          <div className="max-w-2xl mx-auto mt-16 text-center animate-fade-in">
            <h1 className="text-4xl font-extrabold text-white tracking-tight sm:text-5xl mb-4">
              Automated Car Damage <span className="gradient-text">Severity Assessment</span>
            </h1>
            <p className="text-lg text-gray-400 mb-8 max-w-xl mx-auto">
              Upload a single photo of your vehicle panel. Our CV pipeline localizes scratches, dents, and rust, estimates physical dimensions, and generates an AI repair guide.
            </p>

            <div className="glass-panel p-12 border-dashed border-2 border-gray-700 hover:border-blue-500/50 cursor-pointer transition relative group">
              <input 
                type="file" 
                accept="image/*"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <div className="flex flex-col items-center space-y-4">
                <div className="bg-gray-800 p-5 rounded-full text-gray-400 group-hover:text-blue-400 group-hover:bg-blue-500/10 transition duration-300">
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                  </svg>
                </div>
                <div className="text-gray-200">
                  <span className="font-semibold text-blue-400 hover:underline">Click to upload</span> or drag and drop image
                </div>
                <div className="text-xs text-gray-500">Supports PNG, JPG, JPEG, WEBP</div>
              </div>
            </div>

            <div className="mt-8 flex justify-center space-x-4">
              <button onClick={handleUseSample} className="btn-primary">
                Try with a Sample Image
              </button>
            </div>
          </div>
        )}

        {/* Selected Image but not analyzed yet */}
        {previewUrl && !results && !loading && (
          <div className="max-w-xl mx-auto glass-panel p-6 animate-fade-in">
            <h3 className="font-semibold text-lg text-white mb-4">Selected Image</h3>
            <img src={previewUrl} alt="Selected preview" className="w-full rounded-lg object-contain max-h-96 bg-black/40 mb-6 border border-gray-800" />
            <div className="flex space-x-4">
              <button onClick={handleReset} className="flex-1 bg-gray-800 hover:bg-gray-700 text-gray-300 font-semibold py-3 px-4 rounded-lg transition">
                Change Image
              </button>
              <button onClick={triggerUpload} className="flex-1 btn-primary">
                Run Assessment
              </button>
            </div>
          </div>
        )}

        {/* Loading Spinner */}
        {loading && (
          <div className="max-w-md mx-auto mt-24 text-center glass-panel p-8 animate-fade-in">
            <div className="inline-block relative w-16 h-16 mb-4">
              <div className="absolute inset-0 border-4 border-blue-500/20 rounded-full"></div>
              <div className="absolute inset-0 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
            <h3 className="font-bold text-lg text-white mb-2">Analyzing Vehicle Surface</h3>
            <p className="text-sm text-gray-400 animate-pulse">{loadingStage}</p>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="max-w-xl mx-auto mt-8 bg-red-900/20 border border-red-500/50 p-4 rounded-xl text-red-200 text-center animate-fade-in">
            <svg className="w-8 h-8 mx-auto text-red-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
            </svg>
            <p className="font-semibold">{error}</p>
            <button onClick={handleReset} className="mt-4 bg-red-900/40 hover:bg-red-900/60 border border-red-500/30 text-white text-xs font-semibold px-4 py-2 rounded-lg transition">
              Reset and Try Again
            </button>
          </div>
        )}

        {/* Results Dashboard */}
        {results && (
          <div className="dashboard-grid animate-fade-in">
            
            {/* Left Side: Images & Interactive Visualizer */}
            <div className="space-y-6">
              <div className="glass-panel p-4 flex flex-col h-full">
                
                {/* Visualizer Header Tabs */}
                <div className="flex items-center justify-between border-b border-gray-800 pb-3 mb-4">
                  <div className="flex items-center space-x-2">
                    <span className="font-bold text-gray-100">Surface Visualizer</span>
                    <span className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded border border-gray-700">
                      {results.calibration.reference_found ? 'Calibrated (cm)' : 'Approx. Scale'}
                    </span>
                  </div>
                  <div className="flex space-x-2 bg-gray-900 p-1 rounded-lg border border-gray-800">
                    <button 
                      onClick={() => setActiveImageTab('annotated')} 
                      className={`text-xs px-3 py-1.5 rounded-md font-semibold transition ${activeImageTab === 'annotated' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}
                    >
                      Detections
                    </button>
                    <button 
                      onClick={() => setActiveImageTab('original')} 
                      className={`text-xs px-3 py-1.5 rounded-md font-semibold transition ${activeImageTab === 'original' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-white'}`}
                    >
                      Original
                    </button>
                    <button 
                      onClick={() => setActiveImageTab('restored')} 
                      className={`text-xs px-3 py-1.5 rounded-md font-semibold transition ${activeImageTab === 'restored' ? 'bg-emerald-600 text-white shadow' : 'text-gray-400 hover:text-emerald-400'}`}
                      title="Simulates damage removal restoration"
                    >
                      Restored Preview
                    </button>
                  </div>
                </div>

                {/* Display Image with interactive overlays */}
                <div className="relative flex-grow flex items-center justify-center bg-black/60 rounded-xl overflow-hidden min-h-[350px] max-h-[500px] border border-gray-800">
                  <img 
                    src={activeImageTab === 'annotated' ? results.annotated_image : previewUrl!} 
                    alt="Vehicle scan" 
                    className="max-w-full max-h-full object-contain transition-all duration-300"
                    style={{
                      // When restored is active, hide annotations and apply clean-up gloss filter
                      filter: activeImageTab === 'restored' ? 'brightness(1.03) contrast(0.97) saturate(1.05)' : 'none'
                    }}
                  />
                  
                  {/* Before/After sliding indicator overlay for restored */}
                  {activeImageTab === 'restored' && (
                    <div className="absolute inset-0 bg-blue-500/5 pointer-events-none flex items-center justify-center border-4 border-dashed border-emerald-500/20">
                      <span className="bg-emerald-950/80 border border-emerald-500 text-emerald-400 text-[10px] font-bold tracking-wider px-3 py-1 rounded-full uppercase backdrop-blur-sm shadow-lg">
                        ✨ Simulated Restoration Active (Clean View)
                      </span>
                    </div>
                  )}

                  {/* Interactive Box Highlight Overlays on Hover */}
                  {activeImageTab === 'annotated' && results.damages.map(d => {
                    const colors: Record<string, string> = {
                      scratch: 'border-cyan-400',
                      dent: 'border-fuchsia-400',
                      crack: 'border-yellow-400',
                      rust: 'border-orange-500',
                      glass_shatter: 'border-sky-400',
                      broken_lamp: 'border-amber-400'
                    };
                    const borderColor = colors[d.class] || 'border-green-400';
                    const isHovered = hoveredDamageId === d.id;

                    return (
                      <div 
                        key={d.id}
                        className={`absolute border-2 ${borderColor} transition-all duration-200 pointer-events-none ${isHovered ? 'bg-white/10 ring-4 ring-offset-2 ring-blue-500' : 'opacity-0'}`}
                      />
                    );
                  })}
                </div>

                <div className="mt-4 flex justify-between items-center text-xs text-gray-500">
                  <span>💡 Tip: Click 'Restored Preview' to simulate the vehicle after repairs.</span>
                  <button onClick={handleReset} className="text-blue-400 hover:underline font-semibold">
                    Analyze another image
                  </button>
                </div>
              </div>

              {/* AI Repair Advice (Gemini Output) */}
              <div className="glass-panel p-6">
                <div className="flex items-center justify-between border-b border-gray-800 pb-3 mb-4">
                  <div className="flex items-center space-x-2">
                    <div className="bg-purple-600/20 text-purple-400 p-2 rounded-lg">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                      </svg>
                    </div>
                    <h3 className="font-bold text-lg text-white">AI-Powered Repair Guidance</h3>
                  </div>
                  <button 
                    onClick={handleExportPDF}
                    className="text-xs bg-blue-600/20 hover:bg-blue-600/40 text-blue-300 font-bold border border-blue-500/30 px-3 py-1.5 rounded-lg transition"
                  >
                    Print Report / Save PDF
                  </button>
                </div>
                <div className="markdown-content text-sm leading-relaxed text-gray-300">
                  {renderMarkdown(results.repair_guide)}
                </div>
              </div>
            </div>

            {/* Right Side: Metrics, Ratings, list of Findings */}
            <div className="space-y-6">
              
              {/* Overall Severity Dashboard Summary Card */}
              <div className="glass-panel p-6">
                <h3 className="font-bold text-gray-400 text-sm uppercase tracking-wider mb-4">Overall Vehicle Condition</h3>
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-3xl font-extrabold text-white">
                      {results.overall_severity === 'Severe' && <span className="text-red-500">Severe Damage</span>}
                      {results.overall_severity === 'Moderate' && <span className="text-yellow-500">Moderate Damage</span>}
                      {results.overall_severity === 'Mild' && <span className="text-green-500">Mild Damage</span>}
                      {results.overall_severity === 'Good' && <span className="text-emerald-400">Excellent</span>}
                    </h2>
                    <span className="text-xs text-gray-500 block mt-1">Overall condition score based on highest severity finding</span>
                  </div>
                  <div className="flex space-x-1">
                    <div className={`w-3 h-8 rounded-full ${results.overall_severity === 'Mild' ? 'bg-green-500 shadow-md shadow-green-500/50' : 'bg-gray-800'}`}></div>
                    <div className={`w-3 h-8 rounded-full ${results.overall_severity === 'Moderate' ? 'bg-yellow-500 shadow-md shadow-yellow-500/50' : 'bg-gray-800'}`}></div>
                    <div className={`w-3 h-8 rounded-full ${results.overall_severity === 'Severe' ? 'bg-red-500 shadow-md shadow-red-500/50' : 'bg-gray-800'}`}></div>
                  </div>
                </div>
                
                <div className="mt-6 grid grid-cols-3 gap-2 text-center border-t border-gray-800 pt-4">
                  <div>
                    <span className="text-xs text-gray-500 block">Mild</span>
                    <span className="text-sm font-bold text-green-400">{results.summary.Mild}</span>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500 block">Moderate</span>
                    <span className="text-sm font-bold text-yellow-400">{results.summary.Moderate}</span>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500 block">Severe</span>
                    <span className="text-sm font-bold text-red-400">{results.summary.Severe}</span>
                  </div>
                </div>
              </div>

              {/* Spatial Damage Blueprint Mapping */}
              <div className="glass-panel p-6">
                <h3 className="font-bold text-white text-md border-b border-gray-800 pb-3 mb-4">Spatial Bumper Blueprint</h3>
                <div className="relative w-full h-40 bg-gray-950/40 rounded-xl flex items-center justify-center border border-gray-900 p-2">
                  
                  {/* Vector Outline SVG of a Car */}
                  <svg className="w-full h-full opacity-30 text-gray-500" viewBox="0 0 200 80" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M20,20 C40,20 60,10 80,10 C100,10 110,10 130,10 C150,10 160,20 180,20 L190,40 L180,60 C160,60 150,70 130,70 L80,70 C60,70 40,60 20,60 Z" />
                    {/* Headlights */}
                    <path d="M185,25 L190,30 L190,35 L182,30 Z" fill="currentColor" />
                    <path d="M185,55 L190,50 L190,45 L182,50 Z" fill="currentColor" />
                    {/* Windshield */}
                    <path d="M130,22 L110,25 L110,55 L130,58 Z" />
                    {/* Wheels */}
                    <circle cx="45" cy="65" r="10" fill="currentColor" />
                    <circle cx="155" cy="65" r="10" fill="currentColor" />
                  </svg>

                  {/* Absolute Indicators overlay matching detected damage classes */}
                  <div className="absolute inset-0 flex items-center justify-between px-6 pointer-events-none">
                    
                    {/* Rear Bumper Area */}
                    <div className="flex flex-col items-center space-y-1">
                      <span className="text-[9px] text-gray-500 uppercase tracking-widest">Rear Bumper</span>
                      <div className={`w-3.5 h-3.5 rounded-full transition ${hasDamage('glass_shatter') || hasDamage('broken_lamp') ? 'bg-red-500 animate-ping' : 'bg-gray-800'}`} />
                    </div>

                    {/* Side Panel Area */}
                    <div className="flex flex-col items-center space-y-1">
                      <span className="text-[9px] text-gray-500 uppercase tracking-widest">Side Panel</span>
                      <div className={`w-3.5 h-3.5 rounded-full transition ${hasDamage('scratch') || hasDamage('dent') ? 'bg-yellow-500 animate-pulse' : 'bg-gray-800'}`} />
                    </div>

                    {/* Front Bonnet/Lights Area */}
                    <div className="flex flex-col items-center space-y-1">
                      <span className="text-[9px] text-gray-500 uppercase tracking-widest">Front End</span>
                      <div className={`w-3.5 h-3.5 rounded-full transition ${hasDamage('rust') || hasDamage('crack') ? 'bg-orange-500 animate-pulse' : 'bg-gray-800'}`} />
                    </div>

                  </div>
                </div>
                <span className="text-[10px] text-gray-500 block text-center mt-2">Highlights represent active visual damages located on the chassis frame</span>
              </div>

              {/* List of Detected Findings */}
              <div className="glass-panel p-6">
                <div className="flex items-center justify-between border-b border-gray-800 pb-3 mb-4">
                  <h3 className="font-bold text-white text-md">Detected Damages ({results.damages.length})</h3>
                  <span className="text-xs text-gray-400">Hover item to highlight</span>
                </div>
                
                <div className="space-y-3">
                  {results.damages.map(d => {
                    const label = d.class.replace('_', ' ');
                    const isHovered = hoveredDamageId === d.id;
                    
                    return (
                      <div 
                        key={d.id}
                        onMouseEnter={() => setHoveredDamageId(d.id)}
                        onMouseLeave={() => setHoveredDamageId(null)}
                        className={`p-4 rounded-xl border transition duration-150 cursor-pointer ${isHovered ? 'bg-gray-800/80 border-blue-500/50 shadow' : 'bg-gray-900/40 border-gray-800/80 hover:bg-gray-900/80'}`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <div 
                              className="w-3 h-3 rounded-full" 
                              style={{ 
                                backgroundColor: d.class === 'scratch' ? '#22d3ee' : 
                                                 d.class === 'dent' ? '#e879f9' : 
                                                 d.class === 'crack' ? '#facc15' : 
                                                 d.class === 'rust' ? '#ea580c' : 
                                                 d.class === 'glass_shatter' ? '#38bdf8' : '#f59e0b'
                              }}
                            ></div>
                            <span className="font-bold text-sm text-gray-100">{label.replace(/\b\w/g, c => c.toUpperCase())}</span>
                          </div>
                          <span className={`badge ${d.severity === 'Severe' ? 'badge-severe' : d.severity === 'Moderate' ? 'badge-moderate' : 'badge-mild'}`}>
                            {d.severity}
                          </span>
                        </div>

                        <div className="grid grid-cols-2 gap-y-1 gap-x-4 text-xs text-gray-400 mt-2">
                          {d.metrics.length_cm && (
                            <div className="flex justify-between border-b border-gray-800/50 pb-1">
                              <span>Estimated Length:</span>
                              <span className="font-semibold text-gray-200">{d.metrics.length_cm} cm</span>
                            </div>
                          )}
                          {d.metrics.depth_cm && (
                            <div className="flex justify-between border-b border-gray-800/50 pb-1">
                              <span>Estimated Depth:</span>
                              <span className="font-semibold text-gray-200">{d.metrics.depth_cm} cm</span>
                            </div>
                          )}
                          {d.metrics.width_cm && (
                            <div className="flex justify-between border-b border-gray-800/50 pb-1">
                              <span>Estimated Width:</span>
                              <span className="font-semibold text-gray-200">{d.metrics.width_cm} cm</span>
                            </div>
                          )}
                          {d.metrics.cm2_area && (
                            <div className="flex justify-between border-b border-gray-800/50 pb-1">
                              <span>Affected Area:</span>
                              <span className="font-semibold text-gray-200">{d.metrics.cm2_area} cm²</span>
                            </div>
                          )}
                          <div className="flex justify-between border-b border-gray-800/50 pb-1">
                            <span>Detection Confidence:</span>
                            <span className="font-semibold text-blue-400">{Math.round(d.confidence * 100)}%</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>

          </div>
        )}
      </main>
    </div>
  );
}
