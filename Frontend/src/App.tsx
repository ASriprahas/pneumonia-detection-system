import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Activity,
  UploadCloud,
  History,
  LogOut,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Clock,
  FileImage,
  Eye,
  X,
  Loader2,
  UserPlus,
  LogIn,
  Palette,
  Sparkles,
} from 'lucide-react';

// Configure Axios
const api = axios.create({
  baseURL: 'http://localhost:8080/api',
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('pneumonia_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

interface ModelPrediction {
  id?: number;
  modelName: string;
  prediction: string;
  confidence: number;
  normalProbability?: number;
  pneumoniaProbability?: number;
  inferenceTimeMs?: number;
}

interface ScanRecord {
  id: number;
  originalFilename: string;
  storedImagePath: string;
  consensusPrediction: string;
  consensusConfidence: number;
  ensembleStrategy: string;
  totalInferenceTimeMs: number;
  gradcamHeatmap: string;
  createdAt: string;
  predictions: ModelPrediction[];
}

interface UserSession {
  token: string;
  username: string;
  fullName: string;
  role: string;
}

type ThemeOption = 'emerald' | 'amber' | 'ivory';

const THEMES: { id: ThemeOption; label: string; dot: string }[] = [
  { id: 'emerald', label: 'Emerald', dot: '#10b981' },
  { id: 'amber', label: 'Amber', dot: '#f59e0b' },
  { id: 'ivory', label: 'Ivory Light', dot: '#059669' },
];

export function App() {
  const [theme, setTheme] = useState<ThemeOption>(() => {
    return (localStorage.getItem('pneumoscan_theme') as ThemeOption) || 'emerald';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('pneumoscan_theme', theme);
  }, [theme]);

  const [user, setUser] = useState<UserSession | null>(() => {
    const saved = localStorage.getItem('pneumonia_user');
    return saved ? JSON.parse(saved) : null;
  });

  // Auth state
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);

  // Navigation & App state
  const [activeTab, setActiveTab] = useState<'analyze' | 'history'>('analyze');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState('');
  const [currentResult, setCurrentResult] = useState<ScanRecord | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);

  // History state
  const [history, setHistory] = useState<ScanRecord[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [modalScan, setModalScan] = useState<ScanRecord | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (user) {
      fetchHistory();
    }
  }, [user]);

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError('');
    setAuthLoading(true);
    try {
      const endpoint = isRegister ? '/auth/register' : '/auth/login';
      const payload = isRegister
        ? { username, password, fullName }
        : { username, password };

      const res = await api.post<UserSession>(endpoint, payload);
      localStorage.setItem('pneumonia_token', res.data.token);
      localStorage.setItem('pneumonia_user', JSON.stringify(res.data));
      setUser(res.data);
    } catch (err: any) {
      const msg =
        err.response?.data?.message ||
        err.response?.data?.error ||
        'Authentication failed. Check credentials and ensure Spring Boot is running on port 8080.';
      setAuthError(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('pneumonia_token');
    localStorage.removeItem('pneumonia_user');
    setShowLogoutConfirm(false);
    setUser(null);
    setCurrentResult(null);
    setSelectedFile(null);
    setPreviewUrl(null);
    setHistory([]);
  };

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setAnalyzeError('Please select a valid chest X-ray image (JPEG/PNG).');
      return;
    }
    setAnalyzeError('');
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setCurrentResult(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setAnalyzing(true);
    setAnalyzeError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await api.post<ScanRecord>('/predictions/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setCurrentResult(res.data);
      setShowHeatmap(true);
      fetchHistory();
    } catch (err: any) {
      if (err.response?.status === 401) {
        handleLogout();
      } else {
        setAnalyzeError(
          err.response?.data?.message ||
            'Inference failed. Ensure FastAPI (8000) and Spring Boot (8080) are running.'
        );
      }
    } finally {
      setAnalyzing(false);
    }
  };

  const fetchHistory = async () => {
    setLoadingHistory(true);
    try {
      const res = await api.get<ScanRecord[]>('/predictions/history');
      setHistory(res.data);
    } catch (err: any) {
      if (err.response?.status === 401) {
        handleLogout();
      }
    } finally {
      setLoadingHistory(false);
    }
  };

  const formatConfidence = (val?: number) => {
    if (val === undefined || val === null) return '0.0%';
    const pct = val <= 1 ? val * 100 : val;
    return `${pct.toFixed(1)}%`;
  };

  const getConfidenceWidth = (val?: number) => {
    if (val === undefined || val === null) return '0%';
    const pct = val <= 1 ? val * 100 : val;
    return `${Math.min(100, Math.max(0, pct))}%`;
  };

  const formatModelMeta = (name: string) => {
    const lower = name.toLowerCase();
    if (lower.includes('resnet')) {
      return {
        title: 'ResNet50 (Deep Residual)',
        badge: 'Test Acc: 83.17%',
      };
    }
    if (lower.includes('pneumonia')) {
      return {
        title: 'PneumoniaNet (Custom CNN)',
        badge: 'Test Acc: 82.05%',
      };
    }
    if (lower.includes('vgg')) {
      return {
        title: 'VGG16 (Transfer Learning)',
        badge: 'Test Acc: 79.17%',
      };
    }
    return { title: name.toUpperCase(), badge: '' };
  };

  const ThemeSwitcher = () => (
    <div
      className="flex items-center gap-1 p-1 rounded-xl border theme-surface"
      title="Switch Color Theme"
    >
      <Palette
        className="w-3.5 h-3.5 ml-1.5 mr-0.5"
        style={{ color: 'var(--accent)' }}
      />
      {THEMES.map((t) => {
        const active = theme === t.id;
        return (
          <button
            key={t.id}
            type="button"
            onClick={() => setTheme(t.id)}
            className={`px-2 py-1 rounded-lg text-[11px] font-medium flex items-center gap-1.5 transition cursor-pointer ${
              active ? 'shadow-sm' : 'opacity-70 hover:opacity-100'
            }`}
            style={
              active
                ? {
                    backgroundColor: 'var(--accent-soft)',
                    color: 'var(--accent)',
                    border: '1px solid var(--border-card)',
                  }
                : { color: 'var(--text-muted)' }
            }
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: t.dot }}
            />
            <span className="hidden sm:inline">{t.label}</span>
          </button>
        );
      })}
    </div>
  );

  // ==================== LOGIN / REGISTER VIEW ====================
  if (!user) {
    return (
      <div
        className="min-h-screen flex flex-col items-center justify-center p-4 relative"
        style={{ backgroundColor: 'var(--bg-main)', color: 'var(--text-main)' }}
      >
        <div className="absolute top-5 right-5">
          <ThemeSwitcher />
        </div>

        <div className="w-full max-w-md theme-card rounded-2xl shadow-2xl p-8 animate-fade-up">
          <div className="flex items-center justify-center gap-3 mb-6">
            <div
              className="p-3 rounded-xl animate-pulse-subtle"
              style={{
                backgroundColor: 'var(--accent-soft)',
                border: '1px solid var(--border-card)',
                color: 'var(--accent)',
              }}
            >
              <Activity className="w-7 h-7" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                Pneumonia Detection
              </h1>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Multi-Model Chest X-Ray Diagnostics
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 p-1 rounded-xl border theme-surface mb-6">
            <button
              type="button"
              onClick={() => {
                setIsRegister(false);
                setAuthError('');
              }}
              className={`py-2 text-sm font-medium rounded-lg transition cursor-pointer ${
                !isRegister ? 'btn-accent' : 'opacity-75 hover:opacity-100'
              }`}
              style={isRegister ? { color: 'var(--text-muted)' } : undefined}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => {
                setIsRegister(true);
                setAuthError('');
              }}
              className={`py-2 text-sm font-medium rounded-lg transition cursor-pointer ${
                isRegister ? 'btn-accent' : 'opacity-75 hover:opacity-100'
              }`}
              style={!isRegister ? { color: 'var(--text-muted)' } : undefined}
            >
              Register
            </button>
          </div>

          {authError && (
            <div className="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-2 animate-fade-up">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{authError}</span>
            </div>
          )}

          <form onSubmit={handleAuthSubmit} className="space-y-4">
            {isRegister && (
              <div>
                <label
                  className="block text-xs font-medium mb-1.5"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Patient Full Name
                </label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Enter patient name"
                  className="w-full px-3.5 py-2.5 rounded-xl text-sm theme-input"
                />
              </div>
            )}

            <div>
              <label
                className="block text-xs font-medium mb-1.5"
                style={{ color: 'var(--text-muted)' }}
              >
                Username
              </label>
              <input
                type="text"
                required
                minLength={3}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
                className="w-full px-3.5 py-2.5 rounded-xl text-sm theme-input"
              />
            </div>

            <div>
              <label
                className="block text-xs font-medium mb-1.5"
                style={{ color: 'var(--text-muted)' }}
              >
                Password
              </label>
              <input
                type="password"
                required
                minLength={6}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Min 6 characters"
                className="w-full px-3.5 py-2.5 rounded-xl text-sm theme-input"
              />
            </div>

            <button
              type="submit"
              disabled={authLoading}
              className="w-full py-2.5 btn-accent disabled:opacity-50 rounded-xl text-sm flex items-center justify-center gap-2 cursor-pointer"
            >
              {authLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : isRegister ? (
                <>
                  <UserPlus className="w-4 h-4" /> Create Account
                </>
              ) : (
                <>
                  <LogIn className="w-4 h-4" /> Sign In to Workstation
                </>
              )}
            </button>
          </form>
        </div>
      </div>
    );
  }

  // ==================== MAIN CLINICAL WORKSTATION ====================
  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: 'var(--bg-main)', color: 'var(--text-main)' }}
    >
      {/* Top Navbar */}
      <header
        className="border-b backdrop-blur-md sticky top-0 z-30 transition-colors duration-300"
        style={{
          backgroundColor: 'var(--bg-card)',
          borderColor: 'var(--border-card)',
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-2">
          <div className="flex items-center gap-3">
            <div
              className="p-2 rounded-lg animate-pulse-subtle"
              style={{
                backgroundColor: 'var(--accent-soft)',
                border: '1px solid var(--border-card)',
                color: 'var(--accent)',
              }}
            >
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <span className="font-bold tracking-tight">
                Pneumonia Detection
              </span>
              <span
                className="hidden xl:inline-block ml-2 text-xs px-2.5 py-0.5 rounded-full border"
                style={{
                  backgroundColor: 'var(--accent-soft)',
                  color: 'var(--accent)',
                  borderColor: 'var(--border-card)',
                }}
              >
                ResNet50 • VGG16 • PneumoniaNet
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-3">
            <ThemeSwitcher />

            <nav className="flex p-1 rounded-xl border theme-surface">
              <button
                onClick={() => setActiveTab('analyze')}
                className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium flex items-center gap-1.5 transition cursor-pointer ${
                  activeTab === 'analyze' ? 'btn-accent' : 'hover:opacity-100 opacity-75'
                }`}
                style={
                  activeTab !== 'analyze'
                    ? { color: 'var(--text-muted)' }
                    : undefined
                }
              >
                <UploadCloud className="w-4 h-4" />
                <span>Diagnostics</span>
              </button>
              <button
                onClick={() => {
                  setActiveTab('history');
                  fetchHistory();
                }}
                className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium flex items-center gap-1.5 transition cursor-pointer ${
                  activeTab === 'history' ? 'btn-accent' : 'hover:opacity-100 opacity-75'
                }`}
                style={
                  activeTab !== 'history'
                    ? { color: 'var(--text-muted)' }
                    : undefined
                }
              >
                <History className="w-4 h-4" />
                <span>History ({history.length})</span>
              </button>
            </nav>

            <div
              className="hidden lg:flex items-center gap-2 text-xs border-l pl-3"
              style={{
                color: 'var(--text-muted)',
                borderColor: 'var(--border-card)',
              }}
            >
              <ShieldCheck
                className="w-4 h-4"
                style={{ color: 'var(--accent)' }}
              />
              <span>{user.fullName || user.username}</span>
            </div>

            <button
              onClick={() => setShowLogoutConfirm(true)}
              title="Sign Out"
              className="p-2 rounded-lg transition cursor-pointer hover:text-rose-400"
              style={{ color: 'var(--text-muted)' }}
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Top-Center Sign Out Confirmation Popup */}
      {showLogoutConfirm && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-xs z-50 flex items-start justify-center pt-16 p-4"
          onClick={() => setShowLogoutConfirm(false)}
        >
          <div
            className="theme-card rounded-2xl max-w-sm w-full p-5 shadow-2xl animate-fade-up"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start gap-3">
              <div className="p-2.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-500 shrink-0">
                <LogOut className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <h3 className="text-base font-bold">Confirm Sign Out</h3>
                <p
                  className="text-xs mt-1 leading-relaxed"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Are you sure you want to sign out of{' '}
                  <span className="font-semibold" style={{ color: 'var(--text-main)' }}>
                    {user.fullName || user.username}
                  </span>
                  ?
                </p>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2.5 mt-5">
              <button
                type="button"
                onClick={() => setShowLogoutConfirm(false)}
                className="px-4 py-2 rounded-xl text-xs font-medium border theme-surface cursor-pointer hover:opacity-80 transition"
                style={{ color: 'var(--text-main)' }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleLogout}
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white shadow-md cursor-pointer transition"
              >
                Sign Out
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8">
        {activeTab === 'analyze' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Column: Upload & Preview */}
            <div className="lg:col-span-5 space-y-6 animate-fade-up">
              <div className="theme-card rounded-2xl p-6">
                <div className="flex items-center justify-between mb-1">
                  <h2 className="text-base font-semibold">Chest X-Ray Input</h2>
                  <span
                    className="text-[11px] font-medium px-2 py-0.5 rounded-full flex items-center gap-1"
                    style={{
                      backgroundColor: 'var(--accent-soft)',
                      color: 'var(--accent)',
                    }}
                  >
                    <Sparkles className="w-3 h-3" /> GPU Ready
                  </span>
                </div>
                <p
                  className="text-xs mb-4"
                  style={{ color: 'var(--text-muted)' }}
                >
                  Upload a frontal PA/AP chest radiograph (JPEG or PNG) for multi-architecture screening.
                </p>

                <div
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault();
                    if (e.dataTransfer.files?.[0]) {
                      handleFileSelect(e.dataTransfer.files[0]);
                    }
                  }}
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-300 group theme-surface hover:scale-[1.01]"
                  style={{ borderColor: 'var(--border-card)' }}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files?.[0]) {
                        handleFileSelect(e.target.files[0]);
                      }
                    }}
                  />

                  {previewUrl ? (
                    <div className="space-y-3">
                      <div
                        className={`relative inline-block rounded-xl overflow-hidden ${
                          analyzing ? 'xray-scanning' : ''
                        }`}
                      >
                        <img
                          src={previewUrl}
                          alt="Selected Chest X-Ray"
                          className="max-h-72 mx-auto rounded-xl border object-contain bg-black"
                          style={{ borderColor: 'var(--border-card)' }}
                        />
                      </div>
                      <p
                        className="text-xs font-medium truncate"
                        style={{ color: 'var(--accent)' }}
                      >
                        {selectedFile?.name}
                      </p>
                      <p
                        className="text-[11px]"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        {analyzing
                          ? 'Scanning lung fields with neural ensemble...'
                          : 'Click or drag another image to replace'}
                      </p>
                    </div>
                  ) : (
                    <div className="py-10 space-y-3">
                      <div
                        className="w-12 h-12 rounded-full border flex items-center justify-center mx-auto group-hover:scale-110 transition-transform duration-300"
                        style={{
                          backgroundColor: 'var(--accent-soft)',
                          borderColor: 'var(--border-card)',
                          color: 'var(--accent)',
                        }}
                      >
                        <UploadCloud className="w-6 h-6" />
                      </div>
                      <div>
                        <p className="text-sm font-medium">
                          Drop chest X-ray here or click to browse
                        </p>
                        <p
                          className="text-xs mt-1"
                          style={{ color: 'var(--text-muted)' }}
                        >
                          Supports JPEG, PNG radiographs
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {analyzeError && (
                  <div className="mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-2 animate-fade-up">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{analyzeError}</span>
                  </div>
                )}

                <button
                  onClick={handleAnalyze}
                  disabled={!selectedFile || analyzing}
                  className="mt-5 w-full py-3 btn-accent disabled:opacity-40 disabled:cursor-not-allowed rounded-xl text-sm flex items-center justify-center gap-2 cursor-pointer"
                >
                  {analyzing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Running Ensemble & Grad-CAM...
                    </>
                  ) : (
                    <>
                      <Cpu className="w-4 h-4" />
                      Run Multi-Model Diagnosis
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Right Column: Diagnostic Results */}
            <div className="lg:col-span-7 space-y-6">
              {!currentResult ? (
                <div className="theme-card rounded-2xl p-12 text-center h-full flex flex-col items-center justify-center animate-fade-up">
                  <div
                    className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                    style={{
                      backgroundColor: 'var(--accent-soft)',
                      color: 'var(--accent)',
                    }}
                  >
                    <FileImage className="w-8 h-8" />
                  </div>
                  <h3 className="text-base font-semibold">
                    No Active Scan Results
                  </h3>
                  <p
                    className="text-xs max-w-sm mt-1.5 leading-relaxed"
                    style={{ color: 'var(--text-muted)' }}
                  >
                    Upload a chest X-ray on the left and click &ldquo;Run Multi-Model Diagnosis&rdquo; to inspect the diagnostic verdict, individual model breakdown, and Grad-CAM attention heatmaps.
                  </p>
                </div>
              ) : (
                <div className="space-y-6 animate-fade-up">
                  {/* Clear Binary Verdict Banner */}
                  {(() => {
                    const isPneumonia =
                      currentResult.consensusPrediction?.toUpperCase() ===
                      'PNEUMONIA';
                    const totalModels = currentResult.predictions?.length || 3;
                    const agreeingModels =
                      currentResult.predictions?.filter(
                        (p) =>
                          p.prediction?.toUpperCase() ===
                          currentResult.consensusPrediction?.toUpperCase()
                      ).length || 0;

                    return (
                      <div
                        className={`rounded-2xl p-6 border transition-all duration-300 ${
                          isPneumonia
                            ? 'bg-rose-500/10 border-rose-500/40'
                            : 'bg-emerald-500/10 border-emerald-500/40'
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-4">
                          <div>
                            <span
                              className="text-xs uppercase tracking-wider font-semibold"
                              style={{ color: 'var(--text-muted)' }}
                            >
                              Final Diagnostic Result
                            </span>
                            <div className="flex items-center gap-3 mt-1.5">
                              {isPneumonia ? (
                                <AlertTriangle className="w-9 h-9 text-rose-500 animate-pulse shrink-0" />
                              ) : (
                                <CheckCircle2 className="w-9 h-9 text-emerald-500 shrink-0" />
                              )}
                              <div>
                                <h2
                                  className={`text-2xl sm:text-3xl font-extrabold tracking-tight ${
                                    isPneumonia
                                      ? 'text-rose-500'
                                      : 'text-emerald-500'
                                  }`}
                                >
                                  {isPneumonia
                                    ? 'PNEUMONIA DETECTED'
                                    : 'NORMAL (NO PNEUMONIA)'}
                                </h2>
                                <p
                                  className="text-xs mt-0.5"
                                  style={{ color: 'var(--text-muted)' }}
                                >
                                  {isPneumonia
                                    ? 'Radiographic signs of pulmonary opacity identified.'
                                    : 'Clear lung fields with no pneumonia indicators.'}
                                </p>
                              </div>
                            </div>
                          </div>

                          <div className="text-right">
                            <span
                              className={`inline-block text-xs font-bold px-3 py-1 rounded-full ${
                                isPneumonia
                                  ? 'bg-rose-500/15 text-rose-500 border border-rose-500/30'
                                  : 'bg-emerald-500/15 text-emerald-500 border border-emerald-500/30'
                              }`}
                            >
                              {agreeingModels} of {totalModels} Models Agree
                            </span>
                            <div
                              className="text-xs flex items-center justify-end gap-1 mt-2"
                              style={{ color: 'var(--text-muted)' }}
                            >
                              <Clock className="w-3.5 h-3.5" />
                              <span>
                                {currentResult.totalInferenceTimeMs
                                  ? `Analyzed in ${currentResult.totalInferenceTimeMs.toFixed(0)} ms`
                                  : 'Completed'}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })()}

                  {/* Individual Model Breakdown */}
                  <div className="theme-card rounded-2xl p-6">
                    <h3 className="text-sm font-semibold mb-4">
                      Architecture-by-Architecture Breakdown
                    </h3>
                    <div className="space-y-4">
                      {currentResult.predictions?.map((pred, idx) => {
                        const isPneumonia =
                          pred.prediction?.toUpperCase() === 'PNEUMONIA';
                        const meta = formatModelMeta(pred.modelName);
                        return (
                          <div
                            key={idx}
                            className="p-4 rounded-xl border theme-surface transition-transform duration-200 hover:translate-x-1"
                          >
                            <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-medium">
                                  {meta.title}
                                </span>
                                {meta.badge && (
                                  <span
                                    className="text-[10px] font-medium px-2 py-0.5 rounded-full"
                                    style={{
                                      backgroundColor: 'var(--accent-soft)',
                                      color: 'var(--accent)',
                                    }}
                                  >
                                    {meta.badge}
                                  </span>
                                )}
                              </div>
                              <div className="flex items-center gap-2">
                                <span
                                  className={`text-xs font-bold px-2.5 py-0.5 rounded-full ${
                                    isPneumonia
                                      ? 'bg-rose-500/15 text-rose-500 border border-rose-500/30'
                                      : 'bg-emerald-500/15 text-emerald-500 border border-emerald-500/30'
                                  }`}
                                >
                                  {pred.prediction}
                                </span>
                                <span className="text-sm font-semibold">
                                  {formatConfidence(pred.confidence)}
                                </span>
                              </div>
                            </div>
                            <div className="w-full h-2 bg-black/20 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all duration-700 ease-out ${
                                  isPneumonia ? 'bg-rose-500' : 'bg-emerald-500'
                                }`}
                                style={{
                                  width: getConfidenceWidth(pred.confidence),
                                }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Grad-CAM Explainability Viewer */}
                  {currentResult.gradcamHeatmap && (
                    <div className="theme-card rounded-2xl p-6">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="text-sm font-semibold">
                            Grad-CAM Visual Explainability
                          </h3>
                          <p
                            className="text-xs"
                            style={{ color: 'var(--text-muted)' }}
                          >
                            Highlighted lung regions driving the diagnostic prediction
                          </p>
                        </div>
                        {previewUrl && (
                          <button
                            onClick={() => setShowHeatmap(!showHeatmap)}
                            className="px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer"
                            style={{
                              backgroundColor: 'var(--accent-soft)',
                              color: 'var(--accent)',
                              border: '1px solid var(--border-card)',
                            }}
                          >
                            {showHeatmap
                              ? 'Show Original X-Ray'
                              : 'Show Grad-CAM Overlay'}
                          </button>
                        )}
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {previewUrl && (
                          <div
                            className="bg-black rounded-xl p-2 border text-center"
                            style={{ borderColor: 'var(--border-card)' }}
                          >
                            <img
                              src={previewUrl}
                              alt="Original Radiograph"
                              className="max-h-64 mx-auto rounded-lg object-contain"
                            />
                            <span className="block text-[11px] text-neutral-400 mt-2">
                              Original Radiograph
                            </span>
                          </div>
                        )}
                        <div
                          className="bg-black rounded-xl p-2 border text-center"
                          style={{ borderColor: 'var(--accent)' }}
                        >
                          <img
                            src={
                              showHeatmap
                                ? currentResult.gradcamHeatmap
                                : previewUrl || currentResult.gradcamHeatmap
                            }
                            alt="Grad-CAM Attention Map"
                            className="max-h-64 mx-auto rounded-lg object-contain transition-opacity duration-300"
                          />
                          <span
                            className="block text-[11px] mt-2 font-medium"
                            style={{ color: 'var(--accent)' }}
                          >
                            Grad-CAM Activation Heatmap
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ) : (
          /* ==================== HISTORY TAB ==================== */
          <div className="theme-card rounded-2xl overflow-hidden animate-fade-up">
            <div
              className="p-6 border-b flex items-center justify-between"
              style={{ borderColor: 'var(--border-card)' }}
            >
              <div>
                <h2 className="text-base font-semibold">Diagnostic Scan Logs</h2>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Persisted MySQL prediction records for {user.username}
                </p>
              </div>
              <button
                onClick={fetchHistory}
                className="px-3 py-1.5 text-xs rounded-lg font-medium transition cursor-pointer"
                style={{
                  backgroundColor: 'var(--accent-soft)',
                  color: 'var(--accent)',
                  border: '1px solid var(--border-card)',
                }}
              >
                Refresh
              </button>
            </div>

            {loadingHistory ? (
              <div
                className="p-12 text-center text-sm flex items-center justify-center gap-2"
                style={{ color: 'var(--text-muted)' }}
              >
                <Loader2 className="w-4 h-4 animate-spin" /> Loading records...
              </div>
            ) : history.length === 0 ? (
              <div
                className="p-12 text-center text-sm"
                style={{ color: 'var(--text-muted)' }}
              >
                No scans recorded yet. Run your first X-ray analysis in the Diagnostics tab.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr
                      className="border-b text-[11px] uppercase tracking-wider theme-surface"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      <th className="py-3.5 px-4">Scan ID</th>
                      <th className="py-3.5 px-4">File Name</th>
                      <th className="py-3.5 px-4">Diagnostic Verdict</th>
                      <th className="py-3.5 px-4">Model Agreement</th>
                      <th className="py-3.5 px-4">Date</th>
                      <th className="py-3.5 px-4 text-right">Details</th>
                    </tr>
                  </thead>
                  <tbody
                    className="divide-y text-sm"
                    style={{ borderColor: 'var(--border-card)' }}
                  >
                    {history.map((item) => {
                      const isPneumonia =
                        item.consensusPrediction?.toUpperCase() === 'PNEUMONIA';
                      const totalM = item.predictions?.length || 3;
                      const agreeM =
                        item.predictions?.filter(
                          (p) =>
                            p.prediction?.toUpperCase() ===
                            item.consensusPrediction?.toUpperCase()
                        ).length || 0;
                      return (
                        <tr
                          key={item.id}
                          className="transition-colors hover:bg-black/10"
                        >
                          <td
                            className="py-3.5 px-4 font-mono text-xs"
                            style={{ color: 'var(--text-muted)' }}
                          >
                            #{item.id}
                          </td>
                          <td className="py-3.5 px-4 font-medium max-w-xs truncate">
                            {item.originalFilename}
                          </td>
                          <td className="py-3.5 px-4">
                            <span
                              className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                                isPneumonia
                                  ? 'bg-rose-500/15 text-rose-500 border border-rose-500/30'
                                  : 'bg-emerald-500/15 text-emerald-500 border border-emerald-500/30'
                              }`}
                            >
                              {isPneumonia ? 'PNEUMONIA' : 'NORMAL'}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 font-medium text-xs">
                            {agreeM} of {totalM} Models
                          </td>
                          <td
                            className="py-3.5 px-4 text-xs"
                            style={{ color: 'var(--text-muted)' }}
                          >
                            {item.createdAt
                              ? new Date(item.createdAt).toLocaleString()
                              : '—'}
                          </td>
                          <td className="py-3.5 px-4 text-right">
                            <button
                              onClick={() => setModalScan(item)}
                              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer"
                              style={{
                                backgroundColor: 'var(--accent-soft)',
                                color: 'var(--accent)',
                              }}
                            >
                              <Eye className="w-3.5 h-3.5" /> Inspect
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>

      {/* History Detail Modal */}
      {modalScan && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="theme-card rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto p-6 space-y-5 animate-fade-up">
            <div
              className="flex items-center justify-between border-b pb-4"
              style={{ borderColor: 'var(--border-card)' }}
            >
              <div>
                <h3 className="text-base font-bold">
                  Scan Report #{modalScan.id}
                </h3>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {modalScan.originalFilename}
                </p>
              </div>
              <button
                onClick={() => setModalScan(null)}
                className="p-1.5 rounded-lg theme-surface cursor-pointer hover:opacity-80"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-4 p-4 rounded-xl border theme-surface">
              <div>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Final Diagnosis
                </span>
                <p
                  className={`text-lg font-bold ${
                    modalScan.consensusPrediction?.toUpperCase() === 'PNEUMONIA'
                      ? 'text-rose-500'
                      : 'text-emerald-500'
                  }`}
                >
                  {modalScan.consensusPrediction?.toUpperCase() === 'PNEUMONIA'
                    ? 'PNEUMONIA DETECTED'
                    : 'NORMAL (NO PNEUMONIA)'}
                </p>
              </div>
              <div>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Model Agreement
                </span>
                <p className="text-lg font-bold">
                  {modalScan.predictions?.filter(
                    (p) =>
                      p.prediction?.toUpperCase() ===
                      modalScan.consensusPrediction?.toUpperCase()
                  ).length || 0}{' '}
                  of {modalScan.predictions?.length || 3} Models
                </p>
              </div>
            </div>

            <div className="space-y-2">
              <span className="text-xs font-semibold">Model Predictions</span>
              {modalScan.predictions?.map((p, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between text-xs px-3.5 py-2.5 rounded-lg border theme-surface"
                >
                  <span className="font-medium">
                    {formatModelMeta(p.modelName).title}
                  </span>
                  <span className="font-semibold">
                    {p.prediction} ({formatConfidence(p.confidence)})
                  </span>
                </div>
              ))}
            </div>

            {modalScan.gradcamHeatmap && (
              <div>
                <span className="block text-xs font-semibold mb-2">
                  Saved Grad-CAM Heatmap
                </span>
                <div
                  className="bg-black rounded-xl p-3 border text-center"
                  style={{ borderColor: 'var(--border-card)' }}
                >
                  <img
                    src={modalScan.gradcamHeatmap}
                    alt="Saved Grad-CAM"
                    className="max-h-64 mx-auto rounded-lg object-contain"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;