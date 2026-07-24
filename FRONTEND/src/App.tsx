import { useState, useEffect } from 'react'
import ChartComponent from './components/Chart'
import TickerDropdown, { TickerInfo } from './components/TickerDropdown'

function App() {
  const [ticker, setTicker] = useState('JKSE')
  const [windowSize, setWindowSize] = useState(60)
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const [tickers, setTickers] = useState<TickerInfo[]>([])
  
  useEffect(() => {
    const fetchTickers = async () => {
      try {
        const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const res = await fetch(`${baseUrl}/api/tickers`);
        if (res.ok) {
          const list = await res.json();
          setTickers(list);
        }
      } catch (err) {
        console.error("Gagal load tickers", err);
      }
    };
    fetchTickers();
  }, []);

  const fetchData = async (targetTicker: string, targetWindow: number) => {
    setLoading(true)
    setError(null)
    try {
      const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${baseUrl}/api/volatility?ticker=${targetTicker}&window=${targetWindow}`)
      
      if (!response.ok) {
        const isJson = response.headers.get("content-type")?.includes("application/json");
        const errDetail = isJson ? (await response.json()).detail : await response.text();
        throw new Error(isJson ? errDetail : `Mencoba memanggil: ${baseUrl}/api/volatility\nTetapi server membalas: ${errDetail.substring(0, 40)}...`);
      }
      
      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        const text = await response.text();
        throw new Error(`Target API (${baseUrl}) merespons HTML, bukan JSON. Cek VITE_API_URL. Respons server: ${text.substring(0, 40)}...`);
      }

      const json = await response.json()
      setData(json)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Initial fetch on mount
  useEffect(() => {
    fetchData(ticker, windowSize);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleTickerSelect = (newTicker: string) => {
    setTicker(newTicker);
    fetchData(newTicker, windowSize); // Auto trigger with current window
  };

  // Determine regime status for the header
  let regime = 'NORMAL';
  let pillClass = '';
  if (data && data.length > 0) {
    const lastZ = data[data.length - 1].GK_Zscore;
    if (lastZ >= 2.5) {
      regime = 'EXPANSION';
      pillClass = 'amber';
    } else if (lastZ <= -1.0) {
      regime = 'COMPRESSION';
      pillClass = 'cyan';
    }
  }

  const selectedInfo = tickers.find(t => t.ticker === ticker);
  const displayTicker = selectedInfo ? `${selectedInfo.ticker} — ${selectedInfo.name}` : ticker;

  return (
    <div className="app-grid">
      <header className="header-bar">
        <div className="header-left">
           <span className="header-title">GKHV</span>
           <span className="header-subtitle">Market Terminal Intelligence</span>
        </div>
        <div className="header-pills">
           <div className="pill">{displayTicker}</div>
           <div className={`pill ${pillClass}`}>{regime}</div>
           <div className="pill dashed">Market Status</div>
           <div className="pill dashed">+ Add subheading</div>
        </div>
      </header>
      
      <aside className="sidebar">
        <h2 className="sidebar-title">Summary</h2>
        <div className="sidebar-card">SIGNAL</div>
        <div className="sidebar-card">BREADTH</div>
        <div className="sidebar-card">VOLATILITY</div>
        <div className="sidebar-card">REGIME</div>
        <div className="sidebar-card">RISK</div>
        <div className="sidebar-card">TREND</div>
      </aside>
      
      <main className="main-content">
        <div className="top-bar">
           <div style={{ width: '250px' }}>
             <TickerDropdown 
               tickers={tickers} 
               selectedTicker={ticker} 
               onSelect={handleTickerSelect} 
             />
           </div>
           <div className="window-input-group">
             <label htmlFor="window">Window</label>
             <input 
               type="number" 
               id="window" 
               value={windowSize} 
               onChange={(e) => setWindowSize(Number(e.target.value))} 
               min="10" 
               max="200" 
               onKeyDown={(e) => {
                 if (e.key === 'Enter') {
                   fetchData(ticker, windowSize);
                 }
               }}
             />
           </div>
           {loading && <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Processing...</span>}
        </div>
        
        {error && <div className="error-banner">{error}</div>}
        
        <div className="chart-wrapper">
          {data && data.length > 0 && <ChartComponent data={data} />}
        </div>
      </main>
    </div>
  )
}

export default App
