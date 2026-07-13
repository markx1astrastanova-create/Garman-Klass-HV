import { useState, useEffect } from 'react'
import ChartComponent from './components/Chart'
import TickerDropdown, { TickerInfo } from './components/TickerDropdown'

function App() {
  const [ticker, setTicker] = useState('JKSE') // default ticker updated
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

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${baseUrl}/api/volatility?ticker=${ticker}&window=${windowSize}`)
      
      if (!response.ok) {
        // Coba baca error jika berformat JSON, jika tidak ambil teks mentahnya
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

  return (
    <>
      <h1>GKHV Quant Engine</h1>
      
      <div className="controls">
        <div style={{ flex: 1 }}>
          <label style={{ display: 'block', marginBottom: '8px' }}>Ticker: </label>
          <TickerDropdown 
            tickers={tickers} 
            selectedTicker={ticker} 
            onSelect={setTicker} 
          />
        </div>
        
        <div>
          <label htmlFor="window">Window (Z-Score): </label>
          <input 
            id="window"
            type="number" 
            value={windowSize} 
            onChange={(e) => setWindowSize(Number(e.target.value))} 
            min="10"
            max="100"
          />
        </div>

        <button onClick={fetchData} disabled={loading}>
          {loading ? 'Fetching...' : 'Analyze'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      {data && data.length > 0 && (
        <div className="chart-container">
          <ChartComponent data={data} />
        </div>
      )}
    </>
  )
}

export default App
