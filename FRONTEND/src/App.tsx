import { useState } from 'react'
import ChartComponent from './components/Chart'

function App() {
  const [ticker, setTicker] = useState('BBRI')
  const [windowSize, setWindowSize] = useState(60)
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`http://localhost:8000/api/volatility?ticker=${ticker}&window=${windowSize}`)
      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Gagal fetch data')
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
        <div>
          <label htmlFor="ticker">Ticker: </label>
          <input 
            id="ticker"
            type="text" 
            value={ticker} 
            onChange={(e) => setTicker(e.target.value)} 
            placeholder="Contoh: BBRI atau ^JKSE"
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

      {data && data.data && (
        <div className="chart-container">
          <ChartComponent data={data.data} />
        </div>
      )}
    </>
  )
}

export default App
