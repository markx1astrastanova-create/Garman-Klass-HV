import { useEffect, useRef } from 'react'
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts'

interface DataPoint {
  date: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  GK_Vol: number;
  GK_Zscore: number;
}

interface ChartProps {
  data: DataPoint[];
}

const ChartComponent = ({ data }: ChartProps) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!chartContainerRef.current) return

    const handleResize = () => {
      chart.applyOptions({ width: chartContainerRef.current?.clientWidth, height: chartContainerRef.current?.clientHeight })
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#000000' },
        textColor: '#888888',
      },
      grid: {
        vertLines: { color: '#1a1a1a' },
        horzLines: { color: '#1a1a1a' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#333333', style: 2 },
        horzLine: { color: '#333333', style: 2 },
      },
      rightPriceScale: {
        borderColor: '#1a1a1a',
        scaleMargins: {
          top: 0.05,
          bottom: 0.35, // Main chart takes top 65%
        },
      },
      timeScale: {
        borderColor: '#1a1a1a',
      },
    })

    chart.timeScale().fitContent()

    // Candlestick Series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00ff88',
      downColor: '#ff4444',
      borderVisible: false,
      wickUpColor: '#00ff88',
      wickDownColor: '#ff4444',
    })

    const candleData = data.map((d) => ({
      time: d.date,
      open: d.Open,
      high: d.High,
      low: d.Low,
      close: d.Close,
    }))
    candleSeries.setData(candleData)

    // Z-Score Series (Baseline for different area fill colors)
    const zscoreSeries = chart.addBaselineSeries({
      baseValue: { type: 'price', price: 0 },
      topLineColor: '#ff8c00',
      topFillColor1: 'rgba(255, 140, 0, 0.15)',
      topFillColor2: 'rgba(255, 140, 0, 0.05)',
      bottomLineColor: '#ff8c00',
      bottomFillColor1: 'rgba(0, 191, 255, 0.05)',
      bottomFillColor2: 'rgba(0, 191, 255, 0.15)',
      lineWidth: 2,
      priceScaleId: 'zscore',
    })

    chart.priceScale('zscore').applyOptions({
      scaleMargins: {
        top: 0.7, // Vol chart takes bottom 30%
        bottom: 0,
      },
    })

    const zscoreData = data.map((d) => ({
      time: d.date,
      value: d.GK_Zscore,
    }))
    zscoreSeries.setData(zscoreData)

    // Baseline threshold for Z-Score
    zscoreSeries.createPriceLine({
      price: 2.5,
      color: '#ff8c00',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'EXPANSION 2.5',
    });
    
    zscoreSeries.createPriceLine({
      price: -1.0,
      color: '#00bfff',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'COMPRESSION -1.0',
    });

    zscoreSeries.createPriceLine({
      price: 0,
      color: '#333333',
      lineWidth: 1,
      lineStyle: 0,
      axisLabelVisible: false,
    });

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [data])

  return (
    <div
      ref={chartContainerRef}
      style={{ width: '100%', height: '100%' }}
    />
  )
}

export default ChartComponent
