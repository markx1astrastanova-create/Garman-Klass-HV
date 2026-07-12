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
      chart.applyOptions({ width: chartContainerRef.current?.clientWidth })
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1e1e1e' },
        textColor: '#d1d4dc',
      },
      grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: '#2B2B43',
      },
      timeScale: {
        borderColor: '#2B2B43',
      },
    })

    chart.timeScale().fitContent()

    // Candlestick Series
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    })

    const candleData = data.map((d) => ({
      time: d.date,
      open: d.Open,
      high: d.High,
      low: d.Low,
      close: d.Close,
    }))
    candleSeries.setData(candleData)

    // Z-Score Series (Line) with distinct scale
    const zscoreSeries = chart.addLineSeries({
      color: '#e74c3c',
      lineWidth: 2,
      priceScaleId: 'zscore',
    })

    chart.priceScale('zscore').applyOptions({
      scaleMargins: {
        top: 0.7,
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
      color: 'orange',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'EXPANSION (2.5)',
    });
    
    zscoreSeries.createPriceLine({
      price: -1.0,
      color: 'green',
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: true,
      title: 'COMPRESSION (-1.0)',
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
