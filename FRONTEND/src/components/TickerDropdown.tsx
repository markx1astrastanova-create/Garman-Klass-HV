import React, { useState, useMemo, useRef, useEffect } from 'react';

export interface TickerInfo {
  ticker: string;
  name: string;
  type: string;
  region: string;
  exchange: string;
  aliases?: string;
  keywords?: string;
}

interface TickerDropdownProps {
  tickers: TickerInfo[];
  selectedTicker: string;
  onSelect: (ticker: string) => void;
}

const TickerDropdown: React.FC<TickerDropdownProps> = ({ tickers, selectedTicker, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeRegion, setActiveRegion] = useState('Semua');
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Determine selected name
  const selectedInfo = tickers.find((t) => t.ticker === selectedTicker);
  const displayLabel = selectedInfo ? `${selectedInfo.ticker} — ${selectedInfo.name}` : selectedTicker;

  // Regions list
  const regions = ['Semua', 'ASEAN', 'Asia', 'America', 'Europe', 'Global'];

  // Handle click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Filter and sort tickers
  const filteredTickers = useMemo(() => {
    let result = tickers;

    // Filter by Region
    if (activeRegion !== 'Semua') {
      // Commodities are technically 'Global', check region or type based on CSV mapping
      result = result.filter(
        (t) => t.region.toLowerCase() === activeRegion.toLowerCase()
      );
    }

    // Filter by Search
    const query = searchQuery.trim().toLowerCase();
    if (query) {
      const words = query.split(/\s+/);
      result = result.filter((t) => {
        const searchTarget = `${t.ticker} ${t.name} ${t.aliases || ''} ${t.keywords || ''}`.toLowerCase();
        return words.every((word) => searchTarget.includes(word));
      });

      // Sort: exact ticker match first, then starts with, then includes
      result.sort((a, b) => {
        const aTicker = a.ticker.toLowerCase();
        const bTicker = b.ticker.toLowerCase();
        if (aTicker === query) return -1;
        if (bTicker === query) return 1;
        if (aTicker.startsWith(query) && !bTicker.startsWith(query)) return -1;
        if (bTicker.startsWith(query) && !aTicker.startsWith(query)) return 1;
        return 0;
      });
    }

    return result;
  }, [tickers, activeRegion, searchQuery]);

  const getRegionColor = (region: string) => {
    switch (region.toLowerCase()) {
      case 'asean': return '#e74c3c';
      case 'asia': return '#f39c12';
      case 'america': return '#3498db';
      case 'europe': return '#9b59b6';
      case 'global': return '#f1c40f';
      default: return '#555';
    }
  };

  return (
    <div className="ticker-dropdown-container" ref={dropdownRef} style={{ position: 'relative', display: 'inline-block', minWidth: '250px' }}>
      
      {/* Trigger Button */}
      <button 
        className="dropdown-trigger" 
        onClick={() => setIsOpen(!isOpen)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          width: '100%',
          padding: '10px 14px',
          background: '#1e1e1e',
          border: '1px solid #333',
          borderRadius: '8px',
          color: '#fff',
          cursor: 'pointer',
          textAlign: 'left'
        }}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {displayLabel}
        </span>
        <svg 
          width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          style={{ transition: 'transform 0.2s', transform: isOpen ? 'rotate(180deg)' : 'rotate(0)' }}
        >
          <path d="M6 9l6 6 6-6"/>
        </svg>
      </button>

      {/* Dropdown Panel */}
      {isOpen && (
        <div 
          className="dropdown-panel"
          style={{
            position: 'absolute',
            top: 'calc(100% + 8px)',
            left: 0,
            width: '320px',
            maxHeight: '400px',
            background: '#1e1e1e',
            border: '1px solid #333',
            borderRadius: '8px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.5)',
            zIndex: 9999,
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
          }}
        >
          {/* Search Bar */}
          <div style={{ padding: '12px', borderBottom: '1px solid #333', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#888" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <input 
              type="text" 
              placeholder="Cari indeks atau negara..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: '100%',
                background: '#121212',
                border: '1px solid #333',
                padding: '8px 12px',
                color: '#fff',
                borderRadius: '4px',
                outline: 'none'
              }}
              autoFocus
            />
          </div>

          {/* Region Filters */}
          <div 
            className="region-filters"
            style={{ 
              padding: '8px 12px', 
              display: 'flex', 
              gap: '6px', 
              overflowX: 'auto',
              borderBottom: '1px solid #333',
              whiteSpace: 'nowrap'
            }}
          >
            {regions.map((region) => (
              <button
                key={region}
                onClick={() => setActiveRegion(region)}
                style={{
                  padding: '4px 10px',
                  borderRadius: '16px',
                  fontSize: '12px',
                  cursor: 'pointer',
                  border: 'none',
                  background: activeRegion === region ? '#4a90d9' : '#2a2a2a',
                  color: activeRegion === region ? '#fff' : '#888',
                  transition: 'all 0.2s'
                }}
              >
                {region}
              </button>
            ))}
          </div>

          {/* Ticker List */}
          <div 
            role="listbox"
            style={{
              overflowY: 'auto',
              flex: 1,
              padding: '6px 0',
              maxHeight: '280px' // adjust for scroll
            }}
          >
            {filteredTickers.length > 0 ? (
              filteredTickers.map((t) => (
                <div 
                  key={t.ticker}
                  role="option"
                  aria-selected={t.ticker === selectedTicker}
                  onClick={() => {
                    onSelect(t.ticker);
                    setIsOpen(false);
                    setSearchQuery('');
                  }}
                  style={{
                    padding: '10px 16px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    borderLeft: t.ticker === selectedTicker ? '3px solid #4a90d9' : '3px solid transparent',
                    background: t.ticker === selectedTicker ? 'rgba(74, 144, 217, 0.1)' : 'transparent',
                    transition: 'background 0.1s'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = '#2a2a2a'}
                  onMouseLeave={(e) => e.currentTarget.style.background = t.ticker === selectedTicker ? 'rgba(74, 144, 217, 0.1)' : 'transparent'}
                >
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <span style={{ fontWeight: 'bold', fontFamily: 'monospace', fontSize: '14px', color: '#fff' }}>{t.ticker}</span>
                    <span style={{ fontSize: '12px', color: '#888' }}>{t.name}</span>
                  </div>
                  <span style={{
                    fontSize: '10px',
                    padding: '2px 8px',
                    borderRadius: '12px',
                    background: getRegionColor(t.region),
                    color: '#fff',
                    fontWeight: 'bold'
                  }}>
                    {t.region}
                  </span>
                </div>
              ))
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: '#888', fontSize: '14px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                 <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                   <circle cx="11" cy="11" r="8"></circle>
                   <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                 </svg>
                 Tidak ditemukan
              </div>
            )}
          </div>
        </div>
      )}

      <style>{`
        .region-filters::-webkit-scrollbar {
          height: 0px; /* Hide scrollbar for chips */
        }
        div[role="listbox"]::-webkit-scrollbar {
          width: 6px;
        }
        div[role="listbox"]::-webkit-scrollbar-track {
          background: transparent;
        }
        div[role="listbox"]::-webkit-scrollbar-thumb {
          background: #444;
          border-radius: 4px;
        }
      `}</style>
    </div>
  );
};

export default TickerDropdown;
