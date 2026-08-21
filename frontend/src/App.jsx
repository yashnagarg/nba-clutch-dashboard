/*import { useState, useEffect } from "react";
import { api } from './api.js';
import Sidebar from './components/Sidebar.jsx';
import PlayerHeader from './components/PlayerHeader.jsx';
import StatCards from './components/StatCards.jsx';
import ShotChart from './components/ShotChart.jsx';
import HeatTimeline from './components/HeatTimeline.jsx';
import HotZoneOverlay from './components/HotZoneOverlay.jsx';
import ComparePanel from './components/ComparePanel.jsx';

export default function App()
{
    const [players, setPlayers]=useState([]);
    const [selectedId, setSelectedId]=useState(null);
    const [selectedName, setSelectedName]=useState('');
    const [playerStats, setPlayerStats]=useState(null);
    const [shots, setShots]=useState([]);
    const [streaks, setStreaks]=useState([]);
    const [loading, setLoading]=useState(false);
    const [error, setError]=useState(null);
    const [activeTab, setActiveTab]=useState('overview');

    
    //mount once so all players are loaded into the dropdown search
    useEffect(() => {
        api.getPlayers().then(setPlayers).catch((err) => setError(err));
    }, []);

    //fetch player stats when the selectedId changes
    useEffect(()=>
    {
        if(!selectedId) return;
        setLoading(true);
        setError(null);
        setActiveTab('overview');

        Promise.all([
            api.getPlayers(selectedId),
            api.getShotChart(selectedId),
            api.getStreaks(selectedId),
        ])
        .then(([statsData, shotData, streakData])=>
        {
            setPlayerStats(statsData);
            setShots(shotData);
            setStreaks(streakData);
        })
        .catch(err=>setError(err.message))
        .finally(()=>setLoading(false))
    },[selectedId]);

    //function to handle player selection from the sidebar
    function handlePlayerSelect(playerId,playerName)
    {
        setSelectedId(playerId);
        setSelectedName(playerName);
    }
    
    const tabs=['overview','shot chart', 'compare' ];

    return(
        <>
        <div style={{display: "flex", height: "100vh", overflow:"hidden"}}/>

        <Sidebar
        players={players}
        selectedId={selectedId}
        onSelect={handlePlayerSelect}
        />
        <main style={{flex:1,overflow:"auto",padding: "24 px 28px"}}/>

    
        </>
    );
}*/
export default function App() {
  return (
    <>
      <div
        style={{
          background: 'var(--bg)',
          height: '100vh',
          display: 'flex',
        }}
      >
        <div
          style={{
            width: '260px',
            background: 'var(--bg-card)',
            borderRight: '1px solid var(--border)',
          }}
        >
          <div style={{ padding: '16px' }}>
            <div
              style={{
                fontSize: '15px',
                fontWeight: '700',
                color: 'var(--orange)',
                letterSpacing: '0.08em',
              }}
            >
              FIRE SCORE
            </div>

            <div
              style={{
                fontSize: '10px',
                color: 'var(--text-muted)',
                marginTop: '2px',
              }}
            >
              NBA Clutch + Heat Check | 2024-25
            </div>
          </div>
        </div>

        <main style={{ flex: 1, padding: '24px 28px' }}>
            <h1 style={{fontSize:"18px",
                    fontWeight:"600",
                    color:"var(--text"
            }}> NBA Clutch + Heat Check</h1>
            <p style={{
                fontSize:"12px",
                color:"var(--text-muted)",
                marginTop:"4px"
            }}>2024-25 season | select a player to begin</p>
        </main>
      </div>
    </>
  );
}