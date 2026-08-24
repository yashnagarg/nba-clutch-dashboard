import { useState, useEffect } from "react";
import { api } from '../api';
import { motion } from 'framer-motion'
/*
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
    
const [leaderboard, setLeaderboard]=useState([])

useEffect(()=>{ 
    api.getLeaderboard(20,50).then(data=> setLeaderboard(data.top_players)).catch(err=>console.error(err))
},[])
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
          <div style={{flex:1, overflowY:"auto", padding:'8px'}}>
            {leaderboard.map((p,i)=>(
                <div key={p.playerName} style={{
                    display:'flex',alignItems:'center',gap:'8px', padding:'8px',borderRadius:'6px',cursor:'pointer',marginBottom:'2px'
                }}>
                    <span style={{fontSize:'12px',color:'var(--text-muted)',width:'18px',textAlign:'right'}}>
                        {i+1}
                    </span>
                <div style={{flex:1}}>
                    <div style={{fontSize:'13px',color:'var(--text)'}}>{p.playerName}</div>
                    {/*progress bar*/}
                    <div style={{height:'27px',background:'var(--border)',borderRadius:'8px',marginTop:'4px',overflow:'hidden'}}>
                        <motion.div initial={{width:0}}
                        animate={{width:`${Math.round(p.fire_score*100)}%`}} 
                        transition={{duration:0.7,ease:'easeInOut'}} 
                        style={{height:'100%',background:'var(--orange)',borderRadius:'8px'}}/>
                        </div>
                </div>
                {/*score on the right*/}
                <span style={{ fontSize:'10px',color:'var(--text-muted)'}}>
                    {p.fire_score.toFixed(3)}
                </span>
                </div>
            ))}
          </div>
        </div>

        <main style={{ flex: 1, padding: '24px 28px' , display:"flex",flexDirection:'column'}}>
            {/*big heading*/}
            <div style={{
                flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:"16px"
            }}>
                <div style={{ fontSize:"13px",color:"var(--orange)",letterSpacing:"0.2em",textTransform:"uppercase",fontWeight:"600"}}>
                    Exploring the 2024-25 NBA Season
                </div>
                <div style={{position:"relative",display:"flex",alignItems:"center",justifyContent:"center",overflow:"visible"}}>
                <motion.div  animate={{ scale: [1, 1.09, 1] }}  transition={{duration: 4,repeat: Infinity, ease: "easeInOut"}} style={{position:"absolute",width:"1200px",height:"800px",background:"radial-gradient(ellipse, rgba(254,127,45,0.15) 0%, transparent 65%)",pointerEvents:"none"}}/>
                <motion.h1 initial={{opacity:0,y:100,scale:0.96}}
                animate={{opacity:1,y:0,scale:1.12}} 
                transition={{duration:1,ease:"easeInOut"}}
                style={{
                    fontSize:"117px",fontWeight:"800",color:"var(--text)",textAlign:"center",lineHeight:"1.1",letterSpacing:"-0.03em"
                }}>
                    WHO GETS HOT<br />WHEN IT MATTERS?
                </motion.h1> 
                
                </div>

                <motion.p initial={{opacity:0,y:20}}
                animate={{opacity:1,y:0}}
                transition={{delay:"0.27",duration:"0.6",ease:"easeInOut"}} style={{fontSize:"17px",color:"var(--text-muted)",textAlign:"center",maxWidth:"480px",lineHeight:"1.6"}}>
                    Quantifying how NBA players perform when the pressure rises.
                </motion.p>
                <div style={{
                    marginTop:"8px",fontSize:"13px",color:"var(--text-sub)",display:"flex",alignItems:"center",gap:"8px"
                }}>
                    <span style={{color:"var(--orange)"}}>←</span>
                    Select a player from the leaderboard to begin
                </div>
            </div>
        </main>
      </div>
    </>
  );
}