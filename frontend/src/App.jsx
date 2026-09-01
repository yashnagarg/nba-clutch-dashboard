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
const [selectedId, setSelectedId]=useState(null);
const [selectedName, setSelectedName]=useState('');
const [playerStats, setPlayerStats]=useState(null);

useEffect(()=>{ 
    api.getLeaderboard(20,50).then(data=> setLeaderboard(data.top_players)).catch(err=>console.error(err))
},[])
  return (
    <>
      <div
        style={{
          background: 'var(--bg)',
          height: '130vh',
          display: 'flex',
        }}
      >
        <div
          style={{
            width: '400px',
            background: 'var(--bg-card)',
            borderRight: '4px solid var(--bg-card2)',
          }}
        >
          <div style={{ padding: '17px 18px 12px' }}>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '900',
                color: 'var(--orange)',
                letterSpacing: '0.13em',
                marginTop:'4px'
              }}
            >
              FIRE SCORE
            </div>

            <div
              style={{
                fontSize: '13px',
                color: 'var(--text-muted)',
                marginTop: '2px',
              }}
            >
                2024-25 season
                        </div>
          </div>
          <div style={{
            display:"flex",alignItems:"center",gap:"8px",padding:"6px 16px",borderTop:"1px solid var(--border)"
          }}>
            <span style={{fontSize:"9px",color:"var(--text-muted)",letterSpacing:"0.1em",textTransform:"uppercase",width:"18px"}}>#</span>
            <span style={{ fontSize:"9px",color:"var(--text-muted)",letterSpacing:"0.1em",textTransform:"uppercase",flex:1}}>Player</span>
            <span style={{ fontSize:"9px",color:"var(--text-muted)",letterSpacing:"0.1em",textTransform:"uppercase",width:"40px",textAlign:"right"}}>FS</span>
            <span style={{ fontSize:"9px",color:"var(--text-muted)",letterSpacing:"0.1em",textTransform:"uppercase",width:"36px",textAlign:"right"}}>CL%</span>


          </div>
          <div style={{flex:1, overflowY:"auto", padding:'8px'}}>
            {leaderboard.map((p,i)=>(
                <div key={p.playerName} 
                onClick={()=> {setSelectedId(p.personId); setSelectedName(p.playerName);}}
                style={{
                    display:'flex',alignItems:'center',gap:'8px', padding:'10px 12px',borderRadius:'6px',cursor:'pointer',marginBottom:'2px',borderLeft:"3px solid transparent",transition:"all 0.15s"
                }}
                onMouseEnter={e=>{
                    e.currentTarget.style.background='var(--bg)'
                    e.currentTarget.style.borderLeftColor='var(--orange)'}}
                onMouseLeave={e=>{
                    e.currentTarget.style.background='transparent'
                    e.currentTarget.style.borderLeftColor='transparent'
                }}>
                 {/*rank on the left side*/}
                    <span style={{fontSize:'11px',padding:"10px",color:"var(--orange",width:'16px',fontWeight:"700"}}>
                        {i+1}
                    </span>
                {/*player headshot*/}
                <div style={{width:"32px",height:"32px",borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",overflow:"hidden",background:"var(--bg-card2)",border:"1px solid var(--border)"}}>
                    <img src={`https://cdn.nba.com/headshots/nba/latest/1040x760/${p.personId}.png`}
                    alt={p.playerName}
                    style={{
                        width:'100%',width:"100%",objectFit:"cover"
                    }}
                    onError={(e)=>{
                        e.currentTarget.style.display='none'
                    }}/>
                    </div>
                <div style={{flex:1,minWidth:0}}>
                    <div style={{fontSize:'13px',fontWeight:"450",whiteSpace:"nowrap",overflow:"hidden",color:'var(--text)'}}>{p.playerName}</div>
                    {/*progress bar*/}
                    <div style={{height:'2px',background:'var(--border)',borderRadius:'3px',marginTop:'5px',overflow:'hidden'}}>
                        <motion.div initial={{width:0}}
                        animate={{width:`${Math.round(p.fire_score*100)}%`}} 
                        transition={{duration:0.7,ease:'easeInOut'}} 
                        style={{height:'100%',background:'var(--orange)',borderRadius:'3px'}}/>
                        </div>
                </div>
                {/*score on the right*/}
                <span style={{ fontSize:'10px',color:'var(--text-muted)',width:"30px"}}>
                    {p.fire_score.toFixed(3)}
                </span>
                <span style={{ fontSize:'10px',color:"var(--text-muted)",width:"30px",textAlign:"right"}}>
                    {p.clutch_fg_pct.toFixed(2)}
                </span>
                </div>
            ))}
          </div>
        </div>

        <main style={{ flex: 1, padding: '45px 28px' , display:"flex",flexDirection:'column'}}>
            { !selectedId ? (
              //big heading
                <div style={{
                    paddingLeft:"46px",flex:1,display:"flex",flexDirection:"column",alignItems:"left",justifyContent:"left",gap:"16px",overflow:"hidden"
                }}>
                <div style={{ paddingLeft:"6px",fontSize:"14px",color:"var(--orange)",letterSpacing:"0.2em",textTransform:"uppercase",fontWeight:"600",marginTop:"10px"}}>
                    Exploring the 2024-25 NBA Season
                </div>
                <div style={{position:"relative",display:"flex",alignItems:"left",justifyContent:"left",overflow:"visible"}}>
                <motion.div  animate={{ scale: [1, 1.09, 1.04] }}  transition={{duration: 4,repeat: Infinity, ease: "easeInOut"}} style={{position:"absolute",width:"1200px",height:"800px",background:"radial-gradient(ellipse, rgba(254,127,45,0.15) 0%, transparent 65%)",pointerEvents:"none"}}/>
                <motion.h1 initial={{opacity:0,y:60,scale:0.96}}
                animate={{opacity:1,y:0,scale:1.12}} 
                transition={{duration:0.7,ease:"easeInOut"}}
                style={{
                    fontFamily:'var(--display-font)',fontSize:"135px",fontWeight:"900",color:"var(--text)",textAlign:"left",lineHeight:"1",letterSpacing:"-0.02em",textTransform:"uppercase",paddingLeft:"35px"
                }}>
                    WHO GETS {' '}
                    <span style={{color:"var(--orange)"}}> HOT</span>
                    <br />WHEN IT 
                    <br />MATTERS?
                </motion.h1> 
                
                </div>

                {/* Divider */}
                <motion.div
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 64, opacity: 1 }}
                transition={{ duration: 0.6, delay: 0.7, ease: "easeOut" }}
                style={{
                    height: 2,
                    background: "var(--orange)",
                    borderRadius: 1,
                    marginTop: 28,
                    marginBottom: 24,
                }}
                />
                <motion.p initial={{opacity:0,y:18}}
                animate={{opacity:1,y:0}}
                transition={{delay:"0.80",duration:"0.55",ease:"easeInOut"}} style={{fontSize:"17px",color:"var(--text-muted)",textAlign:"left",maxWidth:"480px",lineHeight:"1.4",paddingLeft:"6px",paddingBottom:"6px"}}>
                    Quantifying how NBA players perform when the pressure rises.
                </motion.p>
                <div style={{
                    marginTop:"10px",fontSize:"14px",color:"var(--text-sub)",display:"flex",alignItems:"left",gap:"8px"
                }}>
                    <span style={{color:"var(--orange)"}}>←←←</span>
                    Select a player from the leaderboard to begin
                </div>
            </div>
            ):(
                
                <div style={{color:"var(--orange)"}}>
                    <h2 style={{ fontFamily:"var(--display-font)",fontSize:"32px",fontWeight:"800",marginBottom:"8px"}}>{selectedName}</h2>
                    <p style={{ color:"var(--text-muted)",fontsize:"14px",marginBottom:"16px"}}>Dashboard coming next!!</p>
                </div>
            )
            }
            
        </main>
      </div>
    </>
  );
}