import { useState, useEffect } from "react";
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
        

    );
}