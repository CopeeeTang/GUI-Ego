import React, { useState, useEffect } from 'react';
import { GlassHUD } from './components/GlassHUD';
import { PhaseWalking } from './components/PhaseWalking';
import { PhaseSetup } from './components/PhaseSetup';
import { PhaseFocus } from './components/PhaseFocus';
import { AppPhase, SystemStatus } from './types';

// Mock backgrounds
const BG_WALKING = "https://picsum.photos/id/122/1920/1080"; // City/Campus feel
const BG_STUDY = "https://picsum.photos/id/20/1920/1080"; // Desk items

export default function App() {
  const [phase, setPhase] = useState<AppPhase>(AppPhase.WALKING);
  const [status, setStatus] = useState<SystemStatus>({
    battery: 85,
    wifi: true,
    dnd: false,
    time: '14:23',
  });

  // Time simulation
  useEffect(() => {
    const timer = setInterval(() => {
      // Very simple time ticker for demo
      setStatus(prev => {
        const [h, m] = prev.time.split(':').map(Number);
        let newM = m + 1;
        let newH = h;
        if (newM >= 60) {
          newM = 0;
          newH += 1;
        }
        return {
          ...prev,
          time: `${newH}:${newM.toString().padStart(2, '0')}`
        };
      });
    }, 30000); // Update time every 30 real seconds
    return () => clearInterval(timer);
  }, []);

  // Handle phase transitions specific actions
  useEffect(() => {
    if (phase === AppPhase.FOCUS) {
      setStatus(prev => ({ ...prev, dnd: true }));
    } else {
      setStatus(prev => ({ ...prev, dnd: false }));
    }
  }, [phase]);

  const renderContent = () => {
    switch (phase) {
      case AppPhase.WALKING:
        return <PhaseWalking onArrive={() => setPhase(AppPhase.SETUP)} />;
      case AppPhase.SETUP:
        return <PhaseSetup onComplete={() => setPhase(AppPhase.FOCUS)} />;
      case AppPhase.FOCUS:
        return <PhaseFocus />;
      default:
        return null;
    }
  };

  const currentBg = phase === AppPhase.WALKING ? BG_WALKING : BG_STUDY;

  return (
    <GlassHUD status={status} backgroundUrl={currentBg}>
      {renderContent()}
    </GlassHUD>
  );
}