import React, { ReactNode } from 'react';
import { Icons } from './Icon';
import { SystemStatus } from '../types';

interface GlassHUDProps {
  children: ReactNode;
  status: SystemStatus;
  backgroundUrl: string;
}

export const GlassHUD: React.FC<GlassHUDProps> = ({ children, status, backgroundUrl }) => {
  return (
    <div className="relative w-screen h-screen overflow-hidden text-green-400 select-none">
      {/* Background Image simulating First-Person View */}
      <div 
        className="absolute inset-0 bg-cover bg-center transition-all duration-1000 ease-in-out"
        style={{ 
          backgroundImage: `url(${backgroundUrl})`,
          filter: 'brightness(0.6) contrast(1.1)' 
        }}
      />
      
      {/* Glass Reflection/Vignette Overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-transparent to-black/60 pointer-events-none" />
      <div className="absolute inset-0 shadow-[inset_0_0_100px_rgba(0,0,0,0.9)] pointer-events-none" />

      {/* Top Bar (System Status) */}
      <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-start z-10">
        <div className="flex items-center space-x-4 bg-black/40 backdrop-blur-sm px-4 py-2 rounded-full border border-green-500/20">
          <Icons.Clock className="w-4 h-4 text-green-300" />
          <span className="text-xl font-bold tracking-widest text-white">{status.time}</span>
        </div>

        <div className="flex items-center space-x-3 bg-black/40 backdrop-blur-sm px-4 py-2 rounded-full border border-green-500/20">
          {status.dnd && <Icons.BellOff className="w-4 h-4 text-yellow-400 animate-pulse" />}
          <Icons.Wifi className="w-4 h-4 text-green-400" />
          <div className="flex items-center space-x-1">
            <span className="text-sm font-bold text-green-400">{status.battery}%</span>
            <Icons.Battery className="w-4 h-4 text-green-400" />
          </div>
        </div>
      </div>

      {/* Main Content Area (Safe Zone) */}
      <div className="absolute inset-0 z-0 flex flex-col justify-center items-center pointer-events-none">
        {/* We use pointer-events-auto on children that need interaction */}
        <div className="w-full h-full p-12 relative pointer-events-auto">
          {children}
        </div>
      </div>

      {/* HUD Frame Decorations (Cyberpunk/Sci-fi corners) */}
      <div className="absolute top-8 left-8 w-32 h-32 border-l-2 border-t-2 border-green-500/30 rounded-tl-3xl pointer-events-none" />
      <div className="absolute top-8 right-8 w-32 h-32 border-r-2 border-t-2 border-green-500/30 rounded-tr-3xl pointer-events-none" />
      <div className="absolute bottom-8 left-8 w-32 h-32 border-l-2 border-b-2 border-green-500/30 rounded-bl-3xl pointer-events-none" />
      <div className="absolute bottom-8 right-8 w-32 h-32 border-r-2 border-b-2 border-green-500/30 rounded-br-3xl pointer-events-none" />
    </div>
  );
};