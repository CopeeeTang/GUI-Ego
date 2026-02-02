import React, { useState, useEffect } from 'react';
import { Icons } from './Icon';

const TOTAL_TIME = 25 * 60; // 25 minutes in seconds

export const PhaseFocus: React.FC = () => {
  const [timeLeft, setTimeLeft] = useState(TOTAL_TIME);
  const [isActive, setIsActive] = useState(true);
  const [progress, setProgress] = useState(100);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isActive && timeLeft > 0) {
      interval = setInterval(() => {
        setTimeLeft((prev) => prev - 1);
        setProgress(((timeLeft - 1) / TOTAL_TIME) * 100);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isActive, timeLeft]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="w-full h-full relative">
      
      {/* Top Right: Peripheral Timer (Unobtrusive) */}
      <div className="absolute top-0 right-0 p-6 flex flex-col items-end pointer-events-auto">
        <div className="relative w-32 h-32 flex items-center justify-center">
           {/* SVG Circle Progress */}
           <svg className="w-full h-full transform -rotate-90">
             <circle
               cx="64"
               cy="64"
               r="56"
               stroke="currentColor"
               strokeWidth="4"
               fill="transparent"
               className="text-gray-800/50"
             />
             <circle
               cx="64"
               cy="64"
               r="56"
               stroke="currentColor"
               strokeWidth="4"
               fill="transparent"
               strokeDasharray={351} // 2 * PI * 56
               strokeDashoffset={351 - (351 * progress) / 100}
               className={`text-green-500 transition-all duration-1000 ease-linear ${isActive ? 'opacity-100' : 'opacity-50'}`}
             />
           </svg>
           <div className="absolute text-center">
             <div className="text-3xl font-bold text-white font-mono">{formatTime(timeLeft)}</div>
             <div className="text-xs text-green-400 uppercase tracking-widest">Focus</div>
           </div>
        </div>
        
        {/* Play/Pause Control (Hover to reveal) */}
        <div className="mt-2 opacity-30 hover:opacity-100 transition-opacity flex space-x-2">
           <button 
             onClick={() => setIsActive(!isActive)}
             className="p-2 bg-gray-800/80 rounded-full text-white hover:bg-green-600 transition-colors"
           >
             {isActive ? <div className="w-4 h-4 border-l-2 border-r-2 border-white ml-0.5" /> : <Icons.Navigation className="w-4 h-4 rotate-90" />}
           </button>
        </div>
      </div>

      {/* Top Left: Current Task (Persistent Reminder) */}
      <div className="absolute top-24 left-0 w-64">
        <div className="bg-black/40 backdrop-blur-sm border-l-4 border-green-500 pl-4 py-2">
          <p className="text-xs text-green-400 uppercase tracking-wider mb-1">Current Goal</p>
          <p className="text-white text-lg font-medium leading-tight shadow-black drop-shadow-md">Finish Thesis Draft</p>
        </div>
      </div>

      {/* Bottom Right: Next Appointment (Context Awareness) */}
      <div className="absolute bottom-12 right-0 flex items-center space-x-3 bg-black/60 backdrop-blur-md px-6 py-3 rounded-l-full border border-gray-700/50">
         <div className="text-right">
           <p className="text-xs text-gray-400 uppercase">Up Next (16:00)</p>
           <p className="text-white font-bold">Advisor Meeting</p>
         </div>
         <div className="h-8 w-px bg-gray-600"></div>
         <Icons.Calendar className="w-6 h-6 text-yellow-500" />
      </div>

      {/* Bottom Left: Audio Visualizer (Subtle) */}
      <div className="absolute bottom-12 left-0 flex items-center space-x-4">
        <div className="flex items-end space-x-1 h-8">
          {[1,2,3,4,5].map((i) => (
             <div 
               key={i} 
               className="w-1 bg-green-500/80 animate-pulse"
               style={{ 
                 height: `${Math.random() * 100}%`,
                 animationDuration: `${0.5 + Math.random()}s`
               }} 
             />
          ))}
        </div>
        <div>
           <p className="text-xs text-gray-400 uppercase">Now Playing</p>
           <p className="text-sm text-green-300 font-medium">Focus Piano No. 4</p>
        </div>
      </div>

    </div>
  );
};