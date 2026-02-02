import React, { useState, useEffect } from 'react';
import { Icons } from './Icon';

interface PhaseSetupProps {
  onComplete: () => void;
}

const CHECKLIST_ITEMS = [
  { id: '1', text: 'Analyzing environment noise...', icon: Icons.Volume2, duration: 1000 },
  { id: '2', text: 'Connecting to "Focus Piano" playlist...', icon: Icons.Music, duration: 2000 },
  { id: '3', text: 'Silencing notifications (DND Mode)...', icon: Icons.BellOff, duration: 3000 },
  { id: '4', text: 'Setting goal: "Thesis Draft"...', icon: Icons.Target, duration: 4000 },
  { id: '5', text: 'Initializing Pomodoro Timer (25m)...', icon: Icons.Clock, duration: 4500 },
];

export const PhaseSetup: React.FC<PhaseSetupProps> = ({ onComplete }) => {
  const [completedItems, setCompletedItems] = useState<string[]>([]);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    CHECKLIST_ITEMS.forEach((item) => {
      setTimeout(() => {
        setCompletedItems((prev) => [...prev, item.id]);
      }, item.duration);
    });

    setTimeout(() => {
      setReady(true);
    }, 5000);
  }, []);

  return (
    <div className="w-full h-full flex items-center justify-center">
      <div className="w-full max-w-md bg-black/80 backdrop-blur-xl border border-green-500/50 rounded-3xl p-8 shadow-[0_0_50px_rgba(34,197,94,0.2)]">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center space-x-3 border-b border-green-500/30 pb-4">
          <Icons.BrainCircuit className="w-8 h-8 text-green-400" />
          <span>Deep Work Setup</span>
        </h2>

        <div className="space-y-6">
          {CHECKLIST_ITEMS.map((item) => {
            const isCompleted = completedItems.includes(item.id);
            const Icon = item.icon;
            
            return (
              <div 
                key={item.id} 
                className={`flex items-center space-x-4 transition-all duration-500 ${
                  isCompleted ? 'opacity-100 translate-x-0' : 'opacity-30 -translate-x-4'
                }`}
              >
                <div className={`p-2 rounded-full ${isCompleted ? 'bg-green-500/20 text-green-400' : 'bg-gray-800 text-gray-500'}`}>
                  {isCompleted ? <Icons.CheckCircle2 className="w-5 h-5" /> : <Icon className="w-5 h-5" />}
                </div>
                <span className={`text-lg ${isCompleted ? 'text-white' : 'text-gray-400'}`}>
                  {item.text}
                </span>
              </div>
            );
          })}
        </div>

        <div className={`mt-8 transition-all duration-700 ${ready ? 'opacity-100 transform translate-y-0' : 'opacity-0 transform translate-y-4'}`}>
          <button 
            onClick={onComplete}
            className="w-full bg-green-600 hover:bg-green-500 text-black font-bold text-xl py-4 rounded-xl shadow-[0_0_20px_rgba(34,197,94,0.6)] transition-all hover:scale-[1.02]"
          >
            ENTER FOCUS MODE
          </button>
        </div>
      </div>
    </div>
  );
};