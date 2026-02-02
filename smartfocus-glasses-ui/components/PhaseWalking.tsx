import React, { useState, useEffect } from 'react';
import { Icons } from './Icon';
import { StudyRoom } from '../types';

interface PhaseWalkingProps {
  onArrive: () => void;
}

const MOCK_ROOMS: StudyRoom[] = [
  { id: '1', name: 'Library 301', occupancy: 85, distance: '120m', quietLevel: 'Moderate' },
  { id: '2', name: 'Quiet Zone B', occupancy: 15, distance: '50m', quietLevel: 'High' },
  { id: '3', name: 'Commons A', occupancy: 92, distance: '200m', quietLevel: 'Low' },
];

export const PhaseWalking: React.FC<PhaseWalkingProps> = ({ onArrive }) => {
  const [distance, setDistance] = useState(480); // meters
  const [isNear, setIsNear] = useState(false);

  // Simulate walking closer
  useEffect(() => {
    const timer = setInterval(() => {
      setDistance((prev) => {
        const newDist = Math.max(0, prev - 2);
        if (newDist < 20) setIsNear(true);
        return newDist;
      });
    }, 100);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="w-full h-full flex flex-col justify-between">
      
      {/* Center: Navigation Arrow */}
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
        <div className="relative">
          <div className="absolute inset-0 bg-green-500/20 blur-xl rounded-full animate-pulse" />
          <Icons.Navigation className="w-24 h-24 text-green-400 drop-shadow-[0_0_10px_rgba(74,222,128,0.8)]" />
        </div>
        <div className="mt-4 text-center">
          <h2 className="text-4xl font-bold text-white drop-shadow-md">{distance}m</h2>
          <p className="text-green-300 tracking-wider text-sm uppercase">Destination: Quiet Zone B</p>
        </div>
      </div>

      {/* Right Side: Context Data (Occupancy) */}
      <div className="absolute right-0 top-1/4 w-80 bg-black/60 backdrop-blur-md rounded-l-2xl border-l border-y border-green-500/30 p-6 transform transition-all hover:scale-105 origin-right">
        <div className="flex items-center space-x-2 mb-4 border-b border-green-500/30 pb-2">
          <Icons.Users className="w-5 h-5 text-green-400" />
          <h3 className="text-lg font-bold text-white">Study Room Status</h3>
        </div>
        
        <div className="space-y-4">
          {MOCK_ROOMS.map((room) => {
            const isRecommended = room.name === 'Quiet Zone B';
            return (
              <div key={room.id} className={`relative ${isRecommended ? 'opacity-100' : 'opacity-60'}`}>
                <div className="flex justify-between text-sm mb-1">
                  <span className={`font-semibold ${isRecommended ? 'text-green-300' : 'text-gray-400'}`}>
                    {room.name} {isRecommended && '(Recommended)'}
                  </span>
                  <span className="text-white">{room.occupancy}% Full</span>
                </div>
                <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${
                      room.occupancy > 80 ? 'bg-red-500' : 
                      room.occupancy > 50 ? 'bg-yellow-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${room.occupancy}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>{room.distance}</span>
                  <span>{room.quietLevel} Noise</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom Action Hint */}
      {isNear && (
        <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 animate-bounce">
          <button 
            onClick={onArrive}
            className="group flex items-center space-x-2 bg-green-500/20 hover:bg-green-500/40 backdrop-blur-md border border-green-500 text-white px-8 py-3 rounded-full transition-all"
          >
            <span>Arrive at Location</span>
            <Icons.ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      )}
    </div>
  );
};