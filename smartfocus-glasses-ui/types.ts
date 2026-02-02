export enum AppPhase {
  WALKING = 'WALKING',
  SETUP = 'SETUP',
  FOCUS = 'FOCUS',
  SUMMARY = 'SUMMARY'
}

export interface StudyRoom {
  id: string;
  name: string;
  occupancy: number; // 0-100
  distance: string;
  quietLevel: 'High' | 'Moderate' | 'Low';
}

export interface Task {
  id: string;
  title: string;
  completed: boolean;
}

export interface SystemStatus {
  battery: number;
  wifi: boolean;
  dnd: boolean; // Do Not Disturb
  time: string;
}
