/*
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

import { unsafeCSS, css } from "lit";
import * as Styles from "@a2ui/web_core/styles/index";

const arGlassStyles = css`
  :host {
    font-family: 'Rajdhani', 'Roboto Mono', monospace;
    --ar-green-500: #22c55e;
    --ar-green-400: #4ade80;
    --ar-green-300: #86efac;
    --ar-bg-glass: rgba(0, 0, 0, 0.6);
    --ar-border-dim: rgba(34, 197, 94, 0.3);
    --ar-text-primary: #00ff41;
    --ar-text-secondary: rgba(0, 255, 65, 0.7);
    
    --md-sys-color-primary: var(--ar-green-500);
    --md-sys-color-on-primary: #000000;
  }

  /* Base HUD Container Reset */
  section {
    background: transparent !important;
    color: var(--ar-text-primary) !important;
    box-shadow: none !important;
    border: none !important;
  }

  /* HUD-style Interactive Elements */
  button {
    background: rgba(34, 197, 94, 0.2) !important; /* bg-green-500/20 */
    border: 1px solid var(--ar-green-500) !important;
    border-radius: 9999px !important; /* rounded-full */
    backdrop-filter: blur(8px);
    color: white !important;
    font-family: 'Rajdhani', monospace !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }

  button:hover {
    background: rgba(34, 197, 94, 0.4) !important; /* bg-green-500/40 */
    transform: translateY(-1px);
    box-shadow: 0 0 15px rgba(34, 197, 94, 0.4) !important;
  }

  /* Typography */
  h1, h2, h3, h4, h5, h6 {
    color: var(--ar-text-primary) !important;
    font-family: 'Rajdhani', monospace !important;
    letter-spacing: 0.05em;
    text-transform: none;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
  }

  p, span, div {
    color: rgba(0, 255, 65, 0.9) !important;
  }

  /* ========================================
   * A2UI Variant Classes - Semantic Token System
   * ======================================== */

  /* Card Variants - True AR Overlay Style */
  .a2ui-card-glass {
    background: rgba(0, 10, 0, 0.2) !important; /* High transparency for AR */
    backdrop-filter: blur(4px) !important; /* Reduced blur for clarity */
    -webkit-backdrop-filter: blur(4px) !important;
    border-left: 1px solid var(--ar-border-dim) !important;
    border-top: 1px solid var(--ar-border-dim) !important;
    border-bottom: 1px solid var(--ar-border-dim) !important;
    /* Open on right side or corner style */
    border-radius: 12px !important;
    box-shadow: none !important;
  }

  .a2ui-card-solid {
    background: #000000 !important;
    border: 1px solid var(--ar-border-dim) !important;
    border-radius: 16px !important;
  }

  .a2ui-card-outline {
    background: transparent !important;
    border: 1px solid var(--ar-border-dim) !important;
    border-radius: 12px !important;
    backdrop-filter: none !important;
  }
  
  /* Initial "HUD Panel" feel for standard cards */
  .a2ui-card {
    position: relative;
  }
  
  /* Decorative bracket for Cards */
  .a2ui-card::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px;
    width: 20px; height: 20px;
    border-top: 2px solid var(--ar-green-500);
    border-left: 2px solid var(--ar-green-500);
    border-radius: 4px 0 0 0;
    opacity: 0.8;
    pointer-events: none;
  }

  .a2ui-card-alert {
    background: rgba(239, 68, 68, 0.2) !important; /* bg-red-500/20 */
    border-left: 4px solid #ef4444 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    color: #ffffff !important;
  }

  /* Text Variants */
  .a2ui-text-hero {
    font-size: 36px !important;
    font-weight: 700 !important;
    color: var(--ar-text-primary) !important;
    text-shadow: 0 0 10px rgba(74, 222, 128, 0.4);
  }

  .a2ui-text-h1 {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--ar-text-primary) !important;
  }

  .a2ui-text-h2 {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: var(--ar-text-primary) !important;
  }

  .a2ui-text-body {
    font-size: 16px !important;
    font-weight: 400 !important;
    color: var(--ar-text-primary) !important;
  }

  .a2ui-text-caption {
    font-size: 13px !important;
    font-weight: 400 !important;
    color: var(--ar-text-secondary) !important;
    font-family: 'Rajdhani', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .a2ui-text-label {
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--ar-green-400) !important;
  }

  /* Button Variants */
  .a2ui-btn-primary {
    background: rgba(34, 197, 94, 0.2) !important;
    border: 1px solid var(--ar-green-500) !important;
    color: #ffffff !important;
    padding: 8px 24px !important;
    border-radius: 9999px !important;
    box-shadow: 0 0 15px rgba(34, 197, 94, 0.3) !important;
  }

  .a2ui-btn-primary:hover {
    background: rgba(34, 197, 94, 0.4) !important;
    box-shadow: 0 0 25px rgba(34, 197, 94, 0.5) !important;
  }

  .a2ui-btn-secondary {
    background: transparent !important;
    border: 1px solid var(--ar-text-secondary) !important;
    color: var(--ar-text-primary) !important;
    padding: 8px 24px !important;
    border-radius: 9999px !important;
  }

  .a2ui-btn-ghost {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: rgba(0, 255, 65, 0.8) !important;
    padding: 10px 24px !important;
    text-decoration: underline;
  }

  .a2ui-btn-ghost:hover {
    background: rgba(0, 255, 65, 0.1) !important;
    color: #00ff41 !important;
  }

  .a2ui-btn-icon {
    background: rgba(255, 255, 255, 0.1) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
  }

  /* Badge Variants */
  .a2ui-badge-info {
    background: rgba(59, 130, 246, 0.2) !important;
    color: #60a5fa !important;
    padding: 4px 10px !important;
    border-radius: 12px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
  }

  .a2ui-badge-success {
    background: rgba(34, 197, 94, 0.2) !important;
    color: #4ade80 !important;
    padding: 4px 10px !important;
    border-radius: 12px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
  }

  .a2ui-badge-warning {
    background: rgba(245, 158, 11, 0.2) !important;
    color: #fbbf24 !important;
    padding: 4px 10px !important;
    border-radius: 12px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
  }

  .a2ui-badge-error {
    background: rgba(239, 68, 68, 0.2) !important;
    color: #f87171 !important;
    padding: 4px 10px !important;
    border-radius: 12px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
  }

  /* Icon Size Variants */
  .a2ui-icon-sm {
    width: 16px !important;
    height: 16px !important;
    font-size: 14px !important;
  }

  .a2ui-icon-md {
    width: 24px !important;
    height: 24px !important;
    font-size: 20px !important;
  }

  .a2ui-icon-lg {
    width: 32px !important;
    height: 32px !important;
    font-size: 28px !important;
  }

  /* ProgressBar Variants */
  .a2ui-progress-default {
    background: rgba(0, 255, 65, 0.1) !important;
    border: 1px solid rgba(0, 255, 65, 0.3) !important;
    border-radius: 2px !important;
    height: 10px !important;
    overflow: hidden !important;
  }

  .a2ui-progress-default .a2ui-progress-fill {
    background: #00ff41 !important;
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.5) !important;
    height: 100% !important;
    transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  }

  .a2ui-progress-slim {
    background: rgba(0, 255, 65, 0.05) !important;
    height: 4px !important;
    overflow: hidden !important;
  }

  .a2ui-progress-slim .a2ui-progress-fill {
    background: #00ff41 !important;
    height: 100% !important;
  }
`;

export const structuralStyles = css`
  ${unsafeCSS(Styles.structuralStyles)}
  ${arGlassStyles}
`;
