'use client';

import React, { useState, useEffect } from 'react';

// Importas tus vistas de cada fase
import MnistPhase from './phases/MnistPhase';
// import MathSymbolsPhase from './phases/MathSymbolsPhase'; // Próxima fase

const PHASES = [
  { id: 'v1', title: 'Fase 1: MNIST Baseline', component: <MnistPhase /> },
  { id: 'v2', title: 'Fase 2: Segmentación y Símbolos', component: <div className="p-10">Próximamente...</div> },
  { id: 'v3', title: 'Fase 3: Motor Simbólico', component: <div className="p-10">Próximamente...</div> },
];

export default function Home() {
  const [currentPhase, setCurrentPhase] = useState(0);

  const prevPhase = () => setCurrentPhase((prev) => Math.max(prev - 1, 0));
  const nextPhase = () => setCurrentPhase((prev) => Math.min(prev + 1, PHASES.length - 1));

  // Atajos de teclado (flechas izquierda / derecha)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') prevPhase();
      if (e.key === 'ArrowRight') nextPhase();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <main className="relative min-h-screen bg-zinc-950 text-white overflow-hidden flex flex-col justify-between">
      {/* Header / Indicator Superior */}
      <header className="py-4 px-8 flex justify-between items-center bg-zinc-900/50 backdrop-blur-md border-b border-zinc-800 z-10">
        <span className="font-mono text-sm text-zinc-400">Cuaderno Inteligente</span>
        
        {/* Stepper de Fases */}
        <div className="flex items-center gap-3">
          {PHASES.map((phase, idx) => (
            <button
              key={phase.id}
              onClick={() => setCurrentPhase(idx)}
              className={`px-3 py-1 rounded-full text-xs font-mono transition-all ${
                currentPhase === idx
                  ? 'bg-indigo-600 text-white font-bold scale-105'
                  : 'bg-zinc-800 text-zinc-500 hover:text-zinc-300'
              }`}
            >
              0{idx + 1}
            </button>
          ))}
        </div>
      </header>

      {/* Área Principal de la Fase Activa */}
      <div className="flex-1 flex items-center justify-center relative p-6">
        {PHASES[currentPhase].component}
      </div>

      {/* Flecha Izquierda */}
      {currentPhase > 0 && (
        <button
          onClick={prevPhase}
          className="absolute left-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-zinc-900/80 hover:bg-indigo-600 border border-zinc-700 hover:border-indigo-400 text-white transition-all shadow-lg backdrop-blur-sm z-20 group"
          aria-label="Fase Anterior"
        >
          <span className="block group-hover:-translate-x-0.5 transition-transform text-xl">←</span>
        </button>
      )}

      {/* Flecha Derecha */}
      {currentPhase < PHASES.length - 1 && (
        <button
          onClick={nextPhase}
          className="absolute right-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-zinc-900/80 hover:bg-indigo-600 border border-zinc-700 hover:border-indigo-400 text-white transition-all shadow-lg backdrop-blur-sm z-20 group"
          aria-label="Siguiente Fase"
        >
          <span className="block group-hover:translate-x-0.5 transition-transform text-xl">→</span>
        </button>
      )}
    </main>
  );
}