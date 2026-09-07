'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import katex from 'katex';
import { SYMBOLS_LIST, MODEL_METRICS, ClassMetric } from '@/data/symbolsMetrics';

type OrtModule = typeof import('onnxruntime-web');

interface TopPrediction {
  symbol: string;
  probability: number;
}

export default function MathSymbolsPhase() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mathRenderRef = useRef<HTMLDivElement | null>(null);

  const [ort, setOrt] = useState<OrtModule | null>(null);
  const [session, setSession] = useState<any | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isReady, setIsReady] = useState(false);

  // Estados de predicción
  const [predictedSymbol, setPredictedSymbol] = useState<string | null>(null);
  const [topProbabilities, setTopProbabilities] = useState<TopPrediction[]>([]);
  const [currentMetric, setCurrentMetric] = useState<ClassMetric | null>(null);

  // 1. Limpieza del canvas
  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPredictedSymbol(null);
    setTopProbabilities([]);
    setCurrentMetric(null);

    if (mathRenderRef.current) {
      mathRenderRef.current.innerHTML = '';
    }
  }, []);

  // 2. Renderizado dinámico de KaTeX cuando cambia la predicción
  useEffect(() => {
    if (predictedSymbol && mathRenderRef.current) {
      try {
        katex.render(predictedSymbol, mathRenderRef.current, {
          throwOnError: false,
          displayMode: true,
        });
      } catch (err) {
        console.error('KaTeX render error:', err);
      }
    }
  }, [predictedSymbol]);

  // 3. Inicialización del modelo ONNX (/public/models/math_symbols.onnx)
  useEffect(() => {
    let active = true;

    async function loadMathModel() {
      console.group('🔍 [ONNX Symbols] Cargando modelo multiclase');
      try {
        const ortModule = await import('onnxruntime-web');
        setOrt(ortModule);

        ortModule.env.wasm.wasmPaths = '/onnx-wasm/';
        ortModule.env.wasm.numThreads = 1;
        ortModule.env.wasm.simd = false;

        const modelCheck = await fetch('/models/math_symbols.onnx');
        if (!modelCheck.ok) {
          throw new Error(`HTTP ${modelCheck.status}: No se encontró /models/math_symbols.onnx`);
        }

        const inferenceSession = await ortModule.InferenceSession.create(
          '/models/math_symbols.onnx',
          { executionProviders: ['wasm'] }
        );

        if (active) {
          setSession(inferenceSession);
          setIsReady(true);
          console.log('✔ Sesión multiclase lista.');
          console.log('Inputs:', inferenceSession.inputNames);
          console.log('Outputs:', inferenceSession.outputNames);
        }
      } catch (err: any) {
        console.error('❌ Error al inicializar math_symbols.onnx:', err);
      } finally {
        console.groupEnd();
      }
    }

    loadMathModel();
    clearCanvas();

    return () => {
      active = false;
    };
  }, [clearCanvas]);

  // 4. Preprocesamiento e inferencia (32x32)
  const runInference = useCallback(async () => {
    if (!session || !canvasRef.current || !ort) return;

    const canvas = canvasRef.current;

    // Redimensionar al tamaño esperado por el entrenamiento (32x32)
    const targetSize = 32;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = targetSize;
    tempCanvas.height = targetSize;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    tempCtx.drawImage(canvas, 0, 0, targetSize, targetSize);
    const imgData = tempCtx.getImageData(0, 0, targetSize, targetSize);
    const data = imgData.data;

    // Normalización HASYv2: mean = 0.1500, std = 0.3200
    const float32Data = new Float32Array(targetSize * targetSize);
    for (let i = 0; i < data.length; i += 4) {
      const avg = data[i] / 255.0; // Canal R
      const normalized = (avg - 0.1500) / 0.3200;
      float32Data[i / 4] = normalized;
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 1, targetSize, targetSize]);

    try {
      const outputMap = await session.run({ [session.inputNames[0]]: inputTensor });
      const outputData = outputMap[session.outputNames[0]].data as Float32Array;

      // Softmax numéricamente estable
      const maxLogit = Math.max(...Array.from(outputData));
      const expValues = Array.from(outputData).map((v) => Math.exp(v - maxLogit));
      const sumExp = expValues.reduce((a, b) => a + b, 0);
      const softmaxProbs = expValues.map((v) => v / sumExp);

      // Extraer Top-5
      const mapped = softmaxProbs.map((prob, idx) => ({
        symbol: SYMBOLS_LIST[idx] ?? `C${idx}`,
        probability: prob,
      }));

      mapped.sort((a, b) => b.probability - a.probability);
      const top5 = mapped.slice(0, 5);

      const winner = top5[0].symbol;
      setPredictedSymbol(winner);
      setTopProbabilities(top5);
      setCurrentMetric(MODEL_METRICS[winner] ?? null);
    } catch (err) {
      console.error('Error durante inferencia de símbolos:', err);
    }
  }, [session, ort]);

  // 5. Handlers de Dibujo
  const getCoordinates = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;

    const { x, y } = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    const { x, y } = getCoordinates(e);
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'white';

    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) ctx.beginPath();
    runInference();
  };

  return (
    <div className="flex flex-col items-center w-full max-w-5xl animate-in fade-in zoom-in-95 duration-500">
      {/* Título & Badge de Arquitectura */}
      <div className="flex flex-col items-center mb-6 text-center">
        <div className="flex items-center gap-2 mb-2">
          <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-indigo-950 text-indigo-300 border border-indigo-800">
            Fase 2: 44 Clases
          </span>
          <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-zinc-900 text-zinc-400 border border-zinc-800">
            WASM + ONNX Runtime
          </span>
        </div>
        <h2 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
          Reconocimiento de Símbolos Matemáticos
        </h2>
        <p className="text-sm text-zinc-400 mt-1">
          Inferencia cliente con proyección directa a LaTeX y auditoría de métricas de entrenamiento.
        </p>
      </div>

      <div className="flex flex-col lg:flex-row gap-6 items-start w-full justify-center">
        {/* Columna Izquierda: Canvas interactivo */}
        <div className="flex flex-col items-center gap-4 bg-zinc-900/80 p-6 rounded-2xl border border-zinc-800 shadow-xl backdrop-blur-sm">
          <div className="relative">
            <canvas
              ref={canvasRef}
              width={280}
              height={280}
              onMouseDown={startDrawing}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              onMouseMove={draw}
              onTouchStart={startDrawing}
              onTouchEnd={stopDrawing}
              onTouchMove={draw}
              className="border-2 border-zinc-700 rounded-xl cursor-crosshair bg-black touch-none shadow-inner"
            />
            {!isReady && (
              <div className="absolute inset-0 bg-black/80 flex items-center justify-center rounded-xl font-mono text-xs text-indigo-400">
                Cargando runtime ONNX...
              </div>
            )}
          </div>

          <button
            onClick={clearCanvas}
            className="px-4 py-2.5 bg-rose-600/90 hover:bg-rose-500 transition-colors font-semibold rounded-lg text-sm w-full text-white tracking-wide shadow-md"
          >
            Limpiar Trazo
          </button>
        </div>

        {/* Columna Derecha: Dos Rectángulos Horizontales */}
        <div className="flex flex-col gap-4 w-full lg:w-[480px]">
          {/* Rectángulo Superior: Dual LaTeX Display */}
          <div className="bg-zinc-900/80 p-5 rounded-2xl border border-zinc-800 shadow-xl backdrop-blur-sm flex flex-col gap-3">
            <div className="flex justify-between items-center border-b border-zinc-800 pb-2">
              <span className="text-xs font-mono uppercase tracking-wider text-zinc-400 font-semibold">
                Salida Estructurada LaTeX
              </span>
              <span className="text-xs font-mono text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/60">
                Direct Token
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4 items-center">
              {/* Token Crudo */}
              <div className="flex flex-col bg-zinc-950 p-4 rounded-xl border border-zinc-800/80">
                <span className="text-[10px] font-mono text-zinc-500 uppercase">Raw Code</span>
                <code className="text-lg font-mono text-amber-400 font-bold mt-1 truncate">
                  {predictedSymbol !== null ? predictedSymbol : '-'}
                </code>
              </div>

              {/* Render Visual KaTeX */}
              <div className="flex flex-col bg-zinc-950 p-4 rounded-xl border border-zinc-800/80 min-h-[76px] justify-center items-center">
                <span className="text-[10px] font-mono text-zinc-500 uppercase self-start mb-1">Visual Render</span>
                <div
                  ref={mathRenderRef}
                  className="text-2xl font-serif text-white tracking-wide"
                />
                {!predictedSymbol && <span className="text-zinc-600 font-mono text-sm">-</span>}
              </div>
            </div>
          </div>

          {/* Rectángulo Inferior: Top-5 Probabilidades & Distribución */}
          <div className="bg-zinc-900/80 p-5 rounded-2xl border border-zinc-800 shadow-xl backdrop-blur-sm flex flex-col gap-4">
            <div className="flex justify-between items-center border-b border-zinc-800 pb-2">
              <span className="text-xs font-mono uppercase tracking-wider text-zinc-400 font-semibold">
                Distribución de Probabilidad (Top-5)
              </span>
              <span className="text-xs font-mono text-zinc-500">Softmax Normalizado</span>
            </div>

            <div className="flex items-end justify-around gap-2 h-36 pt-4 px-2 bg-zinc-950 rounded-xl border border-zinc-800/80">
              {topProbabilities.length > 0 ? (
                topProbabilities.map((item, idx) => {
                  const heightPercent = Math.max(item.probability * 100, 6);
                  const isTop1 = idx === 0;

                  return (
                    <div key={item.symbol} className="flex flex-col items-center h-full justify-end flex-1 max-w-[56px]">
                      <span className="text-[10px] font-mono text-zinc-400 mb-1">
                        {(item.probability * 100).toFixed(0)}%
                      </span>
                      <div className="w-full bg-zinc-900 rounded-t-md overflow-hidden flex items-end justify-center h-20 border border-zinc-800">
                        <div
                          style={{ height: `${heightPercent}%` }}
                          className={`w-full transition-all duration-300 ease-out rounded-t-sm ${
                            isTop1
                              ? 'bg-indigo-500 shadow-[0_0_12px_rgba(99,102,241,0.6)]'
                              : 'bg-zinc-700'
                          }`}
                        />
                      </div>
                      <span className="text-xs font-mono mt-2 text-zinc-300 truncate max-w-[48px] text-center" title={item.symbol}>
                        {item.symbol}
                      </span>
                    </div>
                  );
                })
              ) : (
                <div className="text-zinc-600 text-xs font-mono flex items-center justify-center h-full">
                  Dibuja un trazo para evaluar probabilidades
                </div>
              )}
            </div>

            {/* Factor Diferenciador: Model Observability / Metrics Card */}
            {predictedSymbol && currentMetric && (
              <div className="mt-1 pt-3 border-t border-zinc-800/80 flex flex-col gap-2">
                <div className="flex justify-between items-center">
                  <span className="text-[11px] font-mono text-zinc-400 uppercase">
                    Métricas de Entrenamiento ({predictedSymbol})
                  </span>
                  <span className="text-[10px] font-mono text-zinc-500">
                    Soporte: {currentMetric.support} muestras
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center font-mono">
                  <div className="bg-zinc-950/60 p-1.5 rounded border border-zinc-800">
                    <div className="text-[10px] text-zinc-500">Precision</div>
                    <div className="text-xs font-bold text-zinc-300">{(currentMetric.precision * 100).toFixed(1)}%</div>
                  </div>
                  <div className="bg-zinc-950/60 p-1.5 rounded border border-zinc-800">
                    <div className="text-[10px] text-zinc-500">Recall</div>
                    <div className="text-xs font-bold text-zinc-300">{(currentMetric.recall * 100).toFixed(1)}%</div>
                  </div>
                  <div className="bg-zinc-950/60 p-1.5 rounded border border-zinc-800">
                    <div className="text-[10px] text-zinc-500">F1-Score</div>
                    <div className={`text-xs font-bold ${currentMetric['f1-score'] < 0.5 ? 'text-amber-400' : 'text-emerald-400'}`}>
                      {(currentMetric['f1-score'] * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Alerta inteligente sobre debilidades conocidas en el dataset */}
                {currentMetric['f1-score'] < 0.5 && (
                  <div className="text-[11px] font-mono text-amber-400/90 bg-amber-950/30 p-2 rounded border border-amber-900/40">
                    ⚠️ Ambigüedad topológica: el modelo confunde {predictedSymbol} con operadores similares (\times).
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}