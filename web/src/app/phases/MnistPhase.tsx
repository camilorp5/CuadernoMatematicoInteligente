'use client';

import React, { useRef, useState, useEffect } from 'react';

type OrtModule = typeof import('onnxruntime-web');

export default function MnistPhase() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [ort, setOrt] = useState<OrtModule | null>(null);
  const [session, setSession] = useState<any | null>(null);
  const [probabilities, setProbabilities] = useState<number[]>(Array(10).fill(0));
  const [predictedDigit, setPredictedDigit] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadModel() {
      try {
        const ortModule = await import('onnxruntime-web');
        if (!active) return;

        setOrt(ortModule);
        const sess = await ortModule.InferenceSession.create('/mnist_cnn.onnx');
        setSession(sess);
      } catch (e) {
        console.error('Error al cargar el modelo ONNX:', e);
      }
    }

    loadModel();
    return () => {
      active = false;
    };
  }, []);

  const runInference = async () => {
    if (!session || !canvasRef.current || !ort) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imgData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;

    const float32Data = new Float32Array(28 * 28);
    for (let i = 0; i < data.length; i += 4) {
      const avg = data[i] / 255.0;
      const normalized = (avg - 0.1307) / 0.3081;
      float32Data[i / 4] = normalized;
    }

    const inputTensor = new ort.Tensor('float32', float32Data, [1, 1, 28, 28]);

    try {
      const outputMap = await session.run({ [session.inputNames[0]]: inputTensor });
      const outputData = outputMap[session.outputNames[0]].data as Float32Array;

      const expValues = Array.from(outputData).map((val) => Math.exp(val));
      const sumExp = expValues.reduce((a, b) => a + b, 0);
      const softmaxProbs = expValues.map((v) => v / sumExp);

      setProbabilities(softmaxProbs);
      setPredictedDigit(softmaxProbs.indexOf(Math.max(...softmaxProbs)));
    } catch (err) {
      console.error('Error durante inferencia:', err);
    }
  };

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
    const ctx = canvasRef.current?.getContext('2d');
    if (ctx) ctx.beginPath();
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);

    runInference(); 
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setProbabilities(Array(10).fill(0));
    setPredictedDigit(null);
  };

  return (
    <div className="flex flex-col items-center w-full max-w-4xl animate-in fade-in zoom-in-95 duration-500">
      <h2 className="text-2xl font-bold mb-8 tracking-tight text-white">Reconocimiento Base (MNIST)</h2>
      
      <div className="flex flex-col md:flex-row gap-8 items-center bg-zinc-900/80 p-8 rounded-2xl border border-zinc-800 shadow-2xl backdrop-blur-sm">
        <div className="flex flex-col items-center gap-4">
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            onMouseDown={startDrawing}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onMouseMove={draw}
            className="border-2 border-zinc-700 rounded-xl cursor-crosshair bg-black touch-none shadow-inner"
          />
          <button
            onClick={clearCanvas}
            className="px-4 py-2.5 bg-rose-600/90 hover:bg-rose-500 transition-colors font-semibold rounded-lg text-sm w-full text-white tracking-wide"
          >
            Limpiar Trazo
          </button>
        </div>

        <div className="w-64 flex flex-col gap-3">
          <div className="text-center mb-4">
            <span className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">Predicción</span>
            <div className="text-7xl font-black text-indigo-400 mt-2 drop-shadow-md">
              {predictedDigit !== null ? predictedDigit : '-'}
            </div>
          </div>

          <div className="space-y-2">
            {probabilities.map((prob, digit) => (
              <div key={digit} className="flex items-center gap-3 text-xs">
                <span className="w-3 text-right font-mono text-zinc-500">{digit}</span>
                <div className="flex-1 bg-zinc-950 h-2.5 rounded-full overflow-hidden border border-zinc-800">
                  <div
                    className="bg-indigo-500 h-full transition-all duration-100 ease-out rounded-full shadow-[0_0_10px_rgba(99,102,241,0.5)]"
                    style={{ width: `${(prob * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-zinc-400">
                  {(prob * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}