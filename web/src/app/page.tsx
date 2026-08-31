'use client';

import React, { useRef, useState, useEffect } from 'react';
import * as ort from 'onnxruntime-web';

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [probabilities, setProbabilities] = useState<number[]>(Array(10).fill(0));
  const [predictedDigit, setPredictedDigit] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // 1. Cargar el modelo ONNX en el cliente
  useEffect(() => {
    async function loadModel() {
      try {
        // Asegúrate de que mnist_cnn.onnx esté en public/
        const sess = await ort.InferenceSession.create('/mnist_cnn.onnx');
        setSession(sess);
      } catch (e) {
        console.error("Error al cargar el modelo ONNX:", e);
      }
    }
    loadModel();
  }, []);

  // 2. Preprocesamiento de la imagen del Canvas
  const runInference = async () => {
    if (!session || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Crear canvas temporal de 28x28 para redimensionar
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    if (!tempCtx) return;

    // Redimensionar el contenido del canvas principal a 28x28
    tempCtx.drawImage(canvas, 0, 0, 28, 28);
    const imgData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imgData.data;

    // Convertir RGBA a 1 Canal Grayscale + Normalizar (mean=0.1307, std=0.3081)
    const float32Data = new Float32Array(28 * 28);
    for (let i = 0; i < data.length; i += 4) {
      // Tomamos el canal Alpha/Rojo para el fondo negro y trazo blanco
      const avg = data[i] / 255.0; 
      const normalized = (avg - 0.1307) / 0.3081;
      float32Data[i / 4] = normalized;
    }

    // Tensor con forma [1, 1, 28, 28]
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 1, 28, 28]);

    try {
      const outputMap = await session.run({ [session.inputNames[0]]: inputTensor });
      const outputData = outputMap[session.outputNames[0]].data as Float32Array;

      // Softmax para convertir logits a probabilidades
      const expValues = Array.from(outputData).map(val => Math.exp(val));
      const sumExp = expValues.reduce((a, b) => a + b, 0);
      const softmaxProbs = expValues.map(v => v / sumExp);

      setProbabilities(softmaxProbs);
      setPredictedDigit(softmaxProbs.indexOf(Math.max(...softmaxProbs)));
    } catch (err) {
      console.error("Error durante inferencia:", err);
    }
  };

  // Eventos de Dibujo
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

    runInference(); // Inferencia en tiempo real por cada trazo
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
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-zinc-950 text-white font-sans">
      <h1 className="text-3xl font-bold mb-6 tracking-tight">Cuaderno Inteligente — Motor MNIST</h1>

      <div className="flex flex-col md:flex-row gap-8 items-center bg-zinc-900 p-6 rounded-xl border border-zinc-800 shadow-2xl">
        {/* Canvas de Dibujo */}
        <div className="flex flex-col items-center gap-4">
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            onMouseDown={startDrawing}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onMouseMove={draw}
            className="border-2 border-zinc-700 rounded-lg cursor-crosshair bg-black touch-none"
          />
          <button
            onClick={clearCanvas}
            className="px-4 py-2 bg-rose-600 hover:bg-rose-700 transition-colors font-semibold rounded-md text-sm w-full"
          >
            Limpiar Trazo
          </button>
        </div>

        {/* Panel de Predicciones Animadas */}
        <div className="w-64 flex flex-col gap-2">
          <div className="text-center mb-2">
            <span className="text-xs text-zinc-400 uppercase tracking-widest">Predicción principal</span>
            <div className="text-6xl font-black text-indigo-400 mt-1">
              {predictedDigit !== null ? predictedDigit : '-'}
            </div>
          </div>

          <div className="space-y-1.5">
            {probabilities.map((prob, digit) => (
              <div key={digit} className="flex items-center gap-2 text-xs">
                <span className="w-3 text-right font-mono text-zinc-400">{digit}</span>
                <div className="flex-1 bg-zinc-800 h-3 rounded-full overflow-hidden">
                  <div
                    className="bg-indigo-500 h-full transition-all duration-75 ease-out rounded-full"
                    style={{ width: `${(prob * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-zinc-500">
                  {(prob * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}