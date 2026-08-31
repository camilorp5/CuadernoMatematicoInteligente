/** @type {import('next').NextConfig} */
const nextConfig = {
  // Ignora o configura el empaquetado del paquete ONNX Runtime
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      'onnxruntime-web': 'onnxruntime-web/dist/ort.all.min.js',
    };
    return config;
  },
};

export default nextConfig;