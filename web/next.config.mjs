/** @type {import('next').NextConfig} */
const nextConfig = {
  // Evita que Next.js empaquete la librería en el servidor de Node
  serverExternalPackages: ['onnxruntime-web'],

  webpack: (config, { isServer }) => {
    // Redirige al bundle CommonJS en lugar de ort.all.min.js
    config.resolve.alias = {
      ...config.resolve.alias,
      'onnxruntime-web': 'onnxruntime-web/dist/ort-wasm-simd-threaded.cjs',
    };

    // Ignora módulos nativos de Node cuando se compila para el cliente
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
      };
    }

    return config;
  },
};

export default nextConfig;