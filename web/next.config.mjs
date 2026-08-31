/** @type {import('next').NextConfig} */
const nextConfig = {
  // Transpila el paquete para que SWC entienda y procese los módulos ES/WASM
  transpilePackages: ['onnxruntime-web'],

  webpack: (config, { isServer }) => {
    // Permite el soporte nativo de WebAssembly en Webpack 5
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      topLevelAwait: true,
    };

    // Apunta al archivo CommonJS exacto dentro del paquete
    config.resolve.alias = {
      ...config.resolve.alias,
      'onnxruntime-web': 'onnxruntime-web/dist/ort.node.js',
    };

    // Ignora librerías de Node en la compilación del navegador
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        child_process: false,
      };
    }

    return config;
  },
};

export default nextConfig;