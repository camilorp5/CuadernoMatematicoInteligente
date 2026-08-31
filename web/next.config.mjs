/** @type {import('next').NextConfig} */
const nextConfig = {
  // Transpila el paquete para resolver correctamente los módulos ES
  transpilePackages: ['onnxruntime-web'],

  webpack: (config, { isServer }) => {
    // Activa características experimentales de Webpack para soporte WASM
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      topLevelAwait: true,
    };

    // Redirige la importación a la build compatible con Node/Webpack sin dist.mjs
    config.resolve.alias = {
      ...config.resolve.alias,
      'onnxruntime-web': 'onnxruntime-web/dist/ort.node.js',
    };

    // Ignora los módulos nativos de Node.js en el bundle del cliente
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