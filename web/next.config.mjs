/** @type {import('next').NextConfig} */
const nextConfig = {
  // Activar exportación si la usas, o dejar vacío si es app normal de Next
  webpack: (config, { isServer }) => {
    // 1. Desactivar módulos nativos de Node en el bundle web
    config.resolve.alias = {
      ...config.resolve.alias,
      'onnxruntime-node$': false,
    };

    // 2. Soportar experimentos de WebAssembly
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    };

    // 3. Evitar que webpack procese import.meta en scripts de ONNX
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