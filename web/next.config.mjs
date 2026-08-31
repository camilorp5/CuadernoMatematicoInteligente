/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  webpack: (config, { isServer }) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      'onnxruntime-node$': false,
    };

    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }

    config.module.rules.push({
      test: /ort\.node\.min\.(mjs|js)$/,
      loader: 'null-loader',
    });

    config.optimization = {
      ...config.optimization,
      minimizer: (config.optimization?.minimizer ?? []).map((plugin) => {
        if (plugin && typeof plugin === 'object' && 'options' in plugin) {
          plugin.options = {
            ...plugin.options,
            exclude: /ort\.bundle\.min\./i,
          };
        }
        return plugin;
      }),
    };

    return config;
  },
};

export default nextConfig;