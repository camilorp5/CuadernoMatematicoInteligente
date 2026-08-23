/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export', // Genera HTML/CSS/JS estático sin servidor Node
  images: {
    unoptimized: true, // Requerido para Next/Image en exportación estática
  },
  // Si tu repo en GitHub no es username.github.io, sino username.github.io/smart-math-v1:
  // basePath: '/smart-math-v1',
};

export default nextConfig;
