/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  // Requerido si despliegas en un repositorio de usuario/proyecto (ej. usuario.github.io/nombre-repo)
  // basePath: '/nombre-de-tu-repositorio', 
  images: {
    unoptimized: true, // GitHub Pages no soporta la optimización por servidor de Next Image
  },
};

module.exports = nextConfig;