/** @type {import('next').NextConfig} */
const path = require('path');

const nextConfig = {
  output: 'standalone',
  turbopack: {
    root: __dirname,
  },
};

module.exports = nextConfig;
