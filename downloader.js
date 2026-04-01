#!/usr/bin/env node
// Usage: node downloader.js <url> <output_dir>
// Prints the downloaded file path to stdout, or writes ERROR: ... to stderr and exits 1.

const { igdl, youtube, ttdl, aio } = require('ab-downloader');
const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

async function getDirectUrl(url) {
  if (url.includes('instagram.com')) {
    const res = await igdl(url);
    if (Array.isArray(res) && res[0]?.url) return res[0].url;
    throw new Error(res.message || 'Instagram: no download URL returned');
  }
  if (url.includes('youtube.com') || url.includes('youtu.be')) {
    const res = await youtube(url);
    if (res.mp4) return res.mp4;
    throw new Error(res.message || 'YouTube: no mp4 URL returned');
  }
  if (url.includes('tiktok.com')) {
    const res = await ttdl(url);
    if (res.video) return res.video;
    throw new Error(res.message || 'TikTok: no video URL returned');
  }
  // fallback: aio handles Twitter, Pinterest, Facebook, etc.
  const res = await aio(url);
  if (res.url) return res.url;
  throw new Error(res.message || 'Could not resolve download URL');
}

function downloadFile(url, dest, redirects = 0) {
  if (redirects > 5) return Promise.reject(new Error('Too many redirects'));
  return new Promise((resolve, reject) => {
    const proto = url.startsWith('https') ? https : http;
    const file = fs.createWriteStream(dest);
    proto.get(url, { headers: { 'User-Agent': 'Mozilla/5.0' } }, res => {
      if (res.statusCode === 301 || res.statusCode === 302 || res.statusCode === 307) {
        file.destroy();
        fs.unlink(dest, () => {});
        return downloadFile(res.headers.location, dest, redirects + 1).then(resolve).catch(reject);
      }
      if (res.statusCode !== 200) {
        file.destroy();
        return reject(new Error(`HTTP ${res.statusCode}`));
      }
      res.pipe(file);
      file.on('finish', () => file.close(() => resolve(dest)));
    }).on('error', err => { fs.unlink(dest, () => {}); reject(err); });
  });
}

async function main() {
  const [,, url, outputDir] = process.argv;
  if (!url || !outputDir) {
    process.stderr.write('Usage: node downloader.js <url> <output_dir>\n');
    process.exit(1);
  }
  try {
    const directUrl = await getDirectUrl(url);
    const dest = path.join(outputDir, 'video.mp4');
    await downloadFile(directUrl, dest);
    process.stdout.write(dest + '\n');
  } catch (err) {
    process.stderr.write('ERROR: ' + err.message + '\n');
    process.exit(1);
  }
}

main();
