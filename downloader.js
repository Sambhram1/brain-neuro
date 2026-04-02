#!/usr/bin/env node
// Usage: node downloader.js <url> <output_dir>
// Prints the downloaded file path to stdout, or writes ERROR: ... to stderr and exits 1.
// Requires yt-dlp to be installed (already installed by colab_backend.ipynb).

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function main() {
  const [,, url, outputDir] = process.argv;
  if (!url || !outputDir) {
    process.stderr.write('Usage: node downloader.js <url> <output_dir>\n');
    process.exit(1);
  }

  const outTemplate = path.join(outputDir, 'video.%(ext)s');

  const args = [
    '--no-playlist',
    '--format', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    '--merge-output-format', 'mp4',
    '--output', outTemplate,
    '--quiet',
    '--no-warnings',
  ];

  // Use cookies.txt if present (enables Instagram/TikTok private/login-gated content)
  const cookiesPath = path.join(__dirname, 'cookies.txt');
  if (fs.existsSync(cookiesPath)) {
    args.push('--cookies', cookiesPath);
  }

  args.push(url);

  const result = spawnSync('yt-dlp', args, { stdio: ['ignore', 'pipe', 'pipe'] });

  if (result.error) {
    process.stderr.write('ERROR: yt-dlp not found — install it with: pip install yt-dlp\n');
    process.exit(1);
  }

  if (result.status !== 0) {
    const errMsg = (result.stderr || result.stdout || Buffer.alloc(0)).toString().trim();
    process.stderr.write('ERROR: ' + (errMsg || `yt-dlp exited with code ${result.status}`) + '\n');
    process.exit(1);
  }

  // Find the downloaded mp4 file in outputDir
  const files = fs.readdirSync(outputDir).filter(f => f.endsWith('.mp4'));
  if (files.length === 0) {
    process.stderr.write('ERROR: yt-dlp ran but no .mp4 file found in output dir\n');
    process.exit(1);
  }

  const videoPath = path.join(outputDir, files[0]);
  process.stdout.write(videoPath + '\n');
}

main();
