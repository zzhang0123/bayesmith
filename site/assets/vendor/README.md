# Offline math rendering

KaTeX 0.18.7 browser assets, copied from the official npm package:
https://registry.npmjs.org/katex/-/katex-0.18.7.tgz

Included: minified JS/CSS, WOFF2 fonts, and the upstream MIT license.
The notebook copies these files alongside its generated HTML. It makes no CDN
requests. KaTeX rendering uses `trust: false`.

Browser integration: https://katex.org/docs/browser

Documentation copy: removed unused WOFF and TTF fallback URLs from the KaTeX stylesheet. All 20 WOFF2 fonts are bundled locally. JavaScript and WOFF2 files are unchanged.
