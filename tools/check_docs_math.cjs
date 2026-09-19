#!/usr/bin/env node
'use strict';
// Validate exactly the expressions collected from generated HTML by build_docs.py.
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(process.argv[2] || 'site');
const katex = require(path.join(directory, 'assets/vendor/katex/katex.min.js'));
const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'manifest.json'), 'utf8'));
if (!manifest.math || !Object.keys(manifest.math).length) throw new Error('No math inventory; rebuild the documentation.');
let count = 0;
for (const [page, expressions] of Object.entries(manifest.math)) {
  for (const expression of expressions) {
    try {
      katex.renderToString(expression.tex, {displayMode: expression.display,
        throwOnError: true, strict: 'error', trust: false, output: 'htmlAndMathml'});
      count++;
    } catch (error) { throw new Error(`${page}: ${expression.tex}\n${error.message}`); }
  }
}
console.log(`${count} math expressions validated on ${Object.keys(manifest.math).length} pages (KaTeX ${katex.version}).`);
