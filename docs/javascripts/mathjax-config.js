// The `textmacros` extension is required for `\_` inside `\text{...}`.
// The bundled tex-svg build does not include it, and without it the content of
// `\text{...}` is taken verbatim, so an option name written as
// `\text{tile\_size}` renders with a visible backslash instead of an
// underscore. A plain underscore is not an alternative: inside `\text{...}` it
// is a TeX syntax error.
window.MathJax = {
  loader: {
    load: ['[tex]/textmacros']
  },
  tex: {
    packages: { '[+]': ['textmacros'] },
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true
  },
  svg: {
    fontCache: 'global'
  }
};
