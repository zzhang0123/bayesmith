(() => {
  'use strict';
  document.querySelectorAll('[data-math], [data-tex]').forEach(element => {
    try {
      window.katex.render(element.dataset.math ?? element.dataset.tex, element, {
        displayMode: element.hasAttribute('data-math'), throwOnError: true,
        strict: 'error', trust: false, output: 'htmlAndMathml'
      });
    } catch (error) {
      element.classList.add('math-error');
      console.error('Math rendering failed', error);
    }
  });
  document.querySelectorAll('[data-viewer]').forEach(viewer => {
    const controls = viewer.querySelector('.viewer-controls');
    const buttons = [...controls.querySelectorAll('[data-view-target]')];
    const panels = buttons.map(button => document.getElementById(button.dataset.viewTarget));
    controls.setAttribute('role', 'tablist');
    function select(index, focus = false) {
      buttons.forEach((button, i) => {
        button.setAttribute('aria-selected', String(i === index));
        button.tabIndex = i === index ? 0 : -1;
        panels[i].hidden = i !== index;
      });
      if (focus) buttons[index].focus();
    }
    buttons.forEach((button, i) => {
      button.setAttribute('role', 'tab');
      button.setAttribute('aria-controls', panels[i].id);
      panels[i].setAttribute('role', 'tabpanel');
      panels[i].setAttribute('aria-labelledby', button.id);
      panels[i].tabIndex = 0;
      button.addEventListener('click', () => select(i));
      button.addEventListener('keydown', event => {
        const next = {ArrowRight: (i + 1) % buttons.length,
          ArrowLeft: (i + buttons.length - 1) % buttons.length,
          Home: 0, End: buttons.length - 1}[event.key];
        if (next !== undefined) { event.preventDefault(); select(next, true); }
      });
    });
    select(0);
  });
  const root = document.documentElement;
  try {
    root.dataset.theme = localStorage.getItem('bayesmith-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  } catch (_) { root.dataset.theme = 'light'; }
  document.getElementById('theme-toggle').addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('bayesmith-theme', root.dataset.theme); } catch (_) { /* Storage may be unavailable for local files. */ }
  });
  const menu = document.getElementById('menu-toggle');
  const sidebar = document.getElementById('sidebar');
  menu.addEventListener('click', () => {
    const open = menu.getAttribute('aria-expanded') !== 'true';
    menu.setAttribute('aria-expanded', String(open));
    sidebar.classList.toggle('is-open', open);
  });
  const dialog = document.getElementById('search-dialog');
  const input = document.getElementById('search-input');
  const results = document.getElementById('search-results');
  const status = document.getElementById('search-status');
  const script = [...document.scripts].find(s => s.src.endsWith('/assets/docs.js'));
  const base = new URL('../', script.src);
  function openSearch() { dialog.showModal(); input.focus(); }
  document.getElementById('search-toggle').addEventListener('click', openSearch);
  document.addEventListener('keydown', event => {
    if (event.key === '/' && !event.ctrlKey && !event.metaKey && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
      event.preventDefault(); if (!dialog.open) openSearch();
    }
    if (event.key === 'Escape' && sidebar.classList.contains('is-open')) {
      sidebar.classList.remove('is-open'); menu.setAttribute('aria-expanded', 'false'); menu.focus();
    }
  });
  input.addEventListener('input', () => {
    const query = input.value.trim().toLowerCase();
    results.replaceChildren();
    if (!query) { status.textContent = 'Type a concept or API name.'; return; }
    const words = query.split(/\s+/);
    function rank(page) {
      const title = page.title.toLowerCase();
      const leaf = title.split('.').pop();
      return (title === query ? 2000 : 0) + (leaf === query ? 1000 : 0)
        + (title.includes(query) ? 100 : 0) + (page.kind === 'module' ? 10 : 0);
    }
    const found = (window.BAYESMITH_SEARCH || []).filter(p => words.every(w => (p.title + ' ' + p.text).toLowerCase().includes(w)))
      .sort((a, b) => rank(b) - rank(a));
    status.textContent = found.length ? `${found.length} results. Showing the first ${Math.min(found.length, 30)}.` : 'No results. Try a shorter concept or an API name.';
    found.slice(0, 30).forEach(page => {
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = new URL(page.path, base).href;
      link.textContent = page.title;
      const description = document.createElement('small');
      description.textContent = page.text.replace(/\s+/g, ' ').slice(0, 145);
      link.append(description); item.append(link); results.append(item);
    });
  });
})();
