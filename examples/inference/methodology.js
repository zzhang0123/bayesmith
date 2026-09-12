// Paired translations share controls, formulas and evidence; no inference runs here.
(() => {
  const guide = document.getElementById('methodology');
  if (!guide) return;
  const chapters = [...guide.querySelectorAll('[data-design-tab]')].map(b => b.dataset.designTab);
  const aliases = { overview: 'methods', document: 'proofs', implementation: 'proofs' };
  const languages = ['en', 'zh'];

  function exclusive(buttonKey, panelKey, value) {
    guide.querySelectorAll(`[${buttonKey}]`).forEach(button => {
      button.setAttribute('aria-pressed', String(button.getAttribute(buttonKey) === value));
    });
    guide.querySelectorAll(`[${panelKey}]`).forEach(panel => {
      panel.hidden = panel.getAttribute(panelKey) !== value;
    });
  }

  function setLanguage(language) {
    const lang = languages.includes(language) ? language : 'en';
    guide.dataset.language = lang;
    guide.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.documentElement.dataset.guideLanguage = lang;
    document.querySelectorAll('[data-language-choice]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.languageChoice === lang));
    });
    document.documentElement.lang = guide.lang;
    document.querySelectorAll('a[data-gallery-link]').forEach(link => {
      const target = new URL(link.getAttribute('href'), location.href);
      target.searchParams.set('lang', lang);
      link.href = target.href;
    });
    window.dispatchEvent(new Event('notebooklanguagechange'));
    const url = new URL(location.href);
    url.searchParams.set('lang', lang);
    history.replaceState(null, '', url);
  }

  function route() {
    if (!location.hash.startsWith('#design')) {
      document.documentElement.lang = guide.lang || 'en';
      return;
    }
    const parts = location.hash.slice(1).split('/');
    const chapter = chapters.includes(parts[1]) ? parts[1] : aliases[parts[1]] || 'methods';
    exclusive('data-design-tab', 'data-design-panel', chapter);
    document.documentElement.lang = guide.lang || 'en';
    if (chapter === 'methods' && /^[1-8]$/.test(parts[2] || '')) {
      guide.querySelector(`#method-${parts[2]}`).scrollIntoView({block: 'center', behavior: 'instant'});
    }
    // Old source-section bookmarks still reach the preserved full reference.
    const referenceLink = guide.querySelector('[data-reference-link]');
    referenceLink.href = parts[1] === 'document' && /^\d+$/.test(parts[2] || '')
      ? `design-reference.html#design-source-${parts[2]}` : 'design-reference.html';
  }

  guide.querySelectorAll('[data-design-tab], [data-design-go]').forEach(button => {
    button.addEventListener('click', () => {
      const chapter = button.dataset.designTab || button.dataset.designGo;
      location.hash = `design/${chapter}`;
      exclusive('data-design-tab', 'data-design-panel', chapter);
      guide.querySelector(`[data-design-tab="${chapter}"]`).focus({ preventScroll: true });
      guide.querySelector('.design-controls').scrollIntoView({ block: 'start', behavior: 'instant' });
    });
  });
  document.querySelectorAll('[data-language-choice]').forEach(button => {
    button.addEventListener('click', () => setLanguage(button.dataset.languageChoice));
  });
  guide.querySelectorAll('[data-goal]').forEach(button => button.addEventListener('click', () => {
    exclusive('data-goal', 'data-goal-panel', button.dataset.goal);
  }));
  guide.querySelectorAll('[data-stop]').forEach(button => button.addEventListener('click', () => {
    exclusive('data-stop', 'data-stop-panel', button.dataset.stop);
  }));
  for (const [selector, attribute] of [
    ['.design-tabs', 'data-design-tab'], ['.goal-switch', 'data-goal'],
    ['.language-switch', 'data-language-choice'],
    ['.stop-switch', 'data-stop'],
  ]) {
    guide.querySelector(selector).addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      const buttons = [...guide.querySelectorAll(`[${attribute}]`)];
      const current = buttons.indexOf(event.target);
      if (current < 0) return;
      event.preventDefault();
      const index = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 :
        Math.max(0, Math.min(buttons.length - 1, current + (event.key === 'ArrowRight' ? 1 : -1)));
      buttons[index].click();
      buttons[index].focus();
    });
  }
  exclusive('data-design-tab', 'data-design-panel', 'methods');
  exclusive('data-goal', 'data-goal-panel', 'estimate');
  exclusive('data-stop', 'data-stop-panel', 'fixed');
  route();
  setLanguage(new URL(location.href).searchParams.get('lang'));
  window.addEventListener('hashchange', route);
  let printDetails = [];
  window.addEventListener('beforeprint', () => {
    if (guide.hidden) return;
    printDetails = [...guide.querySelectorAll('details')].map(node => [node, node.open]);
    printDetails.forEach(([node]) => { node.open = true; });
  });
  window.addEventListener('afterprint', () => {
    printDetails.forEach(([node, wasOpen]) => { node.open = wasOpen; });
    printDetails = [];
  });
  if (typeof katex !== 'undefined') {
    document.querySelectorAll('[data-math], [data-tex]').forEach(node => {
      katex.render(node.dataset.math || node.dataset.tex, node, {
        displayMode: node.hasAttribute('data-math'), throwOnError: false,
        trust: false, output: 'htmlAndMathml',
      });
    });
  }
})();
