// View controls only: no simulation, inference, or network requests in the page.
(() => {
  const articles = [...document.querySelectorAll('[data-case]')];
  const caseButtons = [...document.querySelectorAll('[data-select-case]')];
  const keys = ['model', 'methods', 'diagnostics', 'sampling', 'recovery'];
  const aliases = {simulation: 'model', inference: 'sampling'};
  const labels = {en: ['model & simulation', 'methods and blocks', 'diagnostics & priors', 'sampling', 'recovery'], zh: ['模型与模拟', '方法与参数块', '诊断与先验', '采样', '恢复结果']};
  const steps = new Map(articles.map(article => [article.id, 0]));
  let activeCase = articles[0]?.id;
  let activeStep = 'model';
  const design = document.getElementById('methodology');
  const designButton = document.querySelector('[data-open-design]');

  function updateComparisonLinks() {
    document.querySelectorAll('a[data-preserve-step]').forEach(link => {
      const target = new URL(link.getAttribute('href'), location.href);
      const params = new URLSearchParams(target.hash.slice(1));
      params.set('step', activeStep);
      target.hash = params.toString();
      link.href = target.href;
    });
  }

  function showStep(article, index) {
    steps.set(article.id, index);
    if (article.id === activeCase) {
      activeStep = keys[index];
      updateComparisonLinks();
    }
    article.querySelectorAll('[data-stage]').forEach(stage => {
      stage.hidden = stage.dataset.stage !== keys[index];
    });
    article.querySelectorAll('[data-step]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.step === keys[index]));
    });
    article.querySelector('.previous').disabled = index === 0;
    const next = article.querySelector('.next');
    next.disabled = index === keys.length - 1;
    const lang = document.documentElement.dataset.guideLanguage === 'zh' ? 'zh' : 'en';
    const caseLabels = article.dataset.caseKind === 'real_observations'
      ? {en: ['data & maps', 'methods & parameter blocks', 'diagnostics & priors', 'posterior sampling', 'findings'], zh: ['数据与重建图', '方法与参数块', '诊断与先验', '后验采样', '发现与解释']}
      : labels;
    next.textContent = index === keys.length - 1 ? (lang === 'zh' ? '阅读完成' : 'Walkthrough complete') : `${lang === 'zh' ? '下一步' : 'Next'}: ${caseLabels[lang][index + 1]}`;
    article.querySelector('.step-position').textContent = `${index + 1} / ${keys.length}`;
  }

  function showCase(id) {
    if (!articles.some(article => article.id === id)) return;
    activeCase = id;
    if (design) design.hidden = true;
    designButton?.setAttribute('aria-pressed', 'false');
    articles.forEach(article => { article.hidden = article.id !== id; });
    caseButtons.forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.selectCase === id));
    });
  }

  articles.forEach(article => {
    showStep(article, 0);
    article.querySelectorAll('[data-step]').forEach(button => {
      button.addEventListener('click', () => {
        location.hash = `case=${article.id}&step=${button.dataset.step}`;
        showStep(article, keys.indexOf(button.dataset.step));
      });
    });
    for (const [selector, direction] of [['.previous', -1], ['.next', 1]]) {
      article.querySelector(selector).addEventListener('click', () => {
        const index = Math.max(0, Math.min(keys.length - 1, steps.get(article.id) + direction));
        location.hash = `case=${article.id}&step=${keys[index]}`;
        showStep(article, index);
        article.querySelector(`[data-step="${keys[index]}"]`).focus();
      });
    }
    const filter = article.querySelector('[data-parameter-filter]');
    filter?.addEventListener('change', () => {
      article.querySelectorAll('[data-parameter]').forEach(parameter => {
        parameter.hidden = filter.value !== 'all' && parameter.dataset.parameter !== filter.value;
      });
    });
  });
  caseButtons.forEach(button => button.addEventListener('click', () => {
    location.hash = `case=${button.dataset.selectCase}&step=${activeStep}`;
    showCase(button.dataset.selectCase);
    showStep(document.getElementById(activeCase), keys.indexOf(activeStep));
    document.getElementById(button.dataset.selectCase).scrollIntoView({block: 'start', behavior: 'instant'});
  }));
  designButton?.addEventListener('click', () => {
    location.hash = 'design/methods';
    design?.scrollIntoView({ block: 'start', behavior: 'instant' });
  });
  function routeNotebook() {
    if (location.hash.startsWith('#design') && design) {
      design.hidden = false;
      articles.forEach(article => { article.hidden = true; });
      caseButtons.forEach(button => button.setAttribute('aria-pressed', 'false'));
      designButton?.setAttribute('aria-pressed', 'true');
      return;
    }
    const params = new URLSearchParams(location.hash.slice(1));
    showCase(params.get('case') || activeCase);
    const article = articles.find(item => item.id === activeCase);
    const requested = params.get('step');
    const index = keys.indexOf(aliases[requested] || requested);
    if (article) showStep(article, index >= 0 ? index : keys.indexOf(activeStep));
  }
  // Native buttons support Tab/Enter; arrow keys also move between adjacent steps.
  document.querySelectorAll('.steps').forEach(nav => nav.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const article = nav.closest('[data-case]');
    const focused = event.target.closest('[data-step]');
    const current = focused ? keys.indexOf(focused.dataset.step) : steps.get(article.id);
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? keys.length - 1 :
      Math.max(0, Math.min(keys.length - 1, current + (event.key === 'ArrowRight' ? 1 : -1)));
    location.hash = `case=${article.id}&step=${keys[index]}`;
    showStep(article, index);
    nav.querySelector(`[data-step="${keys[index]}"]`).focus();
  }));
  showCase(activeCase);
  routeNotebook();
  window.addEventListener('hashchange', routeNotebook);
  window.addEventListener('notebooklanguagechange', () => articles.forEach(article => showStep(article, steps.get(article.id))));
  document.documentElement.classList.add('js');
})();
