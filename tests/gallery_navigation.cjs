// Exercise the shipped controller with a small DOM fixture; no browser/network.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function element(dataset = {}) {
  return {
    dataset, handlers: {}, attributes: {},
    addEventListener(name, fn) { this.handlers[name] = fn; },
    setAttribute(name, value) { this.attributes[name] = value; },
    getAttribute(name) { return name === 'href' ? this.href : this.attributes[name]; },
    click() { this.handlers.click(); },
    focus() {}, scrollIntoView() {},
  };
}
const keys = ['model', 'methods', 'diagnostics', 'sampling', 'recovery'];
const articles = ['first', 'second'].map(id => {
  const article = element();
  article.id = id;
  article.stages = keys.map(stage => element({stage}));
  article.buttons = keys.map(step => element({step}));
  article.controls = Object.fromEntries(['.previous', '.next', '.step-position'].map(k => [k, element()]));
  article.querySelectorAll = selector => selector === '[data-stage]' ? article.stages :
    selector === '[data-step]' ? article.buttons : [];
  article.querySelector = selector => article.controls[selector] || null;
  return article;
});
const essay = element({readingLayout: 'continuous'});
essay.id = 'essay';
essay.stages = [...keys, 'distributions', 'campbell', 'priors'].map(stage => {
  const node = element({stage}); node.id = `essay-${stage}`;
  node.scrollIntoView = () => { node.scrolled = true; };
  return node;
});
essay.querySelectorAll = selector => selector === '[data-stage]' ? essay.stages : [];
essay.querySelector = () => null; // Continuous articles have no step controls.
essay.contains = node => essay.stages.includes(node);
articles.push(essay);
const buttons = articles.map(a => element({selectCase: a.id}));
const counterpart = element();
counterpart.href = '../mild-prior/index.html?lang=zh#case=second&step=methods';
const deliberateLink = element();
deliberateLink.href = '#case=second&step=diagnostics';
const location = new URL('http://localhost/index.html?lang=zh#case=first&step=recovery');
const window = element();
const document = {
  documentElement: {dataset: {guideLanguage: 'zh'}, classList: {add() {}}},
  querySelectorAll(selector) {
    return {'[data-case]': articles, '[data-select-case]': buttons,
      'a[data-preserve-step]': [counterpart]}[selector] || [];
  },
  querySelector() { return null; },
  getElementById(id) { return articles.find(a => a.id === id) || essay.stages.find(s => s.id === id); },
};
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), {document, window, location, URL, URLSearchParams});
function check(id, step) {
  const current = articles.find(a => a.id === id);
  assert.equal(current.hidden, false);
  assert.deepEqual(current.stages.filter(s => !s.hidden).map(s => s.dataset.stage), [step]);
  assert.equal(new URL(counterpart.href).hash, `#case=second&step=${step}`);
  assert.equal(new URL(counterpart.href).searchParams.get('lang'), 'zh');
  assert.equal(deliberateLink.href, '#case=second&step=diagnostics');
}
check('first', 'recovery');
buttons[1].click();
check('second', 'recovery');
assert.equal(location.hash, '#case=second&step=recovery');
articles[1].buttons[1].click();
buttons[0].click();
check('first', 'methods');
// Explicit deep links / history override the carried step.
location.hash = '#case=second&step=sampling';
window.handlers.hashchange();
check('second', 'sampling');
window.handlers.notebooklanguagechange();
check('second', 'sampling');
location.hash = '#case=first&step=inference';
window.handlers.hashchange();
check('first', 'sampling');
location.hash = '#case=essay&section=priors';
window.handlers.hashchange();
assert.equal(essay.hidden, false);
assert.ok(essay.stages.every(stage => !stage.hidden));
assert.ok(essay.stages.find(stage => stage.dataset.stage === 'priors').scrolled);
window.handlers.notebooklanguagechange();
assert.ok(essay.stages.every(stage => !stage.hidden));
location.hash = '#case=essay&step=recovery';
window.handlers.hashchange();
assert.ok(essay.stages.find(stage => stage.dataset.stage === 'recovery').scrolled);
buttons[0].click();
check('first', 'recovery');
console.log('Gallery step and continuous-essay navigation passed');

// Execute the actual language controller: English by default, explicit Chinese retained.
const path = require('node:path');
const languageController = fs.readFileSync(path.join(path.dirname(process.argv[2]), 'methodology.js'), 'utf8');
for (const query of ['', '?lang=zh', '?lang=invalid']) {
  const languageLocation = new URL(`http://localhost/index.html${query}#case=first&step=recovery`);
  const guide = element();
  guide.querySelectorAll = () => [];
  guide.querySelector = () => element();
  const english = element({languageChoice: 'en'});
  const chinese = element({languageChoice: 'zh'});
  const link = element();
  link.href = '../proposals/index.html#case=first&step=recovery';
  const languageDocument = {
    documentElement: {dataset: {}},
    getElementById: () => guide,
    querySelectorAll: selector => selector === '[data-language-choice]' ? [english, chinese] :
      selector === 'a[data-gallery-link]' ? [link] : [],
  };
  const languageWindow = element();
  languageWindow.dispatchEvent = () => {};
  vm.runInNewContext(languageController, {
    document: languageDocument, window: languageWindow, location: languageLocation,
    URL, Event: class {}, history: {replaceState(_, __, url) {languageLocation.href = url.href;}},
  });
  assert.equal(languageDocument.documentElement.dataset.guideLanguage, query === '?lang=zh' ? 'zh' : 'en');
  chinese.click();
  assert.equal(languageDocument.documentElement.lang, 'zh-CN');
  assert.equal(new URL(link.href).searchParams.get('lang'), 'zh');
  assert.equal(languageLocation.hash, '#case=first&step=recovery');
  english.click();
  assert.equal(languageDocument.documentElement.lang, 'en');
}
console.log('Default English and explicit language selection passed');
