// Exercise the actual local report using cached headless Chromium and CDP pipes.
// No npm install or browser server is required.
import {spawn} from 'node:child_process';
import {readFile, writeFile, mkdtemp} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const report=path.resolve(process.argv[2]||'results/pilot-v1/gallery.html');
const browser=process.env.KUROSAWA_CHROME||'/home/koushik/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell';
await readFile(report);
const chrome=spawn(browser,[
  '--headless','--no-sandbox','--disable-dev-shm-usage','--disable-gpu',
  '--disable-background-networking','--no-first-run','--remote-debugging-pipe',
  '--user-data-dir=/tmp/curtain-kurosawa-browser',
],{stdio:['ignore','ignore','pipe','pipe','pipe']});
let counter=0, buffer=Buffer.alloc(0), stderr='';
const pending=new Map(), exceptions=[];
chrome.stderr.on('data',b=>{stderr=(stderr+b.toString()).slice(-4000);});
function failPending(error){
  for(const p of pending.values()){clearTimeout(p.timer);p.reject(error);}
  pending.clear();
}
for(const stream of [chrome.stdio[3],chrome.stdio[4]])stream.on('error',e=>failPending(new Error(`${e.message}\n${stderr}`)));
chrome.on('error',e=>failPending(e));
chrome.on('exit',code=>failPending(new Error(`Chromium exited ${code}\n${stderr}`)));
chrome.stdio[4].on('data',chunk=>{
  buffer=Buffer.concat([buffer,chunk]);
  let at;
  while((at=buffer.indexOf(0))>=0){
    const raw=buffer.subarray(0,at).toString();buffer=buffer.subarray(at+1);
    if(!raw)continue;
    const m=JSON.parse(raw);
    if(m.method==='Runtime.exceptionThrown')exceptions.push(m.params.exceptionDetails);
    if(m.id&&pending.has(m.id)){
      const p=pending.get(m.id);pending.delete(m.id);clearTimeout(p.timer);
      m.error?p.reject(new Error(JSON.stringify(m.error))):p.resolve(m.result);
    }
  }
});
function send(method,params={},sessionId){
  return new Promise((resolve,reject)=>{
    const id=++counter;
    const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`CDP timeout: ${method}\n${stderr}`));},20000);
    pending.set(id,{resolve,reject,timer});
    chrome.stdio[3].write(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})})+'\0');
  });
}
try{
  const {targetId}=await send('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
  await send('Runtime.enable',{},sessionId);
  await send('Page.enable',{},sessionId);
  await send('Emulation.setDeviceMetricsOverride',{width:1280,height:1000,deviceScaleFactor:1,mobile:false},sessionId);
  await send('Page.navigate',{url:pathToFileURL(report).href},sessionId);
  async function evaluate(expression){
    const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);
    if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  }
  let ready=false;
  for(let i=0;i<40;i++){
    ready=await evaluate('document.readyState === "complete" && !!document.querySelector(".pair")');
    if(ready)break;
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  assert(ready,'Gallery did not render any pairs');
  const initial=await evaluate(`({cards:document.querySelectorAll('.pair').length,filmOptions:document.getElementById('film').options.length,targetOptions:document.getElementById('target').options.length,count:document.getElementById('count').textContent,intro:document.getElementById('intro').textContent})`);
  assert(initial.cards>0);
  const expected=JSON.parse(await readFile(path.join(path.dirname(report),'summary.json'),'utf8'));
  assert.equal(initial.filmOptions,expected.source_movies+1);
  assert.equal(initial.targetOptions,expected.target_movies+1);
  const reviewed=await evaluate(`(()=>{const button=document.querySelector('[data-mark="Useful lead"]');const k=button.dataset.pair;button.click();const input=document.querySelector('[data-note="'+k+'"]');input.value='Browser check: spatial arrangement';input.dispatchEvent(new Event('input',{bubbles:true}));return {mark:notes[k].mark,note:notes[k].note};})()`);
  assert.equal(reviewed.mark,'Useful lead');assert(reviewed.note.includes('spatial'));
  const filtered=await evaluate(`(()=>{const f=document.getElementById('film');f.value=records[data.shortlist[0].source].movie;f.dispatchEvent(new Event('change'));return {cards:document.querySelectorAll('.pair').length,valid:[...document.querySelectorAll('[data-pair]')].every(b=>records[Number(b.dataset.pair.split(':')[0])].movie===f.value)};})()`);
  assert(filtered.cards>0&&filtered.valid,'Source film filter failed');
  const targetFiltered=await evaluate(`(()=>{const f=document.getElementById('film');f.value='';const v=document.getElementById('view');v.value='films';const t=document.getElementById('target');t.value=records[data.film_matches[0].target].movie;t.dispatchEvent(new Event('change'));return {cards:document.querySelectorAll('.pair').length,valid:[...document.querySelectorAll('[data-pair]')].every(b=>records[Number(b.dataset.pair.split(':')[1])].movie===t.value)};})()`);
  assert(targetFiltered.cards>0&&targetFiltered.valid,'Target film filter failed');
  const downloadDir=await mkdtemp('/tmp/kurosawa-review-check-');
  await send('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloadDir});
  await evaluate(`document.getElementById('export').click()`);
  let exported;
  for(let i=0;i<50;i++){
    try{exported=JSON.parse(await readFile(path.join(downloadDir,'kurosawa-review.json'),'utf8'));break;}catch{}
    await new Promise(resolve=>setTimeout(resolve,100));
  }
  assert.equal(exported?.review.length,1,'Review export did not contain the marked pair');
  assert.equal(exported.review[0].mark,'Useful lead');
  assert(exported.review[0].note.includes('spatial'));
  await evaluate(`document.getElementById('target').value=''`);
  await evaluate(`(()=>{const f=document.getElementById('film');f.value='';f.dispatchEvent(new Event('change'));const v=document.getElementById('view');v.value='earlier';v.dispatchEvent(new Event('change'));})()`);
  const earlier=await evaluate(`({cards:document.querySelectorAll('.pair').length,chronology:[...document.querySelectorAll('[data-pair]')].every(b=>{const [q,t]=b.dataset.pair.split(':').map(Number);return records[t].year<records[q].year;})})`);
  assert(earlier.cards>0&&earlier.chronology,'Earlier-film view failed');
  await evaluate(`(()=>{for(const k of Object.keys(notes))delete notes[k];const v=document.getElementById('view');v.value='shortlist';v.dispatchEvent(new Event('change'));})()`);
  // Wait for the visible images before taking the screenshot.
  const images=await evaluate(`Promise.all([...document.querySelectorAll('img')].slice(0,2).map(im=>new Promise(resolve=>{if(im.complete)return resolve(im.naturalWidth>0);im.onload=()=>resolve(true);im.onerror=()=>resolve(false);})))`);
  assert(images.every(Boolean),'A visible frame image failed to load');
  assert.equal(exceptions.length,0,JSON.stringify(exceptions));
  const shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:false},sessionId);
  const screenshot=path.join(path.dirname(report),'gallery-preview.png');
  await writeFile(screenshot,Buffer.from(shot.data,'base64'));
  console.log(JSON.stringify({initial,reviewed,targetFiltered,exportedReviewPairs:exported.review.length,earlier,visibleImagesLoaded:images.length,exceptions:exceptions.length,screenshot},null,2));
} finally{
  try{await send('Browser.close');}catch{}
  chrome.kill();
}
