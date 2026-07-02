/* =========================================================================
   Neural Proofs — from-scratch ML + Canvas 2D. No external ML libraries.
   Chart.js is used only for the S4 bar chart.
   ========================================================================= */

/* ---------- seeded RNG (mulberry32) so results are reproducible ---------- */
function makeRng(seed){
  let a = seed >>> 0;
  return function(){
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng){ // Box–Muller
  let u=0,v=0; while(u===0)u=rng(); while(v===0)v=rng();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}

/* ---------- activations ---------- */
const relu = z => z>0?z:0;
const drelu = z => z>0?1:0;
const sigmoid = z => 1/(1+Math.exp(-z));

/* ---------- a tiny fully-connected net with manual backprop ----------
   layers: [{in,out,act}]  act ∈ 'relu' | 'linear' | 'sigmoid'
   Binary classifier: final layer has out=1 & act='sigmoid', trained with BCE. */
function makeNet(shape, rng, seed){
  const r = rng || makeRng(seed||12345);
  const layers = [];
  for(let i=0;i<shape.length;i++){
    const {inp,out,act} = shape[i];
    const scale = Math.sqrt(2/inp);
    const W = Array.from({length:out},()=>Array.from({length:inp},()=>gauss(r)*scale));
    const b = Array.from({length:out},()=>0);
    layers.push({W,b,act,inp,out});
  }
  return {layers};
}
function forward(net, x){
  const acts=[x]; const zs=[];
  let a=x;
  for(const L of net.layers){
    const z=new Array(L.out), o=new Array(L.out);
    for(let j=0;j<L.out;j++){
      let s=L.b[j], Wj=L.W[j];
      for(let k=0;k<L.inp;k++) s+=Wj[k]*a[k];
      z[j]=s;
      o[j]= L.act==='relu'?relu(s): L.act==='sigmoid'?sigmoid(s): s;
    }
    zs.push(z); acts.push(o); a=o;
  }
  return {acts,zs,out:a};
}
// one SGD step on a single (x,y) sample, y scalar 0/1. Returns loss.
function trainStep(net, x, y, lr){
  const {acts,zs,out}=forward(net,x);
  const L=net.layers.length;
  // output delta (sigmoid + BCE): dL/dz = a - y
  let delta=[out[0]-y];
  const grads=[];
  for(let li=L-1; li>=0; li--){
    const layer=net.layers[li];
    const aPrev=acts[li];
    const dW=Array.from({length:layer.out},()=>new Array(layer.inp).fill(0));
    const db=new Array(layer.out).fill(0);
    for(let j=0;j<layer.out;j++){
      db[j]=delta[j];
      for(let k=0;k<layer.inp;k++) dW[j][k]=delta[j]*aPrev[k];
    }
    grads[li]={dW,db};
    if(li>0){
      const prev=net.layers[li-1];
      const newDelta=new Array(prev.out).fill(0);
      for(let k=0;k<prev.out;k++){
        let s=0; for(let j=0;j<layer.out;j++) s+=layer.W[j][k]*delta[j];
        // multiply by activation derivative of previous layer's output
        const z=zs[li-1][k];
        newDelta[k]= prev.act==='relu'? s*drelu(z): prev.act==='sigmoid'? s*(sigmoid(z)*(1-sigmoid(z))): s;
      }
      delta=newDelta;
    }
  }
  for(let li=0;li<L;li++){
    const layer=net.layers[li], g=grads[li];
    for(let j=0;j<layer.out;j++){
      layer.b[j]-=lr*g.db[j];
      for(let k=0;k<layer.inp;k++) layer.W[j][k]-=lr*g.dW[j][k];
    }
  }
  const p=Math.min(Math.max(out[0],1e-7),1-1e-7);
  return -(y*Math.log(p)+(1-y)*Math.log(1-p));
}
function predict(net,x){ return forward(net,x).out[0]; }
function accuracy(net, X, Y){
  let c=0; for(let i=0;i<X.length;i++) if((predict(net,X[i])>=0.5?1:0)===Y[i]) c++;
  return c/X.length;
}

/* ---------- data: two concentric noisy rings ---------- */
function makeRings(n, rng){
  const X=[],Y=[];
  for(let i=0;i<n;i++){
    const cls=i%2;                       // 0 inner, 1 outer
    const radius=(cls===0?0.9:2.0)+gauss(rng)*0.18;
    const ang=rng()*Math.PI*2;
    X.push([Math.cos(ang)*radius, Math.sin(ang)*radius]);
    Y.push(cls);
  }
  return {X,Y};
}

/* ---------- canvas helpers ---------- */
function fitCanvas(cv){ // handle HiDPI
  const dpr=Math.min(window.devicePixelRatio||1,2);
  const cssW=cv.clientWidth||cv.width, cssH=cv.clientHeight||cv.height;
  if(cv.width!==Math.round(cssW*dpr)){cv.width=Math.round(cssW*dpr); cv.height=Math.round(cssH*dpr);}
  const ctx=cv.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,W:cssW,H:cssH};
}
// draw a decision boundary for a binary net over input range [-R,R]
function drawBoundary(cv, net, X, Y, R){
  const {ctx,W,H}=fitCanvas(cv);
  ctx.clearRect(0,0,W,H);
  const step=5;
  const toPx=(x,y)=>[ (x+R)/(2*R)*W, H-(y+R)/(2*R)*H ];
  for(let px=0;px<W;px+=step){
    for(let py=0;py<H;py+=step){
      const x=(px/W)*2*R-R, y=(1-py/H)*2*R-R;
      const p=predict(net,[x,y]);
      // red region for class 0, green region for class 1, blended by probability (on white)
      const t=p;
      const r=Math.round(220+(22-220)*t), g=Math.round(38+(163-38)*t), b=Math.round(38+(74-38)*t);
      ctx.fillStyle=`rgba(${r},${g},${b},0.42)`;
      ctx.fillRect(px,py,step,step);
    }
  }
  // data points on top of the decision surface
  for(let i=0;i<X.length;i++){
    const [sx,sy]=toPx(X[i][0],X[i][1]);
    ctx.beginPath(); ctx.arc(sx,sy,3.2,0,Math.PI*2);
    ctx.fillStyle= Y[i]===1? '#16a34a':'#dc2626';
    ctx.strokeStyle='rgba(255,255,255,.95)'; ctx.lineWidth=1.4;
    ctx.fill(); ctx.stroke();
  }
}

/* ============ S1: activations ============ */
function runS1(){
  const status=document.getElementById('status-s1');
  const rng=makeRng(7);
  const {X,Y}=makeRings(300,rng);
  const split=Math.floor(X.length*0.7);
  const Xtr=X.slice(0,split),Ytr=Y.slice(0,split),Xte=X.slice(split),Yte=Y.slice(split);
  const R=2.8;
  const lin=makeNet([{inp:2,out:1,act:'sigmoid'}], null, 11);
  const relN=makeNet([{inp:2,out:16,act:'relu'},{inp:16,out:1,act:'sigmoid'}], null, 23);
  const cvL=document.getElementById('canvas-s1-linear'), cvR=document.getElementById('canvas-s1-relu');
  let epoch=0; const EPOCHS=60, lr=0.08;
  function chunk(){
    for(let e=0;e<3 && epoch<EPOCHS;e++,epoch++){
      const order=[...Array(Xtr.length).keys()].sort(()=>0.5-rng());
      for(const i of order){ trainStep(lin,Xtr[i],Ytr[i],lr); trainStep(relN,Xtr[i],Ytr[i],lr); }
    }
    drawBoundary(cvL,lin,X,Y,R); drawBoundary(cvR,relN,X,Y,R);
    document.getElementById('acc-s1-linear').textContent=(accuracy(lin,Xte,Yte)*100).toFixed(0)+'%';
    document.getElementById('acc-s1-relu').textContent=(accuracy(relN,Xte,Yte)*100).toFixed(0)+'%';
    status.innerHTML=`training… epoch <span class="k">${epoch}/${EPOCHS}</span>`;
    if(epoch<EPOCHS) requestAnimationFrame(chunk);
    else {
      const la=(accuracy(lin,Xte,Yte)*100).toFixed(0), ra=(accuracy(relN,Xte,Yte)*100).toFixed(0);
      status.innerHTML=`done · linear <span class="k">${la}%</span> vs ReLU <span class="k">${ra}%</span>`;
      setTakeaway('takeaway-s1',`The linear model can only draw a straight line, so it stalls near chance (<b>${la}%</b>) — a line can't split two nested rings. Swapping in a single ReLU hidden layer lets the network bend space and wrap the ring to <b>${ra}%</b>. Only the activation changed.`);
    }
  }
  chunk();
}

/* ============ S2: depth collapse ============ */
function matToStr(M){ return M.map(row=>'['+row.map(v=>v.toFixed(2).padStart(6)).join(', ')+']').join('\n'); }
function matmul(A,B){ // A:m×k , B:k×n
  const m=A.length,k=B.length,n=B[0].length,C=Array.from({length:m},()=>new Array(n).fill(0));
  for(let i=0;i<m;i++)for(let j=0;j<n;j++){let s=0;for(let t=0;t<k;t++)s+=A[i][t]*B[t][j];C[i][j]=s;}
  return C;
}
function runS2(){
  const status=document.getElementById('status-s2');
  const rng=makeRng(3);
  const {X,Y}=makeRings(300,rng);
  const split=Math.floor(X.length*0.7);
  const Xtr=X.slice(0,split),Ytr=Y.slice(0,split),Xte=X.slice(split),Yte=Y.slice(split);
  const R=2.8;
  const net1=makeNet([{inp:2,out:1,act:'sigmoid'}],null,5);
  const net5=makeNet([{inp:2,out:4,act:'linear'},{inp:4,out:4,act:'linear'},{inp:4,out:4,act:'linear'},{inp:4,out:4,act:'linear'},{inp:4,out:1,act:'sigmoid'}],null,6);
  const net5r=makeNet([{inp:2,out:8,act:'relu'},{inp:8,out:8,act:'relu'},{inp:8,out:8,act:'relu'},{inp:8,out:8,act:'relu'},{inp:8,out:1,act:'sigmoid'}],null,9);
  const cv1=document.getElementById('canvas-s2-1'),cv5=document.getElementById('canvas-s2-5'),cv5r=document.getElementById('canvas-s2-relu');
  let epoch=0; const EPOCHS=70, lr=0.05;
  function chunk(){
    for(let e=0;e<3 && epoch<EPOCHS;e++,epoch++){
      const order=[...Array(Xtr.length).keys()].sort(()=>0.5-rng());
      for(const i of order){ trainStep(net1,Xtr[i],Ytr[i],lr); trainStep(net5,Xtr[i],Ytr[i],lr*0.6); trainStep(net5r,Xtr[i],Ytr[i],lr); }
    }
    drawBoundary(cv1,net1,X,Y,R); drawBoundary(cv5,net5,X,Y,R); drawBoundary(cv5r,net5r,X,Y,R);
    document.getElementById('acc-s2-1').textContent=(accuracy(net1,Xte,Yte)*100).toFixed(0)+'%';
    document.getElementById('acc-s2-5').textContent=(accuracy(net5,Xte,Yte)*100).toFixed(0)+'%';
    document.getElementById('acc-s2-relu').textContent=(accuracy(net5r,Xte,Yte)*100).toFixed(0)+'%';
    status.innerHTML=`training… epoch <span class="k">${epoch}/${EPOCHS}</span>`;
    if(epoch<EPOCHS) requestAnimationFrame(chunk);
    else {
      status.innerHTML='done';
      showMatrixProof();
      const a1=(accuracy(net1,Xte,Yte)*100).toFixed(0), a5=(accuracy(net5,Xte,Yte)*100).toFixed(0), ar=(accuracy(net5r,Xte,Yte)*100).toFixed(0);
      setTakeaway('takeaway-s2',`The 1-layer (<b>${a1}%</b>) and 5-linear-layer (<b>${a5}%</b>) nets perform the same because stacking linear layers just makes another linear map — the five weight matrices below multiply into one. Inserting ReLUs between the exact same layers is what unlocks depth, reaching <b>${ar}%</b>.`);
    }
  }
  // numeric proof: product of 5 random linear weight matrices collapses to one
  function showMatrixProof(){
    const r=makeRng(42);
    const dims=[2,3,3,3,3,2];
    const Ws=[];
    for(let i=0;i<5;i++){
      const out=dims[i+1],inp=dims[i];
      Ws.push(Array.from({length:out},()=>Array.from({length:inp},()=>gauss(r)*0.8)));
    }
    let prod=Ws[0];
    for(let i=1;i<5;i++) prod=matmul(Ws[i],prod);
    const el=document.getElementById('s2-matrices');
    el.innerHTML=
      `<div class="matrix"><div style="color:var(--dim);margin-bottom:6px">W5·W4·W3·W2·W1  (five ${'linear'} layers)</div>${matToStr(prod)}</div>`+
      `<div class="op">≡</div>`+
      `<div class="matrix"><div style="color:var(--dim);margin-bottom:6px">a single 2→2 linear layer</div>${matToStr(prod)}</div>`;
  }
  chunk();
}

/* ============ S3: embeddings from next-token ============ */
function runS3(){
  const status=document.getElementById('status-s3');
  // vocab grouped by category; grammar cycles: animal→verb→fruit→animal
  const cats={animal:['cat','dog','cow','fox'],verb:['eat','chase','see','hunt'],fruit:['apple','mango','plum','fig']};
  const vocab=[...cats.animal,...cats.verb,...cats.fruit];
  const idx=Object.fromEntries(vocab.map((w,i)=>[w,i]));
  const catOf=w=> cats.animal.includes(w)?'animal':cats.verb.includes(w)?'verb':'fruit';
  const nextGroup={animal:'verb',verb:'fruit',fruit:'animal'};
  const V=vocab.length, D=8;
  const rng=makeRng(17);
  // embedding table E[V][D] and output projection P[D][V]
  const E=Array.from({length:V},()=>Array.from({length:D},()=>gauss(rng)*0.3));
  const P=Array.from({length:D},()=>Array.from({length:V},()=>gauss(rng)*0.3));
  // build (cur -> next) training pairs: next is a random token of the next category
  const pairs=[];
  for(let rep=0;rep<40;rep++) for(const w of vocab){
    const ng=cats[nextGroup[catOf(w)]];
    const nx=ng[Math.floor(rng()*ng.length)];
    pairs.push([idx[w],idx[nx]]);
  }
  const lr=0.15; let epoch=0; const EPOCHS=120;
  function softmax(v){ const m=Math.max(...v); const ex=v.map(z=>Math.exp(z-m)); const s=ex.reduce((a,b)=>a+b,0); return ex.map(z=>z/s); }
  function step(cur,nxt){
    const e=E[cur];
    const logits=new Array(V).fill(0);
    for(let j=0;j<V;j++){let s=0;for(let d=0;d<D;d++)s+=e[d]*P[d][j];logits[j]=s;}
    const p=softmax(logits);
    const dlog=p.slice(); dlog[nxt]-=1;                 // softmax+CE grad
    const de=new Array(D).fill(0);
    for(let d=0;d<D;d++){
      let g=0; for(let j=0;j<V;j++){ g+=dlog[j]*E[cur][d]; P[d][j]-=lr*dlog[j]*e[d]; }
      // grad wrt embedding
      let ge=0; for(let j=0;j<V;j++) ge+=dlog[j]*P[d][j];
      de[d]=ge;
    }
    for(let d=0;d<D;d++) E[cur][d]-=lr*de[d];
  }
  function chunk(){
    for(let e=0;e<4 && epoch<EPOCHS;e++,epoch++){
      const order=[...pairs].sort(()=>0.5-rng());
      for(const [c,n] of order) step(c,n);
    }
    status.innerHTML=`training… epoch <span class="k">${epoch}/${EPOCHS}</span>`;
    if(epoch<EPOCHS) requestAnimationFrame(chunk);
    else {
      status.innerHTML='done · projecting with PCA';
      drawEmbeddings(); showNeighbors();
      setTakeaway('takeaway-s3',`The loss function never mentioned similarity — the only signal was "which token comes next". Yet animals, fruits and verbs each collapsed into their own cluster, and every token's nearest neighbours share its category. Meaning emerged as a by-product of next-token prediction.`);
    }
  }
  // 2-component PCA via power iteration on covariance of embedding rows
  function pca2(M){
    const n=M.length,d=M[0].length;
    const mean=new Array(d).fill(0);
    for(const row of M) for(let j=0;j<d;j++) mean[j]+=row[j]/n;
    const C=Array.from({length:d},()=>new Array(d).fill(0));
    for(const row of M) for(let a=0;a<d;a++)for(let b=0;b<d;b++) C[a][b]+=(row[a]-mean[a])*(row[b]-mean[b])/n;
    function topEig(C){
      let v=new Array(d).fill(0).map(()=>Math.random?0.5:0.5); v=v.map((_,i)=>Math.sin(i+1));
      for(let it=0;it<200;it++){
        const nv=new Array(d).fill(0);
        for(let a=0;a<d;a++)for(let b=0;b<d;b++) nv[a]+=C[a][b]*v[b];
        const norm=Math.sqrt(nv.reduce((s,x)=>s+x*x,0))||1; v=nv.map(x=>x/norm);
      } return v;
    }
    const pc1=topEig(C);
    // deflate
    const C2=C.map((row,a)=>row.map((val,b)=>val-  // remove pc1 component
      (()=>{let lam=0;for(let i=0;i<d;i++)for(let j=0;j<d;j++)lam+=pc1[i]*C[i][j]*pc1[j];return lam;})()*pc1[a]*pc1[b]));
    const pc2=topEig(C2);
    return M.map(row=>{
      let x=0,y=0; for(let j=0;j<d;j++){x+=(row[j]-mean[j])*pc1[j]; y+=(row[j]-mean[j])*pc2[j];}
      return [x,y];
    });
  }
  const cv=document.getElementById('canvas-s3');
  function drawEmbeddings(){
    const pts=pca2(E);
    const {ctx,W,H}=fitCanvas(cv); ctx.clearRect(0,0,W,H);
    let minX=1e9,maxX=-1e9,minY=1e9,maxY=-1e9;
    for(const[x,y]of pts){minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);}
    const pad=48;
    const sx=x=>pad+(x-minX)/((maxX-minX)||1)*(W-2*pad);
    const sy=y=>H-pad-(y-minY)/((maxY-minY)||1)*(H-2*pad);
    const col={animal:'#dc2626',fruit:'#e0a400',verb:'#16a34a'};
    // subtle grid
    ctx.strokeStyle='rgba(150,160,175,.28)';ctx.lineWidth=1;
    for(let g=0;g<=4;g++){const gx=pad+g/4*(W-2*pad);ctx.beginPath();ctx.moveTo(gx,pad);ctx.lineTo(gx,H-pad);ctx.stroke();
      const gy=pad+g/4*(H-2*pad);ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}
    for(let i=0;i<vocab.length;i++){
      const c=col[catOf(vocab[i])];
      const x=sx(pts[i][0]),y=sy(pts[i][1]);
      ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fillStyle=c;ctx.shadowColor=c;ctx.shadowBlur=10;ctx.fill();ctx.shadowBlur=0;
      ctx.strokeStyle='rgba(255,255,255,.9)';ctx.lineWidth=1.4;ctx.stroke();
      ctx.fillStyle='#161a20';ctx.font='600 12px JetBrains Mono, monospace';ctx.fillText(vocab[i],x+11,y+4);
    }
    window._s3={E,vocab,catOf};
  }
  function cos(a,b){let d=0,na=0,nb=0;for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return d/(Math.sqrt(na*nb)||1);}
  function showNeighbors(){
    const el=document.getElementById('s3-neighbors');
    const picks=['cat','apple','chase'];
    el.innerHTML=picks.map(w=>{
      const wi=idx[w];
      const sims=vocab.map((v,i)=>[v,cos(E[wi],E[i])]).filter(([v])=>v!==w).sort((a,b)=>b[1]-a[1]).slice(0,3);
      const col={animal:'#dc2626',fruit:'#e0a400',verb:'#16a34a'}[catOf(w)];
      return `<div><span style="color:${col}">${w}</span> → `+sims.map(([v,s])=>`<span style="color:var(--text)">${v}</span> <span style="color:var(--dim)">${s.toFixed(2)}</span>`).join(', ')+`</div>`;
    }).join('');
  }
  chunk();
}

/* ============ S4: generalization ============ */
let s4State={done:false, surfaces:{}, metrics:{}, sizeKeys:[20,200,2000], cur:20};
function makeSplit(n, rng){
  // label = 1 inside a wavy circle, 0 outside, plus label noise
  const gen=m=>{const X=[],Y=[];for(let i=0;i<m;i++){
    const x=(rng()*2-1)*2, y=(rng()*2-1)*2;
    const r=Math.sqrt(x*x+y*y); const boundary=1.1+0.25*Math.sin(3*Math.atan2(y,x));
    let lab=r<boundary?1:0; if(rng()<0.08) lab=1-lab;    // 8% label noise
    X.push([x,y]);Y.push(lab);
  } return {X,Y};};
  const tr=gen(n), te=gen(400);
  return {Xtr:tr.X,Ytr:tr.Y,Xte:te.X,Yte:te.Y};
}
function trainFull(net,X,Y,epochs,lr,rng){
  for(let e=0;e<epochs;e++){
    const order=[...Array(X.length).keys()].sort(()=>0.5-rng());
    for(const i of order) trainStep(net,X[i],Y[i],lr);
  }
}
// chunked training: spreads epochs across animation frames so large datasets never freeze the page
function trainChunked(net,X,Y,epochs,lr,rng,onProgress){
  return new Promise(resolve=>{
    let e=0;
    const epochsPerFrame=Math.max(1,Math.floor(30000/Math.max(1,X.length))); // ~constant work per frame
    function frame(){
      const target=Math.min(epochs,e+epochsPerFrame);
      for(;e<target;e++){
        const order=[...Array(X.length).keys()].sort(()=>0.5-rng());
        for(const i of order) trainStep(net,X[i],Y[i],lr);
      }
      if(onProgress) onProgress(e/epochs);
      if(e<epochs) requestAnimationFrame(frame); else resolve();
    }
    frame();
  });
}
// reveal a section's "what just happened" explainer
function setTakeaway(id,html){ const el=document.getElementById(id); if(!el)return; el.innerHTML=html; el.classList.add('show'); }
let s4Chart=null;
function runS4(){
  const status=document.getElementById('status-s4');
  status.textContent='training three models…';
  const sizes=[20,200,2000];
  const trainAcc=[],testAcc=[],gap=[];
  const rng=makeRng(99);
  // run sequentially across frames so UI stays alive
  let si=0;
  function nextSize(){
    if(si>=sizes.length){ finish(); return; }
    const n=sizes[si];
    const {Xtr,Ytr,Xte,Yte}=makeSplit(n,rng);
    const net=makeNet([{inp:2,out:32,act:'relu'},{inp:32,out:32,act:'relu'},{inp:32,out:1,act:'sigmoid'}],null,100+si);
    const epochs= n<=20?260 : n<=200?90 : 26;
    trainFull(net,Xtr,Ytr,epochs,0.05,rng);
    const ta=accuracy(net,Xtr,Ytr), va=accuracy(net,Xte,Yte);
    trainAcc.push(+(ta*100).toFixed(1)); testAcc.push(+(va*100).toFixed(1)); gap.push(+((ta-va)*100).toFixed(1));
    s4State.surfaces[n]=buildSurface(net);
    s4State.metrics[n]={train:+(ta*100).toFixed(1),test:+(va*100).toFixed(1),gap:+((ta-va)*100).toFixed(1)};
    status.innerHTML=`trained <span class="k">n=${n}</span> · gap ${((ta-va)*100).toFixed(0)}%`;
    si++; requestAnimationFrame(nextSize);
  }
  function finish(){
    drawBars(sizes,trainAcc,testAcc);
    drawGap(sizes,gap);
    s4State.done=true; s4State.cur=20; drawSurface(); renderSurfaceInfo(20);
    document.getElementById('s4-n').value=20;
    status.innerHTML=`done · gap collapses from <span class="k">${gap[0]}%</span> → <span class="k">${gap[2]}%</span>`;
    setTakeaway('takeaway-s4',`Same over-parameterized model, three dataset sizes. With <b>n=20</b> it hits ${trainAcc[0]}% on training data but only ${testAcc[0]}% on held-out — it memorized, leaving a <b>${gap[0]}%</b> gap. As the data grows, that gap collapses to <b>${gap[2]}%</b> at n=2000. Data is what forces real generalization. Try your own size below.`);
  }
  nextSize();
}
// choose training length so small sets overfit and large sets converge in reasonable time
function epochsFor(n){ return n<=30?260 : n<=100?150 : n<=300?85 : n<=800?45 : n<=2000?26 : 16; }
// train ONE model at a user-chosen dataset size and show its surface + metrics
function trainCustomSurface(){
  const input=document.getElementById('s4-n');
  let n=Math.round(+input.value);
  if(!isFinite(n)||n<2) n=2;
  n=Math.max(2,Math.min(8000,n));
  input.value=n;
  const status=document.getElementById('status-s4');
  const btn=document.getElementById('s4-custom');
  const pbar=document.getElementById('s4-pbar'), pfill=pbar.firstElementChild;
  btn.disabled=true; pbar.classList.add('show'); pfill.style.width='0%';
  status.innerHTML=`training <span class="k">n=${n}</span> …`;
  const rng=makeRng(1000+n);
  const {Xtr,Ytr,Xte,Yte}=makeSplit(n,rng);
  const net=makeNet([{inp:2,out:32,act:'relu'},{inp:32,out:32,act:'relu'},{inp:32,out:1,act:'sigmoid'}],null,7000+(n%997));
  // chunked so even n=8000 trains without freezing the page
  trainChunked(net,Xtr,Ytr,epochsFor(n),0.05,rng,p=>{
    pfill.style.width=(p*100).toFixed(0)+'%';
    status.innerHTML=`training <span class="k">n=${n}</span> · ${(p*100).toFixed(0)}%`;
  }).then(()=>{
    const ta=accuracy(net,Xtr,Ytr), va=accuracy(net,Xte,Yte), gapv=+((ta-va)*100).toFixed(1);
    s4State.surfaces[n]=buildSurface(net);
    s4State.metrics[n]={train:+(ta*100).toFixed(1),test:+(va*100).toFixed(1),gap:gapv};
    s4State.done=true; s4State.cur=n;
    drawSurface(); renderSurfaceInfo(n);
    setTakeaway('takeaway-s4', gapv>=15
      ? `At <b>n=${n}</b> the model memorizes: ${(ta*100).toFixed(0)}% on training data but only ${(va*100).toFixed(0)}% held-out — a <b>${gapv}%</b> gap. It's fitting individual points, not the true boundary. Raise n and watch the gap shrink.`
      : `At <b>n=${n}</b> the model generalizes well: ${(ta*100).toFixed(0)}% train vs ${(va*100).toFixed(0)}% held-out — only a <b>${gapv}%</b> gap. Enough data pushed it to learn the real boundary instead of memorizing.`);
    status.innerHTML=`done · <span class="k">n=${n}</span> · gap ${gapv}%`;
    pbar.classList.remove('show'); btn.disabled=false;
  });
}
// update the hint line + metric readout under the 3D surface for a given n
function renderSurfaceInfo(n){
  const m=s4State.metrics[n];
  const hint=document.getElementById('s4-3d-hint');
  const box=document.getElementById('s4-surface-metrics');
  if(!m){ hint.textContent='n = '+n; box.innerHTML=''; return; }
  const verdict = m.gap>=15?'overfitting — sharp spikes memorising individual points'
                : m.gap>=6 ?'partly generalising — the true boundary is emerging'
                :           'generalising — train and held-out accuracy agree';
  hint.textContent='n = '+n+' · '+verdict;
  const gapColor = m.gap>=15?'var(--red)':m.gap>=6?'var(--yellow)':'var(--green)';
  box.innerHTML=
    `<div class="m">dataset size<b style="color:var(--text)">${n}</b></div>`+
    `<div class="m">train acc<b style="color:var(--green)">${m.train}%</b></div>`+
    `<div class="m">held-out acc<b style="color:var(--red)">${m.test}%</b></div>`+
    `<div class="m">generalization gap<b style="color:${gapColor}">${m.gap}%</b></div>`;
}
function drawBars(sizes,tr,te){
  const ctx=document.getElementById('canvas-s4-bars').getContext('2d');
  if(s4Chart) s4Chart.destroy();
  s4Chart=new Chart(ctx,{type:'bar',data:{labels:sizes.map(s=>'n = '+s),
    datasets:[
      {label:'Train',data:tr,backgroundColor:'#16a34a',borderRadius:8,borderSkipped:false},
      {label:'Held-out',data:te,backgroundColor:'#dc2626',borderRadius:8,borderSkipped:false}
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#5c6470',font:{family:'JetBrains Mono'}}}},
      scales:{y:{min:0,max:100,ticks:{color:'#98a0ac',callback:v=>v+'%'},grid:{color:'#eceef2'}},
              x:{ticks:{color:'#5c6470',font:{family:'JetBrains Mono'}},grid:{display:false}}}}});
}
function drawGap(sizes,gap){
  const cv=document.getElementById('canvas-s4-gap');
  const {ctx,W,H}=fitCanvas(cv); ctx.clearRect(0,0,W,H);
  const pad=46;
  const xs=i=>pad+i/(sizes.length-1)*(W-2*pad);
  const ys=g=>H-pad-(g/100)*(H-2*pad);
  ctx.strokeStyle='#eceef2';ctx.lineWidth=1;
  for(let g=0;g<=100;g+=25){const y=ys(g);ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();
    ctx.fillStyle='#98a0ac';ctx.font='11px JetBrains Mono, monospace';ctx.fillText(g+'%',6,y+4);}
  // area
  ctx.beginPath();ctx.moveTo(xs(0),ys(gap[0]));
  for(let i=1;i<gap.length;i++)ctx.lineTo(xs(i),ys(gap[i]));
  ctx.lineTo(xs(gap.length-1),ys(0));ctx.lineTo(xs(0),ys(0));ctx.closePath();
  const grad=ctx.createLinearGradient(0,pad,0,H-pad);grad.addColorStop(0,'rgba(220,38,38,.30)');grad.addColorStop(1,'rgba(220,38,38,0)');
  ctx.fillStyle=grad;ctx.fill();
  ctx.beginPath();ctx.moveTo(xs(0),ys(gap[0]));
  for(let i=1;i<gap.length;i++)ctx.lineTo(xs(i),ys(gap[i]));
  ctx.strokeStyle='#dc2626';ctx.lineWidth=2.5;ctx.stroke();
  for(let i=0;i<gap.length;i++){ctx.beginPath();ctx.arc(xs(i),ys(gap[i]),5,0,Math.PI*2);ctx.fillStyle='#dc2626';ctx.fill();
    ctx.fillStyle='#161a20';ctx.font='600 12px JetBrains Mono, monospace';ctx.fillText(gap[i]+'%',xs(i)-10,ys(gap[i])-12);
    ctx.fillStyle='#5c6470';ctx.fillText('n='+sizes[i],xs(i)-14,H-pad+18);}
}
// sample a model's probability surface on a grid for the 3D plot
function buildSurface(net){
  const G=26, R=2, grid=[];
  for(let i=0;i<G;i++){const row=[];for(let j=0;j<G;j++){
    const x=(i/(G-1))*2*R-R, y=(j/(G-1))*2*R-R;
    row.push({x,y,h:predict(net,[x,y])});
  }grid.push(row);}
  return {G,R,grid};
}
// hand-rolled 3D isometric surface with drag-rotate + scroll-zoom
let s4view={yaw:-0.7,pitch:0.5,zoom:1};
function drawSurface(){
  const surf=s4State.surfaces[s4State.cur]; if(!surf) return;
  const cv=document.getElementById('canvas-s4-3d');
  const {ctx,W,H}=fitCanvas(cv); ctx.clearRect(0,0,W,H);
  const {G,grid}=surf;
  const cx=W/2, cy=H/2+40, scale=Math.min(W,H)*0.28*s4view.zoom;
  const cyaw=Math.cos(s4view.yaw),syaw=Math.sin(s4view.yaw),cp=Math.cos(s4view.pitch),sp=Math.sin(s4view.pitch);
  function project(x,y,z){
    // rotate around vertical (yaw) then tilt (pitch), orthographic
    let X=x*cyaw - y*syaw, Y=x*syaw + y*cyaw;
    let Z=z*1.4;
    let Yr=Y*cp - Z*sp, Zr=Y*sp + Z*cp;
    return [cx+X*scale, cy - Yr*scale - Zr*scale*0.0];
  }
  // build quads with depth for painter's sort
  const quads=[];
  for(let i=0;i<G-1;i++)for(let j=0;j<G-1;j++){
    const a=grid[i][j],b=grid[i+1][j],c=grid[i+1][j+1],d=grid[i][j+1];
    const hAvg=(a.h+b.h+c.h+d.h)/4;
    const depth=(a.x+b.x+c.x+d.x)/4*syaw + (a.y+b.y+c.y+d.y)/4*cyaw;
    quads.push({pts:[a,b,c,d],h:hAvg,depth});
  }
  quads.sort((p,q)=>p.depth-q.depth);
  const norm=(v,R)=>v/R;
  for(const q of quads){
    ctx.beginPath();
    q.pts.forEach((p,k)=>{const[sx,sy]=project(norm(p.x,2),norm(p.y,2),p.h);k===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);});
    ctx.closePath();
    const t=q.h; // colour by probability: red(0) → green(1)
    const r=Math.round(220+(22-220)*t),g=Math.round(38+(163-38)*t),b=Math.round(38+(74-38)*t);
    ctx.fillStyle=`rgba(${r},${g},${b},0.85)`;
    ctx.strokeStyle='rgba(255,255,255,.7)';ctx.lineWidth=0.6;
    ctx.fill();ctx.stroke();
  }
}
function bindSurfaceControls(){
  const cv=document.getElementById('canvas-s4-3d');
  let drag=false,lx=0,ly=0;
  cv.addEventListener('mousedown',e=>{drag=true;lx=e.clientX;ly=e.clientY;cv.style.cursor='grabbing';});
  window.addEventListener('mouseup',()=>{drag=false;cv.style.cursor='grab';});
  window.addEventListener('mousemove',e=>{if(!drag)return;
    s4view.yaw+=(e.clientX-lx)*0.01; s4view.pitch+=(e.clientY-ly)*0.006;
    s4view.pitch=Math.max(0.05,Math.min(1.3,s4view.pitch));
    lx=e.clientX;ly=e.clientY; drawSurface();});
  cv.addEventListener('wheel',e=>{e.preventDefault();s4view.zoom*=e.deltaY<0?1.08:0.93;
    s4view.zoom=Math.max(0.5,Math.min(2.4,s4view.zoom));drawSurface();},{passive:false});
}

/* ---------- hero canvas: drifting particle field ---------- */
function heroAnim(){
  const cv=document.getElementById('hero-canvas');
  const ctx=cv.getContext('2d');
  let W,H,pts;
  const rng=makeRng(1);
  const palette=['220,38,38','224,164,0','22,163,74']; // red, yellow, green
  function resize(){W=cv.width=cv.offsetWidth;H=cv.height=cv.offsetHeight;
    pts=Array.from({length:Math.min(70,Math.floor(W/16))},(_,i)=>({x:rng()*W,y:rng()*H,vx:(rng()-.5)*0.3,vy:(rng()-.5)*0.3,c:palette[i%3]}));}
  resize(); window.addEventListener('resize',resize);
  function frame(){
    ctx.clearRect(0,0,W,H);
    for(const p of pts){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;}
    for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++){
      const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.hypot(dx,dy);
      if(d<120){ctx.strokeStyle=`rgba(150,160,175,${(1-d/120)*0.22})`;ctx.lineWidth=0.7;
        ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.stroke();}
    }
    for(const p of pts){ctx.beginPath();ctx.arc(p.x,p.y,2,0,Math.PI*2);ctx.fillStyle=`rgba(${p.c},.85)`;ctx.fill();}
    requestAnimationFrame(frame);
  }
  frame();
}

/* ---------- wiring: run experiments when scrolled into view + buttons ---------- */
window.addEventListener('DOMContentLoaded',()=>{
  heroAnim();
  bindSurfaceControls();
  // each experiment runs ONLY when its button is pressed — canvases stay empty until then
  document.getElementById('run-s1').onclick=runS1;
  document.getElementById('run-s2').onclick=runS2;
  document.getElementById('run-s3').onclick=runS3;
  document.getElementById('run-s4').onclick=runS4;
  // one click runs every experiment on the page
  document.getElementById('run-all').onclick=()=>{ runS1(); runS2(); runS3(); runS4(); };
  // custom dataset size → train one model and show its surface + metrics
  document.getElementById('s4-custom').onclick=trainCustomSurface;
  document.getElementById('s4-n').addEventListener('keydown',e=>{ if(e.key==='Enter'){e.preventDefault();trainCustomSurface();} });
});
